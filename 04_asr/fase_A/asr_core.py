# ---- nucleo del ASR: se prueba localmente y se copia tal cual al notebook ----
import hashlib, re, unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000
FUENTE_SK = "menendez_gomez_2025"


# ============================ TEXTO ============================
def normalizar(t):
    """Misma normalizacion que build_dataset_v1.py (text_norm): NFC, minusculas, sin puntuacion.
    Se aplica tambien a lo que predice Whisper, que tiende a poner mayusculas y puntos."""
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = re.sub(r"[¡!¿?.,;:…\"“”«»()\[\]{}—–\-_/\\]", " ", t)
    t = t.replace("'", " ").replace("’", " ")
    return re.sub(r"\s+", " ", t).strip()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================ SPLIT ============================
def hacer_split(meta, lineas_tts, seed, frac_dev, frac_test, frac_otros):
    """Particion fija para todos los experimentos de la tesis.

    - test_otros: grupos de hablante ENTEROS de youtube y workshop (voces que el modelo no oye).
    - dev / test_sk001: clips de SK001 cuyo texto NO aparece en el txt del TTS, para que los
      experimentos con audio sintetico no vean las frases de evaluacion.
    - Un texto de evaluacion nunca queda tambien en train (se saca de train: 'fuera_texto').
    """
    m = meta.sort_values("id").reset_index(drop=True).copy()
    m["split"] = "train"
    rng = np.random.RandomState(seed)

    # 1) hablantes no vistos
    for fuente in sorted(set(m.source) - {FUENTE_SK}):
        d = m[m.source == fuente].groupby("speaker_group").duration_s.sum()
        grupos = sorted(d.index)
        rng.shuffle(grupos)
        acum, elegidos = 0.0, []
        for g in grupos:
            if acum >= frac_otros * d.sum():
                break
            elegidos.append(g)
            acum += d[g]
        m.loc[m.speaker_group.isin(elegidos), "split"] = "test_otros"

    # 2) SK001: textos que no esten contenidos en ninguna linea del TTS
    blob = "\n".join(f" {normalizar(l)} " for l in lineas_tts if l.strip())
    sk = m[m.source == FUENTE_SK]
    elegibles = sorted(t for t in set(sk.text_norm) if f" {t} " not in blob)
    rng.shuffle(elegibles)
    n_test, n_dev = round(frac_test * len(sk)), round(frac_dev * len(sk))
    cuenta = sk.text_norm.value_counts()
    usados = {"test_sk001": 0, "dev": 0}
    for t in elegibles:
        destino = "test_sk001" if usados["test_sk001"] < n_test else "dev" if usados["dev"] < n_dev else None
        if destino is None:
            break
        m.loc[(m.source == FUENTE_SK) & (m.text_norm == t), "split"] = destino
        usados[destino] += int(cuenta[t])

    # 3) sin fuga por texto: evaluacion con texto en el TTS, o train con texto de evaluacion
    evaluacion = m.split != "train"
    en_tts = m.text_norm.map(lambda t: f" {t} " in blob)
    m.loc[evaluacion & en_tts, "split"] = "fuera_tts"
    textos_eval = set(m.loc[m.split.isin(["dev", "test_sk001", "test_otros"]), "text_norm"])
    m.loc[(m.split == "train") & m.text_norm.isin(textos_eval), "split"] = "fuera_texto"
    return m


def resumen_split(m):
    r = m.groupby("split").agg(clips=("id", "size"), minutos=("duration_s", lambda s: round(s.sum() / 60, 1)),
                               hablantes=("speaker_group", "nunique"))
    return r.reindex([s for s in ["train", "dev", "test_sk001", "test_otros", "fuera_tts", "fuera_texto"]
                      if s in r.index])


# ============================ DATOS PARA WHISPER ============================
class ClipsASR(torch.utils.data.Dataset):
    def __init__(self, df, raiz, processor):
        self.filas = list(df[["file_name", "text_norm"]].itertuples(index=False))
        self.raiz, self.p = Path(raiz), processor

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        archivo, texto = self.filas[i]
        y, sr = sf.read(self.raiz / archivo, dtype="float32")
        assert sr == SR, f"{archivo}: {sr} Hz"
        return {"input_features": self.p.feature_extractor(y, sampling_rate=SR).input_features[0],
                "labels": self.p.tokenizer(texto).input_ids}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, feats):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():   # el modelo lo agrega solo
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ============================ MEZCLA REAL + AGREGADO (Fase C, D10) ============================
def leer_agregado(carpetas, frac=1.0):
    """Une los metadata.csv de las carpetas de datos agregados (sintético filtrado o aumento), con rutas absolutas.

    frac < 1 conserva solo esa fracción de frases (frase_id en el sintético, origen_id en el aumento). La elección
    depende solo del id (crc32), así es la misma en todas las carpetas y semillas, y las fracciones quedan anidadas
    (25 % dentro de 50 % dentro de 100 %)."""
    import zlib
    partes = []
    for c in carpetas:
        c = Path(c).expanduser()
        m = pd.read_csv(c / "metadata.csv")
        m["file_name"] = [str(c / f) for f in m.file_name]
        partes.append(m)
    m = pd.concat(partes, ignore_index=True)
    if frac < 1:
        clave = "frase_id" if "frase_id" in m else "origen_id"
        u = m[clave].astype(str).map(lambda k: zlib.crc32(k.encode()) / 2**32)
        m = m[u < frac].reset_index(drop=True)
    return m


def orden_mezcla(real, agregado, n_items, micro_lote, prop_real, semilla):
    """Lista fija de n_items filas para recorrer EN ORDEN: cada micro-lote lleva round(prop_real * micro_lote)
    clips reales y el resto agregados (con lote 16 y 0.5: 8 + 8 siempre, no solo en promedio). Cada fuente se
    recorre en permutaciones sucesivas, así todos sus clips se ven un número parejo de veces."""
    rng = np.random.default_rng(semilla)
    k = round(prop_real * micro_lote)
    es_real = (np.arange(n_items) % micro_lote) < k

    def turnos(n, cuantos):
        if cuantos == 0:                                  # prop_real 0 (solo sintético) o 1
            return np.array([], dtype=int)
        return np.concatenate([rng.permutation(n) for _ in range(-(-cuantos // n))])[:cuantos]

    cols = ["file_name", "text_norm"]
    r = real[cols].iloc[turnos(len(real), int(es_real.sum()))].to_numpy()
    a = agregado[cols].iloc[turnos(len(agregado), int((~es_real).sum()))].to_numpy()
    filas = np.empty((n_items, 2), dtype=object)
    filas[es_real], filas[~es_real] = r, a
    return pd.DataFrame(filas, columns=cols).assign(real=es_real)


def en_orden(clase_trainer):
    """Trainer que recorre train_dataset en orden (el orden de orden_mezcla ya está barajado con la semilla)."""
    class TrainerEnOrden(clase_trainer):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)
    return TrainerEnOrden


# ============================ METRICAS ============================
def wer_cer(refs, hyps):
    import jiwer
    pares = [(r, h) for r, h in zip(refs, hyps) if r]            # jiwer no acepta referencias vacias
    r, h = [p[0] for p in pares], [p[1] for p in pares]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def metricas_trainer(processor):
    tok = processor.tokenizer

    def f(pred):
        ids, lab = pred.predictions, pred.label_ids
        lab = np.where(lab == -100, tok.pad_token_id, lab)
        ids = np.where(ids == -100, tok.pad_token_id, ids)
        hyps = [normalizar(t) for t in tok.batch_decode(ids, skip_special_tokens=True)]
        refs = [normalizar(t) for t in tok.batch_decode(lab, skip_special_tokens=True)]
        w, c = wer_cer(refs, hyps)
        return {"wer": round(w, 2), "cer": round(c, 2)}
    return f


def wer_con_ic(pred, seed=0, n=1000):
    """WER corpus (no promedio por clip) e IC 95% por bootstrap sobre clips."""
    import jiwer
    p = pred[pred.text_norm.str.len() > 0]
    err, pal = [], []
    for a, b in zip(p.text_norm, p.prediccion):
        o = jiwer.process_words(a, b)
        err.append(o.substitutions + o.deletions + o.insertions)
        pal.append(o.substitutions + o.deletions + o.hits)
    err, pal = np.array(err), np.array(pal)
    rng = np.random.RandomState(seed)
    k = rng.randint(0, len(p), size=(n, len(p)))
    boot = 100 * err[k].sum(1) / pal[k].sum(1)
    return dict(wer=round(float(100 * err.sum() / pal.sum()), 2), ic95=[round(float(x), 2) for x in np.percentile(boot, [2.5, 97.5])],
                cer=round(100 * jiwer.cer(list(p.text_norm), list(p.prediccion)), 2), clips=int(len(p)),
                palabras=int(pal.sum()))


@torch.no_grad()
def transcribir(model, processor, df, raiz, device, batch=16, max_tokens=128, num_beams=1):
    """Transcribe df y devuelve una copia con la prediccion y el error por clip."""
    import jiwer
    model.eval()
    dtype = next(model.parameters()).dtype
    salida = []
    for i in range(0, len(df), batch):
        trozo = df.iloc[i:i + batch]
        audios = [sf.read(Path(raiz) / f, dtype="float32")[0] for f in trozo.file_name]
        x = processor.feature_extractor(audios, sampling_rate=SR, return_tensors="pt").input_features
        ids = model.generate(x.to(device, dtype), language="spanish", task="transcribe",
                             max_new_tokens=max_tokens, num_beams=num_beams)
        salida += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    r = df[["id", "file_name", "source", "speaker_group", "duration_s", "text_norm"]].copy()
    r["prediccion"] = [normalizar(t) for t in salida]
    r["wer_clip"] = [round(100 * jiwer.wer(a, b), 1) if a else None for a, b in zip(r.text_norm, r.prediccion)]
    return r