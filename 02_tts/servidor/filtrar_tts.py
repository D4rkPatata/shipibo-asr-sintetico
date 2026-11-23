"""Filtra el audio sintético de un TTS y lo exporta como dataset a 16 kHz (Fase B). Mismo criterio para los dos TTS.

Filtros en orden (un clip cuenta solo en el primero que lo descarta). 0–2 son los de filtrar_sintetico.ipynb, con la
misma lógica (tts_filtro.py es copia exacta de su núcleo); 3 es la ida y vuelta pospuesta en D06:

  0  no generado / fuga        sin wav, error, o el texto contiene una frase de dev/test (incluye test_control)
  1a vacío                     < 0.5 s o RMS < −50 dBFS
  1b saturado                  > 0.1 % de muestras en mesetas pegadas al pico
  1c silencio                  voz (Silero VAD) < 50 % del clip
  1d QC de atención            estado != ok (solo Tacotron2: MMS-TTS no tiene atención)
  2  duración vs texto         segundos de voz por carácter fuera de p1–p99 de SK001
  3  ida y vuelta              dos jueces transcriben el clip: A3 (MMS-1b-all afinado) y A2 (Whisper medium afinado);
                               se descarta si CUALQUIERA da CER sin tildes > 40 %

El umbral de 40 % y la regla "cualquiera" se fijan antes de mirar los resultados. Los jueces tienen CER ~10–12 % en
voces reales no vistas, y en ese habla real solo el 1.4 % de los clips supera 40 % con cada juez: el filtro solo saca
clips que un juez casi no reconoce (TTS que balbucea, se salta o repite palabras). Dos arquitecturas distintas evitan
que pase un fallo al que el juez MMS sea indulgente por venir de la misma familia que MMS-TTS. Se guarda el CER de
cada clip con cada juez para el análisis de sensibilidad (20/30/40/50 %) sin volver a transcribir.

Reanudable: medidas e ida y vuelta se guardan en <salida>/filtrado/ a medida que se calculan.

Uso:  python filtrar_tts.py --tts tacotron --gen ~/tesis/tts/tacotron --salida ~/tesis/datos/sintetico_v1/tacotron \
          --datos ~/tesis/datos/dataset_v1.1 --split ~/tesis/datos/split_v1.1.csv \
          --referencias ~/tesis/evaluacion_v1/referencias.csv --juez ~/tesis/fase_A/A3/mms-1b-all_real_s1234/modelo_final
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_filtro as T                                          # noqa: E402

sys.path.insert(0, str(Path.home() / "tesis/fase_A/codigo"))
import asr_core as C                                            # noqa: E402  (normalizar, igual que en el corpus)

DUR_MIN_S, RMS_MIN_DB, MESETA_MAX_PCT, VOZ_MIN_PCT = 0.5, -50.0, 0.1, 50.0
P_BAJO, P_ALTO = 1, 99
CER_MAX = 40.0
SENSIBILIDAD = [20, 30, 40, 50]
FUENTE_SK = "menendez_gomez_2025"
REPO_JUEZ, REV_JUEZ = "facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
SUBFUENTE = {"tacotron": "tacotron2_hifigan_SKF2", "mms": "mms_tts_shp"}

# ---------------------------------------------------------------- medidas en paralelo (CPU)
_vad = None


def _iniciar():
    global _vad
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _vad = load_silero_vad()


def _medir(args):
    from silero_vad import get_speech_timestamps
    id_, ruta = args
    return {"id": id_, **T.medir(ruta, _vad, get_speech_timestamps)}


def medir_paralelo(filas, cache, procesos):
    """filas: [(id, ruta absoluta)]. Mismas medidas que T.medir_lote, repartidas en varios procesos."""
    hechos = set()
    if cache.exists():
        hechos = set(pd.read_csv(cache, usecols=["id"]).id)
    pend = [f for f in filas if f[0] not in hechos]
    campos = ["id", "dur_s", "rms_db", "meseta_pct", "voz_s", "voz_pct", "error"]
    nuevo = not cache.exists()
    t0 = time.time()
    with open(cache, "a", encoding="utf-8", newline="") as f, Pool(procesos, initializer=_iniciar) as pool:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        for k, r in enumerate(pool.imap_unordered(_medir, pend, chunksize=16)):
            w.writerow(r)
            if k % 1000 == 0:
                f.flush()
                print(f"  medidos {k}/{len(pend)} ({time.time() - t0:.0f} s)", flush=True)
    m = pd.read_csv(cache, keep_default_na=False).drop_duplicates("id", keep="last")
    for c in ("dur_s", "rms_db", "meseta_pct", "voz_s", "voz_pct"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m


# ---------------------------------------------------------------- juez (GPU)
def cargar_juez(carpeta):
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(REPO_JUEZ, revision=REV_JUEZ, target_lang="shp")
    model = Wav2Vec2ForCTC.from_pretrained(REPO_JUEZ, revision=REV_JUEZ, target_lang="shp", ignore_mismatched_sizes=True)
    model.load_adapter("shp")
    pesos = load_file(str(Path(carpeta) / "adapter.shp.safetensors"))
    faltan = set(pesos) - set(model.state_dict())
    assert not faltan, f"pesos del adaptador que el modelo no tiene: {list(faltan)[:3]}"
    model.load_state_dict(pesos, strict=False)
    return proc, model.to("cuda").eval()


def transcriptor_mms(carpeta):
    proc, model = cargar_juez(carpeta)

    def f(audios):
        x = proc(audios, sampling_rate=T.SR, return_tensors="pt", padding=True)
        logits = model(x.input_values.to("cuda"), attention_mask=x.attention_mask.to("cuda")).logits
        return proc.batch_decode(logits.argmax(-1))
    return f


def transcriptor_whisper(carpeta):
    """Mismo decodificado que A0/A2: greedy, token de español, fp16."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(carpeta)
    model = WhisperForConditionalGeneration.from_pretrained(carpeta, torch_dtype=torch.float16).to("cuda").eval()

    def f(audios):
        x = proc.feature_extractor(audios, sampling_rate=T.SR, return_tensors="pt").input_features
        ids = model.generate(x.to("cuda", torch.float16), language="spanish", task="transcribe", max_new_tokens=128,
                             num_beams=1)
        return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
    return f


@torch.no_grad()
def ida_y_vuelta(df, raiz, cache, transcribir, batch=32, hilos=16):
    """df: id, archivo, text_norm. Escribe id, prediccion, cer, cer_sin_tildes. Reanudable.
    Lotes ordenados por duración (menos relleno) y lectura en paralelo con la GPU."""
    hechos = set(pd.read_csv(cache, usecols=["id"]).id) if cache.exists() else set()
    pend = df[~df.id.isin(hechos)].sort_values("dur_s")
    lotes = [pend.iloc[i:i + batch] for i in range(0, len(pend), batch)]
    io, pre = ThreadPoolExecutor(hilos), ThreadPoolExecutor(1)
    cargar = lambda t: list(io.map(T.leer_16k, [str(raiz / a) for a in t.archivo]))
    nuevo = not cache.exists()
    t0 = time.time()
    with open(cache, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prediccion", "cer", "cer_sin_tildes"])
        if nuevo:
            w.writeheader()
        fut = pre.submit(cargar, lotes[0]) if lotes else None
        for k, t in enumerate(lotes):
            audios = fut.result()
            if k + 1 < len(lotes):
                fut = pre.submit(cargar, lotes[k + 1])
            for r, h in zip(t.itertuples(), transcribir(audios)):
                h = C.normalizar(h)
                w.writerow(dict(id=r.id, prediccion=h, cer=T.cer_clip(r.text_norm, h),
                                cer_sin_tildes=T.cer_clip(T.sin_tildes(r.text_norm), T.sin_tildes(h))))
            if k % 50 == 0:
                f.flush()
                print(f"  ida y vuelta {k * batch}/{len(pend)} ({time.time() - t0:.0f} s)", flush=True)
    io.shutdown(); pre.shutdown()
    return pd.read_csv(cache, keep_default_na=False).drop_duplicates("id", keep="last")


# ---------------------------------------------------------------- exportar a 16 kHz
def exportar(fila, raiz, wavs):
    origen, destino = raiz / fila.archivo, wavs / fila.archivo
    if destino.exists():
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    if sf.info(str(origen)).samplerate == T.SR:
        os.link(origen, destino)                                # MMS-TTS ya está a 16 kHz
    else:
        tmp = destino.with_suffix(".part.wav")
        sf.write(str(tmp), T.leer_16k(str(origen)), T.SR, subtype="PCM_16")
        tmp.rename(destino)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", choices=SUBFUENTE, required=True)
    ap.add_argument("--gen", required=True, help="carpeta con manifest.csv y los wav generados")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--referencias", required=True, help="referencias.csv de evaluacion_v1 (para test_control)")
    ap.add_argument("--juez", required=True, help="modelo_final de A3 (adaptador MMS)")
    ap.add_argument("--juez2", required=True, help="modelo_final de A2 (Whisper medium)")
    ap.add_argument("--procesos", type=int, default=24)
    ap.add_argument("--limite", type=int, default=0, help="procesar solo las primeras N frases (prueba)")
    a = ap.parse_args()
    gen, sal = Path(a.gen).expanduser(), Path(a.salida).expanduser()
    fil = sal / "filtrado"
    fil.mkdir(parents=True, exist_ok=True)

    # --- manifest y fuga
    man = pd.read_csv(gen / "manifest.csv", dtype=str, keep_default_na=False)
    if a.limite:
        man = man.head(a.limite)                                # solo para la prueba corta
    man["text_norm"] = man.texto.map(C.normalizar)
    man["n_chars"] = man.text_norm.str.replace(" ", "").str.len()
    datos = Path(a.datos).expanduser()
    meta = pd.read_csv(datos / "metadata.csv")
    split = meta.merge(pd.read_csv(Path(a.split).expanduser())[["id", "split"]], on="id")
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    textos_eval = set(split[split.split.isin(["dev", "test_sk001", "test_otros"])].text_norm) \
        | set(ref[ref.split == "test_control"].text_norm.map(C.normalizar))
    man["fuga_eval"] = man.text_norm.map(lambda t: any(f" {e} " in f" {t} " for e in textos_eval))
    man["texto_en_train_real"] = man.text_norm.isin(set(split[split.split == "train"].text_norm))
    print(f"{a.tts}: {len(man)} frases | fuga: {int(man.fuga_eval.sum())}", flush=True)

    # --- medidas (filtros 1–2)
    con_wav = man[(man.archivo != "") & ~man.estado.str.startswith("error")]
    med = medir_paralelo([(r.id, str(gen / r.archivo)) for r in con_wav.itertuples()], fil / "medidas_tts.csv", a.procesos)
    sk = meta[meta.source == FUENTE_SK]
    msk = medir_paralelo([(r.id, str(datos / r.file_name)) for r in sk.itertuples()], sal.parent / "medidas_sk001.csv",
                         a.procesos)
    sk = sk.merge(msk, on="id")
    sk["seg_por_char"] = sk.voz_s / sk.text_norm.str.replace(" ", "").str.len()
    lim_bajo, lim_alto = np.percentile(sk.seg_por_char.dropna(), [P_BAJO, P_ALTO])
    d = man.drop(columns=[c for c in med.columns if c != "id" and c in man]).merge(med, on="id", how="left")
    d["seg_por_char"] = d.voz_s / d.n_chars.replace(0, np.nan)
    print(f"SK001 seg/carácter p{P_BAJO}={lim_bajo:.4f} p{P_ALTO}={lim_alto:.4f} | TTS mediana {d.seg_por_char.median():.4f}")

    d["descarte"] = ""

    def aplicar(nombre, mascara):
        m = (d.descarte == "") & mascara.fillna(False).astype(bool)
        d.loc[m, "descarte"] = nombre

    etapas = [
        ("0 no generado", (d.archivo == "") | d.estado.str.startswith("error") | d.dur_s.isna()
                          | (d["error"].fillna("").astype(str) != "")),
        ("0 fuga de evaluación", d.fuga_eval),
        ("1a vacío", (d.dur_s < DUR_MIN_S) | (d.rms_db < RMS_MIN_DB)),
        ("1b saturado", d.meseta_pct > MESETA_MAX_PCT),
        ("1c silencio > 50%", d.voz_pct < VOZ_MIN_PCT),
        ("1d QC atención Tacotron2", (d.estado != "ok") if a.tts == "tacotron" else pd.Series(False, index=d.index)),
        ("2 duración vs texto", (d.seg_por_char < lim_bajo) | (d.seg_por_char > lim_alto)),
    ]
    for nombre, m in etapas:
        aplicar(nombre, m)

    # --- ida y vuelta (filtro 3) sobre lo que pasó 0–2, con los dos jueces
    vivos = d[d.descarte == ""][["id", "archivo", "text_norm", "dur_s"]]
    for sufijo, crear, carpeta, cache in [("", transcriptor_mms, a.juez, "ida_y_vuelta.csv"),
                                          ("_whisper", transcriptor_whisper, a.juez2, "ida_y_vuelta_whisper.csv")]:
        f = crear(str(Path(carpeta).expanduser()))
        iv = ida_y_vuelta(vivos, gen, fil / cache, f).rename(columns=lambda c: c if c == "id" else c + sufijo)
        del f
        torch.cuda.empty_cache()
        d = d.merge(iv, on="id", how="left")
        d["cer_sin_tildes" + sufijo] = pd.to_numeric(d["cer_sin_tildes" + sufijo], errors="coerce")
    d["cer_max_jueces"] = d[["cer_sin_tildes", "cer_sin_tildes_whisper"]].max(axis=1)
    nombre3 = f"3 ida y vuelta CER > {CER_MAX:g}% (algún juez)"
    aplicar(nombre3, d.cer_max_jueces > CER_MAX)
    etapas.append((nombre3, None))

    # --- resumen
    h = lambda m: round(float(d.loc[m, "dur_s"].sum()) / 3600, 2)
    filas, fuera = [dict(etapa="generados", descartados="", quedan=len(d), horas=h(d.id.notna()))], set()
    for nombre, _ in etapas:
        fuera.add(nombre)
        filas.append(dict(etapa=nombre, descartados=int((d.descarte == nombre).sum()),
                          quedan=int((~d.descarte.isin(fuera)).sum()), horas=h(~d.descarte.isin(fuera))))
    acept = d[d.descarte == ""].copy()
    filas.append(dict(etapa="ACEPTADOS", descartados=len(d) - len(acept), quedan=len(acept), horas=h(d.descarte == "")))
    resumen = pd.DataFrame(filas)
    resumen.to_csv(fil / "resumen_filtrado.csv", index=False)
    print(resumen.to_string(index=False))
    pasa_2 = d[d.cer_sin_tildes.notna()]
    jueces = {"mms": pasa_2.cer_sin_tildes, "whisper": pasa_2.cer_sin_tildes_whisper, "algun_juez": pasa_2.cer_max_jueces}
    sens = pd.DataFrame([dict(umbral_cer=u, **{f"pct_descarta_{k}": round(100 * float((v > u).mean()), 2)
                                               for k, v in jueces.items()}) for u in SENSIBILIDAD])
    sens.to_csv(fil / "sensibilidad_ida_y_vuelta.csv", index=False)
    for k, v in jueces.items():
        print(f"CER {k} sobre lo que pasó 0–2: mediana {v.median():.2f} | p90 {v.quantile(.9):.2f}")
    print(sens.to_string(index=False))
    d.to_csv(fil / "detalle_filtrado.csv", index=False)

    # --- exportar dataset a 16 kHz
    wavs = sal / "wavs"
    if wavs.exists():                                           # de una corrida anterior con otro criterio
        quedan = {str(wavs / x) for x in acept.archivo}
        for w in wavs.rglob("*.wav"):
            if str(w) not in quedan:
                w.unlink()
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda r: exportar(r, gen, wavs), acept.itertuples()))
    out = pd.DataFrame({
        "id": f"TTS{a.tts.upper()}-" + acept.id, "file_name": "wavs/" + acept.archivo, "transcription": acept.texto,
        "text_norm": acept.text_norm, "source": f"tts_{a.tts}", "subsource": SUBFUENTE[a.tts],
        "speaker_group": f"TTS-{a.tts}", "duration_s": acept.dur_s, "voz_pct": acept.voz_pct,
        "seg_por_char": acept.seg_por_char.round(4), "cer_juez": acept.cer_sin_tildes,
        "cer_juez_whisper": acept.cer_sin_tildes_whisper, "seed_tts": acept.seed,
        "texto_en_train_real": acept.texto_en_train_real, "frase_id": acept.id,
        "corrida": acept.get("corrida", pd.Series("", index=acept.index))})
    out.to_csv(sal / "metadata.csv", index=False)
    import silero_vad
    import transformers
    (fil / "config_filtrado.json").write_text(json.dumps(dict(
        tts=a.tts, gen=str(gen), dur_min_s=DUR_MIN_S, rms_min_db=RMS_MIN_DB, meseta_max_pct=MESETA_MAX_PCT,
        voz_min_pct=VOZ_MIN_PCT, percentiles_sk001=[P_BAJO, P_ALTO], lim_seg_por_char=[float(lim_bajo), float(lim_alto)],
        cer_max=CER_MAX, regla="descarta si algún juez supera cer_max", juez=str(a.juez), juez2=str(a.juez2), juez_base=REPO_JUEZ, juez_revision=REV_JUEZ, generados=len(d),
        aceptados=len(acept), horas_aceptadas=h(d.descarte == ""), silero_vad=getattr(silero_vad, "__version__", "?"),
        transformers=transformers.__version__, torch=torch.__version__), indent=2, ensure_ascii=False))
    print(f"{sal / 'metadata.csv'}: {len(out)} clips, {out.duration_s.sum() / 3600:.2f} h a 16 kHz")


if __name__ == "__main__":
    main()
