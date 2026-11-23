# ---- filtros del audio sintetico: se prueba localmente y se copia tal cual al notebook ----
import csv, math, os, unicodedata
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR = 16000


def a16k(y, sr):
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g)
    return np.ascontiguousarray(y, dtype=np.float32)


def muestras_en_meseta(y, run=3, tol=0.999):
    """Muestras en tramos de >= run muestras seguidas pegadas al pico: la firma de una onda recortada.
    Un pico natural toca el maximo en 1-2 muestras; una saturada se queda plana ahi."""
    a = np.abs(y)
    pico = a.max() if a.size else 0.0
    if pico <= 0:
        return 0
    m = np.concatenate([[0], (a >= tol * pico).astype(np.int8), [0]])
    d = np.diff(m)
    ini, fin = np.where(d == 1)[0], np.where(d == -1)[0]
    largos = fin - ini
    return int(largos[largos >= run].sum())


def medir(ruta, vad, get_ts):
    """Medidas de audio de un clip. voz_s = segundos con voz segun Silero VAD."""
    try:
        y, sr = sf.read(ruta, dtype="float32", always_2d=True)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}"[:120])
    y = y.mean(axis=1)
    dur = len(y) / sr
    if len(y) == 0:
        return dict(dur_s=0.0, rms_db=-120.0, meseta_pct=0.0, voz_s=0.0, voz_pct=0.0)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    y16 = a16k(y, sr)
    ts = get_ts(torch.from_numpy(y16), vad, sampling_rate=SR) if len(y16) >= 512 else []
    voz = sum(t["end"] - t["start"] for t in ts) / SR
    return dict(dur_s=round(dur, 3), rms_db=round(20 * math.log10(rms + 1e-12), 2),
                meseta_pct=round(100 * muestras_en_meseta(y) / len(y), 4),
                voz_s=round(voz, 3), voz_pct=round(100 * voz / dur, 1) if dur else 0.0)


def medir_lote(filas, raiz, cache, vad, get_ts, cada=200):
    """filas: [(id, ruta_relativa)]. Reanudable: lo ya medido en `cache` no se repite."""
    hechos = set()
    if Path(cache).exists():
        with open(cache, encoding="utf-8", newline="") as f:
            hechos = {r["id"] for r in csv.DictReader(f) if r.get("id")}
    pend = [f for f in filas if f[0] not in hechos]
    campos = ["id", "dur_s", "rms_db", "meseta_pct", "voz_s", "voz_pct", "error"]
    nuevo = not Path(cache).exists()
    from tqdm.auto import tqdm
    with open(cache, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        for k, (i, rel) in enumerate(tqdm(pend, desc="midiendo")):
            w.writerow({"id": i, **medir(os.path.join(raiz, rel), vad, get_ts)})
            if k % cada == 0:
                f.flush()
    return len(pend)


# ============================ TEXTO / CER ============================
_ACENTOS = {"́", "̀"}


def sin_tildes(t):
    t = unicodedata.normalize("NFD", str(t))
    return unicodedata.normalize("NFC", "".join(c for c in t if c not in _ACENTOS))


def cer_clip(ref, hyp):
    import jiwer
    if not ref:
        return None
    return round(100 * jiwer.cer(ref, hyp if hyp else ""), 2)


def leer_16k(ruta):
    y, sr = sf.read(ruta, dtype="float32", always_2d=True)
    return a16k(y, sr)


@torch.no_grad()
def transcribir_audios(model, processor, audios, device, max_tokens=128):
    dtype = next(model.parameters()).dtype
    x = processor.feature_extractor(audios, sampling_rate=SR, return_tensors="pt").input_features
    ids = model.generate(x.to(device, dtype), language="spanish", task="transcribe",
                         max_new_tokens=max_tokens, num_beams=1)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)


def orden_de_trabajo(df, seed, bloque=2000):
    """Barajado con semilla (lo transcrito en cualquier momento es una muestra representativa) y, dentro de
    cada bloque, ordenado por largo del texto: un lote dura lo que su frase más larga, así se decodifica menos relleno."""
    d = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    d["_b"] = d.index // bloque
    d["_n"] = d.text_norm.str.len()
    return d.sort_values(["_b", "_n"]).drop(columns=["_b", "_n"])


def ida_y_vuelta(df, raiz, cache, model, processor, device, normalizar, batch=32, cada=10, seed=1234, hilos=16):
    """df con id, archivo, text_norm. Escribe id, prediccion, cer, cer_sin_tildes. Reanudable.
    Lee los wavs en paralelo y prepara el lote siguiente mientras la GPU decodifica el actual:
    leer de Drive archivo por archivo deja la GPU esperando."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    hechos = set()
    if Path(cache).exists():
        with open(cache, encoding="utf-8", newline="") as f:
            hechos = {r["id"] for r in csv.DictReader(f) if r.get("id")}
    pend = orden_de_trabajo(df[~df.id.isin(hechos)], seed)
    lotes = [pend.iloc[i:i + batch] for i in range(0, len(pend), batch)]
    campos = ["id", "prediccion", "cer", "cer_sin_tildes"]
    nuevo = not Path(cache).exists()
    from tqdm.auto import tqdm
    model.eval()
    io_pool, pre = ThreadPoolExecutor(hilos), ThreadPoolExecutor(1)
    cargar = lambda t: list(io_pool.map(leer_16k, [os.path.join(raiz, a) for a in t.archivo]))
    with open(cache, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            w.writeheader()
        barra = tqdm(lotes, desc="ida y vuelta")
        fut = pre.submit(cargar, lotes[0]) if lotes else None
        for k, t in enumerate(barra):
            t0 = time.time()
            audios = fut.result()
            espera = time.time() - t0
            if k + 1 < len(lotes):
                fut = pre.submit(cargar, lotes[k + 1])
            t1 = time.time()
            hyps = transcribir_audios(model, processor, audios, device)
            barra.set_postfix(espera_lectura=f"{espera:.1f}s", gpu=f"{time.time() - t1:.1f}s")
            for r, h in zip(t.itertuples(), hyps):
                h = normalizar(h)
                w.writerow(dict(id=r.id, prediccion=h, cer=cer_clip(r.text_norm, h),
                                cer_sin_tildes=cer_clip(sin_tildes(r.text_norm), sin_tildes(h))))
            if k % cada == 0:
                f.flush()
    io_pool.shutdown(); pre.shutdown()
    return len(pend)