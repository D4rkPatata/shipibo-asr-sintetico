"""
build_aug_v1.py
Aumento de audio OFFLINE sobre el split train del corpus pareado (dataset_v1.0 + split_v1.csv).
Nunca toca dev ni test.

Por cada clip de train genera 3 versiones:
  sp0.9   speed perturbation x0.9 (mas lento y grave)  - Ko et al. (2015), receta estandar de Kaldi
  sp1.1   speed perturbation x1.1 (mas rapido y agudo)
  ruido   ruido de fondo real a SNR uniforme en [10, 20] dB

Ruido: point-source noises de OpenSLR 28 (RIRS_NOISES, extraidos de MUSAN), licencia Apache 2.0, 16 kHz.
Se descarga SOLO esa carpeta del zip de 1.3 GB (lectura por rangos HTTP) y queda en cache.

Reproducible: la semilla de cada clip depende de SEED + id + tipo (no del orden), asi que regenerar
da exactamente el mismo audio aunque cambie el numero de clips.

Salida:
  aug_v1/
    wavs/<id>__<tipo>.wav   16 kHz, mono, PCM 16-bit
    metadata_aug.csv        una fila por clip aumentado; origen_id = id del clip real en dataset_v1.0
    stats.json / stats.md
  aug_v1.zip

Uso:  python build_aug_v1.py                          (datos en la carpeta SHIPIBO_DATOS; por defecto D:\\tesis\\Code)
      python build_aug_v1.py --split ruta/split_v1.csv  (el de Drive, TESIS/asr/split_v1.csv)
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
import zlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(os.environ.get("SHIPIBO_DATOS", r"D:\tesis\Code"))   # datos crudos y salidas
DATASET = ROOT / "dataset_v1.0"
OUT = ROOT / "aug_v1"
CACHE_RUIDO = ROOT / "_cache" / "pointsource_noises"
URL_SLR28 = "https://www.openslr.org/resources/28/rirs_noises.zip"
SR = 16000
SEED = 1234
VELOCIDADES = {"sp0.9": (10, 9), "sp1.1": (10, 11)}   # resample_poly(up, down): x0.9 alarga 10/9, x1.1 acorta 10/11
SNR_DB = (10.0, 20.0)
PICO_MAX = 0.99


# ----------------------------------------------------------------------------- ruido
class _HTTPRango(io.RawIOBase):
    """Archivo remoto de solo lectura con seek: zipfile lee el indice al final y solo los miembros pedidos."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        fin = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{fin}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def ruidos() -> list[Path]:
    wavs = sorted(CACHE_RUIDO.glob("*.wav"))
    if wavs:
        return wavs
    print(f"descargando los point-source noises de OpenSLR 28 (solo esa carpeta) -> {CACHE_RUIDO}")
    tmp = CACHE_RUIDO.with_name(CACHE_RUIDO.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BufferedReader(_HTTPRango(URL_SLR28), buffer_size=1 << 20)) as z:
        miembros = [m for m in z.namelist() if "/pointsource_noises/" in m and m.endswith(".wav")]
        for i, m in enumerate(miembros, 1):
            (tmp / Path(m).name).write_bytes(z.read(m))
            if i % 100 == 0:
                print(f"   {i}/{len(miembros)}")
    tmp.rename(CACHE_RUIDO)
    return sorted(CACHE_RUIDO.glob("*.wav"))


# ----------------------------------------------------------------------------- aumento
def rng_de(id_: str, tipo: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{SEED}|{id_}|{tipo}".encode()))


def velocidad(y: np.ndarray, tipo: str) -> np.ndarray:
    up, down = VELOCIDADES[tipo]
    return resample_poly(y, up, down).astype(np.float32)


def con_ruido(y: np.ndarray, rng: np.random.Generator, lista: list[Path]) -> tuple[np.ndarray, str, float]:
    f = lista[rng.integers(len(lista))]
    n, sr = sf.read(str(f), dtype="float32", always_2d=True)
    n = n.mean(axis=1)
    assert sr == SR, f"{f.name}: {sr} Hz"
    if len(n) < len(y):                                   # ruido corto: se repite
        n = np.tile(n, int(np.ceil(len(y) / len(n))))
    ini = rng.integers(0, len(n) - len(y) + 1)
    n = n[ini:ini + len(y)]
    snr = float(rng.uniform(*SNR_DB))
    p_voz, p_ruido = np.mean(y.astype(np.float64) ** 2), np.mean(n.astype(np.float64) ** 2)
    if p_ruido < 1e-12:                                   # tramo de ruido en silencio: sin ruido util
        return y, f.name, float("nan")
    mezcla = y + n * np.sqrt(p_voz / (p_ruido * 10 ** (snr / 10)))
    pico = np.abs(mezcla).max()
    if pico > PICO_MAX:                                   # se baja todo junto: no cambia la SNR
        mezcla *= PICO_MAX / pico
    return mezcla.astype(np.float32), f.name, round(snr, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "split_v1.csv"))
    ap.add_argument("--limite", type=int, default=0, help="solo para pruebas: primeros N clips")
    a = ap.parse_args()
    split_path = Path(a.split)
    if not split_path.exists():
        raise SystemExit(f"No encuentro {split_path}. Baja TESIS/asr/split_v1.csv de Drive (lo creó train_asr) "
                         f"y ponlo en {ROOT}, o pasa --split.")
    meta = pd.read_csv(DATASET / "metadata.csv")
    split = pd.read_csv(split_path)[["id", "split"]]
    train = meta.merge(split, on="id")
    train = train[train.split == "train"].sort_values("id")
    if a.limite:
        train = train.head(a.limite)
    print(f"train: {len(train)} clips, {train.duration_s.sum() / 60:.1f} min (dev y test no se tocan)")

    lista = ruidos()
    print(f"ruidos: {len(lista)} archivos")
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    filas = []
    for k, r in enumerate(train.itertuples(), 1):
        y, sr = sf.read(str(DATASET / r.file_name), dtype="float32")
        assert sr == SR
        for tipo in ("sp0.9", "sp1.1", "ruido"):
            rng = rng_de(r.id, tipo)
            fac, snr, rf = None, None, ""
            if tipo == "ruido":
                z, rf, snr = con_ruido(y, rng, lista)
            else:
                z, fac = velocidad(y, tipo), float(tipo[2:])
            nombre = f"wavs/{r.id}__{tipo}.wav"
            sf.write(str(OUT / nombre), z, SR, subtype="PCM_16")
            filas.append(dict(id=f"{r.id}__{tipo}", file_name=nombre, transcription=r.transcription,
                              text_norm=r.text_norm, source=r.source, subsource=r.subsource,
                              speaker_group=r.speaker_group, duration_s=round(len(z) / SR, 3),
                              origen_id=r.id, aumento="velocidad" if fac else "ruido", factor_velocidad=fac,
                              snr_db=snr, ruido_archivo=rf, semilla=SEED))
        if k % 250 == 0:
            print(f"   {k}/{len(train)}")
    md = pd.DataFrame(filas)
    md.to_csv(OUT / "metadata_aug.csv", index=False, encoding="utf-8")

    b = lambda d: dict(clips=int(len(d)), minutos=round(d.duration_s.sum() / 60, 1))
    stats = dict(semilla=SEED, ruido="OpenSLR 28 pointsource_noises (MUSAN), Apache 2.0", snr_db=list(SNR_DB),
                 train_original=b(train), aumentado=b(md), por_tipo={t: b(g) for t, g in md.groupby(md.id.str.split("__").str[1])},
                 snr_real=dict(media=round(md.snr_db.mean(), 2), min=md.snr_db.min(), max=md.snr_db.max()))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = ["# Aumento de audio v1 (solo train)", "", "| versión | clips | minutos |", "|---|---|---|",
              f"| train real (origen) | {stats['train_original']['clips']} | {stats['train_original']['minutos']} |"]
    lineas += [f"| {t} | {v['clips']} | {v['minutos']} |" for t, v in stats["por_tipo"].items()]
    lineas += [f"| **total aumentado** | {stats['aumentado']['clips']} | {stats['aumentado']['minutos']} |", "",
               f"Semilla {SEED}. SNR uniforme en {SNR_DB} dB (media real {stats['snr_real']['media']}). "
               f"Ruido: {stats['ruido']}."]
    (OUT / "stats.md").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    with zipfile.ZipFile(ROOT / "aug_v1.zip", "w", zipfile.ZIP_STORED) as z:   # wav no se comprime: STORED es igual de chico y mas rapido
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    print(f"\n{ROOT / 'aug_v1.zip'} ({(ROOT / 'aug_v1.zip').stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
