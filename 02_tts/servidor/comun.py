"""Piezas compartidas por los dos generadores: mismas frases, mismo orden y misma semilla por frase
que generation.ipynb, para que todo el audio sintético de la tesis sea comparable y reproducible."""
import csv
import hashlib
import os
import re
import zlib
from pathlib import Path

import numpy as np

import tts_core as C            # copia exacta del núcleo de generation.ipynb

SEED = 1234
MAX_PALABRAS, MIN_PALABRAS = 20, 2
N_COLAB = 11779                 # frases que ya generó Colab con Tacotron2 (las primeras del orden barajado)


def frases(ruta_texto):
    """Las 23,558 frases en el orden barajado de generation.ipynb (el mismo para los dos TTS)."""
    items = C.construir_items(ruta_texto, MAX_PALABRAS, MIN_PALABRAS)
    orden = list(range(len(items)))
    np.random.RandomState(SEED).shuffle(orden)
    return [items[k] for k in orden]


def semilla_de(id_):
    return (SEED + zlib.crc32(id_.encode())) % (2 ** 31)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Manifiesto:
    """manifest.csv reanudable: lo ya escrito no se repite; filas cortadas o con error se reintentan."""

    def __init__(self, carpeta, campos):
        self.carpeta, self.campos = Path(carpeta), campos
        self.ruta = self.carpeta / "manifest.csv"
        self.carpeta.mkdir(parents=True, exist_ok=True)
        for basura in self.carpeta.rglob("*.part"):
            basura.unlink()
        previas = {}
        if self.ruta.exists():
            with open(self.ruta, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    est = r.get("estado") or ""
                    if re.fullmatch(r"L[0-9]{5}_[0-9]{2}", r.get("id") or "") and est and not est.startswith("error"):
                        previas[r["id"]] = {c: r.get(c, "") for c in campos}
        tmp = self.ruta.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(previas.values())
        os.replace(tmp, self.ruta)
        self.hechos = set(previas)
        self._f = open(self.ruta, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=campos)

    def escribir(self, fila):
        self._w.writerow(fila)
        self._f.flush()

    def cerrar(self):
        self._f.close()


def guardar_wav(carpeta, id_, y, sr):
    """Escritura atómica en subcarpetas de ~1000 archivos: solo existe el wav si se escribió completo."""
    import soundfile as sf
    rel = f"{id_[:3]}/{id_}.wav"
    (Path(carpeta) / id_[:3]).mkdir(exist_ok=True)
    tmp = Path(carpeta) / f"{rel}.part"
    sf.write(tmp, y, sr, subtype="PCM_16", format="WAV")
    os.replace(tmp, Path(carpeta) / rel)
    return rel


def pico_menos_1db(y):
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    return (y * (10 ** (-1 / 20) / pico)).astype(np.float32) if pico > 0 else y.astype(np.float32)
