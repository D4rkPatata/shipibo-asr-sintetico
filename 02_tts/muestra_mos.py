"""R4: muestra para la evaluación perceptual (MOS e IR) del audio sintético, con códigos anónimos.

Muestra (30 audios + 2 de práctica), con semilla fija:
  10 Tacotron2 aceptados, 10 MMS-TTS aceptados, 6 reales de SK001 (test_sk001, referencia oculta),
  4 descartados por el filtro (2 de cada TTS, de las etapas 2 "duración vs texto" y 3 "ida y vuelta").
  Práctica: 1 real y 1 sintético fuera de la muestra (no se analizan).
Duración entre 3 y 8 s, sin frases de deletreo y ninguna frase repetida entre audios (los dos TTS sintetizaron las mismas frases).

Todos los audios se llevan a 16 kHz (la frecuencia del audio real y de MMS-TTS) y al mismo nivel (RMS −20 dBFS,
pico ≤ −1 dBFS), con 0.5 s de silencio al inicio y al final, para que la frecuencia o el volumen no delaten el tipo.
Los códigos (A01–A30) se asignan en orden aleatorio.

Salida (<salida>/): audios/A01.wav…, audios/P1.wav, P2.wav y clave.csv (código → tipo, id, texto de referencia).
La clave no se muestra a los evaluadores.

Uso:  python muestra_mos.py --sintetico ~/tesis/datos/sintetico_v1 --tts ~/tesis/tts \
          --evaluacion ~/tesis/evaluacion_v1 --salida ~/tesis/fase_E/mos
"""
import argparse
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SEMILLA, SR, DUR = 20261109, 16000, (3.0, 8.0)


def deletreo(t):
    """Frases de cartilla con sílabas sueltas (p. ej. "yo pe si"): difíciles de transcribir, no representan el corpus."""
    return sum(len(w) <= 2 for w in t.split()) >= 3


def preparar(origen, destino):
    x, sr = sf.read(origen, dtype="float64")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    x = x * 10 ** (-20 / 20) / max(np.sqrt(np.mean(x ** 2)), 1e-9)
    x = x * min(1.0, 10 ** (-1 / 20) / max(np.abs(x).max(), 1e-9))
    pausa = np.zeros(SR // 2)
    sf.write(destino, np.concatenate([pausa, x, pausa]), SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetico", required=True)
    ap.add_argument("--tts", required=True, help="carpeta con el audio generado sin filtrar (tacotron/, mms/)")
    ap.add_argument("--evaluacion", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    S, T, E, sal = (Path(x).expanduser() for x in (a.sintetico, a.tts, a.evaluacion, a.salida))
    (sal / "audios").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    usadas, filas = set(), []

    def tomar(df, n, texto, tipo, ruta, id_col="id"):
        df = df[df[texto].map(lambda t: t not in usadas)]
        df = df.iloc[rng.permutation(len(df))].drop_duplicates(texto).head(n)
        assert len(df) == n, tipo
        usadas.update(df[texto])
        for _, r in df.iterrows():
            filas.append(dict(tipo=tipo, id=r[id_col], texto=r[texto], origen=str(ruta(r))))

    for tts in ("tacotron", "mms"):
        acept = pd.read_csv(S / tts / "metadata.csv", keep_default_na=False)
        acept = acept[acept.duration_s.between(*DUR)]
        acept = acept[~acept.text_norm.map(deletreo)]
        tomar(acept, 11 if tts == "tacotron" else 10, "text_norm", f"sintetico_{tts}", lambda r, t=tts: S / t / r.file_name)
        det = pd.read_csv(S / tts / "filtrado" / "detalle_filtrado.csv", keep_default_na=False)
        desc = det[det.descarte.str.startswith(("2 ", "3 ")) & pd.to_numeric(det.dur_s, errors="coerce").between(*DUR)]
        tomar(desc, 2, "text_norm", f"descartado_{tts}", lambda r, t=tts: T / t / r.archivo)
    ref = pd.read_csv(E / "referencias.csv", keep_default_na=False)
    real = ref[(ref.split == "test_sk001") & ref.duration_s.between(*DUR)]
    tomar(real, 7, "text_norm", "real_sk001", lambda r: E / r.file_name)

    m = pd.DataFrame(filas)
    # práctica: un real y un sintético (el último tomado de cada grupo), fuera del análisis
    practica = pd.concat([m[m.tipo == "real_sk001"].tail(1), m[m.tipo == "sintetico_tacotron"].tail(1)])
    m = m.drop(practica.index)
    m = m.iloc[rng.permutation(len(m))].reset_index(drop=True)
    m.insert(0, "codigo", [f"A{i + 1:02d}" for i in range(len(m))])
    practica.insert(0, "codigo", ["P1", "P2"])
    clave = pd.concat([practica, m], ignore_index=True)
    for _, r in clave.iterrows():
        preparar(r.origen, sal / "audios" / f"{r.codigo}.wav")
    clave["duracion_s"] = [round(sf.info(sal / "audios" / f"{c}.wav").duration - 1, 2) for c in clave.codigo]
    clave.to_csv(sal / "clave.csv", index=False)
    print(clave.drop(columns="origen").to_string(index=False))
    print(clave.tipo.value_counts().to_string())


if __name__ == "__main__":
    main()
