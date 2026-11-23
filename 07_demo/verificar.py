"""Evidencia de funcionamiento de la demo: transcribe los audios de prueba con el mismo código que usa la página y
compara con la evaluación de la tesis.

  1. CER y WER de la demo (con y sin LM) frente a las referencias, en los audios de prueba.
  2. Si hay predicciones de referencia (las de la evaluación en el servidor), % de transcripciones idénticas.

Uso (desde 05_demo/):
  .venv\\Scripts\\python verificar.py --audios prueba --esperado_lm prueba/tts/mms-1b-all_real_s2024.csv \
      --esperado_greedy prueba/greedy/mms-1b-all_real_s2024.csv
  (prueba/ = referencias.csv + wavs/ de test_otros y test_control, copiados del servidor; no se publican)
Escribe prueba/verificacion.csv (una fila por audio) y prueba/verificacion.txt (resumen).
"""
import argparse
import re
import sys
import time
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "demo"))
from transcriptor import Transcriptor, cargar_audio  # noqa: E402


def normalizar(t):          # la misma de 04_asr/fase_A/evaluar.py
    t = str(t).lower()
    t = re.sub(r"[¡!¿?.,;:…\"“”«»()\[\]{}—–\-_/\\]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audios", required=True, help="carpeta con referencias.csv y wavs/")
    ap.add_argument("--esperado_lm")
    ap.add_argument("--esperado_greedy")
    ap.add_argument("--limite", type=int, default=0, help="solo los primeros N audios (prueba rápida)")
    a = ap.parse_args()
    carpeta = Path(a.audios)
    ref = pd.read_csv(carpeta / "referencias.csv", keep_default_na=False)
    ref = ref[[(carpeta / f).exists() for f in ref.file_name]].reset_index(drop=True)
    if a.limite:
        ref = ref.head(a.limite)

    t0 = time.time()
    T = Transcriptor()
    carga = time.time() - t0
    filas = []
    for k, r in ref.iterrows():
        out = T.transcribir(cargar_audio(str(carpeta / r.file_name)))
        filas.append(dict(id=r.id, split=r.split, referencia=r.text_norm, **out))
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(ref)}", flush=True)
    d = pd.DataFrame(filas)
    d.to_csv(carpeta / "verificacion.csv", index=False)

    R = d.referencia.map(normalizar).tolist()
    lineas = [f"Modelo: {T.c['nombre']}", f"Audios: {len(d)} ({', '.join(sorted(d.split.unique()))})",
              f"Carga del modelo: {carga:.0f} s; por audio: {d.duracion_s.mean():.1f} s de audio, "
              f"{(d.segundos_modelo + d.segundos_lm).mean():.2f} s de proceso (CPU)", ""]
    for col, esperado in (("sin_lm", a.esperado_greedy), ("con_lm", a.esperado_lm)):
        if d[col].isna().all():
            continue
        H = d[col].fillna("").map(normalizar).tolist()
        linea = f"{col}: CER {100 * jiwer.cer(R, H):.2f}  WER {100 * jiwer.wer(R, H):.2f}"
        if esperado:
            e = pd.read_csv(esperado, keep_default_na=False).set_index("id").prediccion_cruda
            E = [normalizar(e[i]) for i in d.id]
            igual = sum(h == x for h, x in zip(H, E)) / len(H)
            linea += (f" | evaluación de la tesis: CER {100 * jiwer.cer(R, E):.2f}  WER {100 * jiwer.wer(R, E):.2f}"
                      f" | transcripciones idénticas: {100 * igual:.1f} %")
        lineas.append(linea)
    texto = "\n".join(lineas)
    (carpeta / "verificacion.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
