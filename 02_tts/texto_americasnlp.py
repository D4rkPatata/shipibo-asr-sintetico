"""Texto shipibo de AmericasNLP 2021 (train + dev + test de shipibo_konibo-spanish) limpio para la Fase C6 y el
modelo de lenguaje.

Fuentes: Galarreta et al. (2017) y Montoya et al. (2019): flashcards (Tatoeba traducido), libros de educación
bilingüe y ejemplos del diccionario SIL. Solo se usa el lado shipibo.

Se descarta una línea si (en este orden):
  vacia        no queda texto al normalizar
  simbolos     tiene dígitos o caracteres que no aparecen ni en el texto del TTS ni en las referencias
  fuga         contiene una frase de evaluación (dev, test_sk001, test_otros, test_control): misma regla que D02
  repetida     otra línea ya dio el mismo texto normalizado
  en_tts       el texto ya está en shp_apto_tts.txt (no aporta nada nuevo)

Uso:  python texto_americasnlp.py --anlp D:/tesis/Code/americasnlp2021_shp --tts_txt shp_apto_tts.txt \
          --referencias evaluacion_v1/referencias.csv --salida D:/tesis/Code/americasnlp2021_shp
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_asr" / "fase_A"))
from asr_core import normalizar      # la misma normalización que text_norm y evaluar.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anlp", required=True, help="carpeta con train.shp, dev.shp, test.shp")
    ap.add_argument("--tts_txt", required=True)
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    tts = {normalizar(l) for l in open(a.tts_txt, encoding="utf-8-sig") if l.strip()}
    ref = pd.read_csv(a.referencias, keep_default_na=False)
    evaluacion = set(ref.text_norm.map(normalizar))
    permitidos = set("".join(tts)) | set("".join(evaluacion))
    permitidos -= set("0123456789")

    filas, vistos = [], set()
    for parte in ["train", "dev", "test"]:
        for i, linea in enumerate(open(Path(a.anlp) / f"{parte}.shp", encoding="utf-8-sig")):
            original, t = linea.strip(), normalizar(linea)
            if not t:
                motivo = "vacia"
            elif any(c not in permitidos for c in t):
                motivo = "simbolos"
            elif any(f" {e} " in f" {t} " for e in evaluacion):
                motivo = "fuga"
            elif t in vistos:
                motivo = "repetida"
            elif t in tts:
                motivo = "en_tts"
            else:
                motivo = "ok"
                vistos.add(t)
            filas.append(dict(parte=parte, linea=i, motivo=motivo, original=original, text_norm=t))

    d = pd.DataFrame(filas)
    sal = Path(a.salida)
    d.to_csv(sal / "americasnlp_detalle.csv", index=False)
    ok = d[d.motivo == "ok"]
    (sal / "shp_americasnlp_tts.txt").write_text("\n".join(ok.original) + "\n", encoding="utf-8")
    print(d.groupby(["parte", "motivo"]).size().unstack(fill_value=0).to_string())
    print(f"\nquedan {len(ok)} oraciones, {ok.text_norm.str.split().str.len().sum()} palabras")
    print("fuga:", d[d.motivo == "fuga"].text_norm.tolist())


if __name__ == "__main__":
    main()
