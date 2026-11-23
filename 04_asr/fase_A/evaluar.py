"""Puntúa las predicciones de cualquier modelo contra las referencias del conjunto de evaluación.

Normaliza referencia y predicción igual (text_norm de build_dataset_v1: NFC, minúsculas, sin puntuación) y reporta
WER y CER de corpus, con y sin tildes, con IC 95 % por bootstrap sobre clips, en:
  dev · test_sk001 · test_otros · test_control · no_vistos (= test_otros + test_control, criterio de selección)
y test_control por fuente.

Uso:  python evaluar.py --referencias evaluacion_v1/referencias.csv --preds fase_A/A0 --salida fase_A/A0/tabla_A0.csv
"""
import argparse
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd


def normalizar(t):
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = re.sub(r"[¡!¿?.,;:…\"“”«»()\[\]{}—–\-_/\\]", " ", t)
    t = t.replace("'", " ").replace("’", " ")
    return re.sub(r"\s+", " ", t).strip()


def sin_tildes(t):
    t = unicodedata.normalize("NFD", t)
    return unicodedata.normalize("NFC", "".join(c for c in t if c not in "́̀"))


def wer_cer(refs, hyps, n_boot=1000, seed=0):
    err, pal, cerr, car = [], [], [], []
    for r, h in zip(refs, hyps):
        o = jiwer.process_words(r, h)
        err.append(o.substitutions + o.deletions + o.insertions)
        pal.append(o.substitutions + o.deletions + o.hits)
        c = jiwer.process_characters(r, h)
        cerr.append(c.substitutions + c.deletions + c.insertions)
        car.append(c.substitutions + c.deletions + c.hits)
    err, pal, cerr, car = map(np.array, (err, pal, cerr, car))
    k = np.random.RandomState(seed).randint(0, len(err), size=(n_boot, len(err)))
    bw, bc = 100 * err[k].sum(1) / pal[k].sum(1), 100 * cerr[k].sum(1) / car[k].sum(1)
    return dict(clips=len(err), palabras=int(pal.sum()),
                wer=round(100 * err.sum() / pal.sum(), 2), wer_ic=[round(float(x), 2) for x in np.percentile(bw, [2.5, 97.5])],
                cer=round(100 * cerr.sum() / car.sum(), 2), cer_ic=[round(float(x), 2) for x in np.percentile(bc, [2.5, 97.5])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--preds", required=True, help="carpeta con <modelo>.csv")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    ref = pd.read_csv(a.referencias, keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    filas = []
    for f in sorted(Path(a.preds).glob("*.csv")):
        if f.name.startswith("tabla"):
            continue
        p = pd.read_csv(f, keep_default_na=False)
        d = ref.merge(p, on="id", how="left")
        if d.prediccion_cruda.isna().any() or len(p) != len(ref):
            print(f"AVISO {f.stem}: {d.prediccion_cruda.isna().sum()} clips sin predicción; se omite")
            continue
        d["hyp"] = d.prediccion_cruda.map(normalizar)
        grupos = {s: d[d.split == s] for s in ("dev", "test_sk001", "test_otros", "test_control")}
        grupos["no_vistos"] = d[d.split.isin(["test_otros", "test_control"])]
        for fuente, g in d[d.split == "test_control"].groupby("source"):
            grupos[f"test_control:{fuente}"] = g
        for nombre, g in grupos.items():
            c = wer_cer(g.ref, g.hyp)
            s = wer_cer(g.ref.map(sin_tildes), g.hyp.map(sin_tildes))
            filas.append(dict(modelo=f.stem, conjunto=nombre, clips=c["clips"], palabras=c["palabras"],
                              CER=c["cer"], CER_IC=c["cer_ic"], WER=c["wer"], WER_IC=c["wer_ic"],
                              CER_sin_tildes=s["cer"], WER_sin_tildes=s["wer"]))
    t = pd.DataFrame(filas)
    t.to_csv(a.salida, index=False)
    piv = t[t.conjunto.isin(["dev", "test_sk001", "test_otros", "test_control", "no_vistos"])].pivot(
        index="modelo", columns="conjunto", values="CER")[["dev", "test_sk001", "test_otros", "test_control", "no_vistos"]]
    print("CER (%) por modelo y conjunto\n" + piv.sort_values("no_vistos").to_string())


if __name__ == "__main__":
    main()
