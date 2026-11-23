"""Tabla del modelo de lenguaje: CER por corrida con greedy y con cada LM (salida de decodificar_lm.py), y la
diferencia pareada LM − greedy en no_vistos sobre los mismos logits (bootstrap 5,000).

Uso:  python tabla_LM.py --referencias ~/tesis/evaluacion_v1/referencias.csv --lm_dir ~/tesis/fase_LM
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluar import normalizar
from tabla_C import cer_por_conjunto, leer, pareado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--lm_dir", required=True)
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    L = Path(a.lm_dir).expanduser()
    lms = sorted(p.name for p in L.iterdir() if p.is_dir() and p.name not in ("greedy", "logits"))
    hp = pd.read_csv(L / "hiperparametros.csv")
    filas, pares = [], []
    for g in sorted((L / "greedy").glob("*.csv")):
        base = leer(g, ref)
        filas.append(dict(corrida=g.stem, decodificado="greedy", **cer_por_conjunto(base)))
        for lm in lms:
            f = L / lm / g.name
            if not f.exists():
                continue
            d = leer(f, ref)
            filas.append(dict(corrida=g.stem, decodificado=lm, **cer_por_conjunto(d)))
            dif, lo, hi = pareado(d, base)
            pares.append(dict(corrida=g.stem, lm=lm, dif_cer_no_vistos=round(dif, 2), ic_inf=round(lo, 2),
                              ic_sup=round(hi, 2), excluye_0=bool(hi < 0 or lo > 0)))
    t = pd.DataFrame(filas).round(2)
    t.to_csv(L / "tabla_LM.csv", index=False)
    p = pd.DataFrame(pares).merge(hp, left_on=["corrida", "lm"], right_on=["corrida", "lm"], how="left")
    p.to_csv(L / "pareado_LM.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    print("\nLM − greedy en no_vistos (negativo = mejora)\n" + p.to_string(index=False))


if __name__ == "__main__":
    main()
