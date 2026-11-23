"""Tabla de la Fase C: CER por escenario (media ± DE entre semillas) y comparación pareada contra la base solo-real
de la misma arquitectura y semilla (MMS: A3; Whisper small: A1), con bootstrap de 5,000 remuestreos sobre clips.

Usa la misma normalización y el mismo CER que evaluar.py. Solo toma las corridas terminadas (con .json).

Uso:  python tabla_C.py --referencias ~/tesis/evaluacion_v1/referencias.csv --fase_C ~/tesis/fase_C --fase_A ~/tesis/fase_A
"""
import argparse
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

from evaluar import normalizar, sin_tildes

CONJUNTOS = ["dev", "test_sk001", "test_otros", "test_control", "no_vistos"]
ESCENARIOS = {"C0": "solo los dos TTS (sin audio real)", "C1":"real + Tacotron2", "C2": "real + MMS-TTS", "C3": "real + los dos TTS", "C5": "real + aumento",
              "C3w": "real + los dos TTS (Whisper small)", "C4": "real + los dos TTS, fracción de frases",
              "C6": "real + MMS-TTS con texto de AmericasNLP"}


def errores(refs, hyps):
    e, n = [], []
    for r, h in zip(refs, hyps):
        c = jiwer.process_characters(r, h)
        e.append(c.substitutions + c.deletions + c.insertions)
        n.append(c.substitutions + c.deletions + c.hits)
    return np.array(e), np.array(n)


def leer(csv, ref):
    p = pd.read_csv(csv, keep_default_na=False)
    d = ref.merge(p, on="id", how="left")
    assert len(p) == len(ref) and not d.prediccion_cruda.isna().any(), csv
    d["hyp"] = d.prediccion_cruda.map(normalizar)
    return d


def cer_por_conjunto(d):
    out = {}
    for s in CONJUNTOS:
        g = d[d.split.isin(["test_otros", "test_control"])] if s == "no_vistos" else d[d.split == s]
        e, n = errores(g.ref, g.hyp)
        out[s] = 100 * e.sum() / n.sum()
    g = d[d.split.isin(["test_otros", "test_control"])]
    e, n = errores(g.ref.map(sin_tildes), g.hyp.map(sin_tildes))
    out["no_vistos_sin_tildes"] = 100 * e.sum() / n.sum()
    return out


def pareado(d, base, n_boot=5000, seed=0):
    """Diferencia de CER en no_vistos (escenario − base) sobre los mismos clips, con IC 95 % por bootstrap."""
    nv = d.split.isin(["test_otros", "test_control"]).values
    e1, n = errores(d.ref[nv], d.hyp[nv])
    e0, _ = errores(base.ref[nv], base.hyp[nv])
    k = np.random.RandomState(seed).randint(0, len(n), size=(n_boot, len(n)))
    dif = 100 * (e1[k].sum(1) - e0[k].sum(1)) / n[k].sum(1)
    return 100 * (e1.sum() - e0.sum()) / n.sum(), *np.percentile(dif, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--fase_C", required=True)
    ap.add_argument("--fase_A", required=True)
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    fC, fA = Path(a.fase_C).expanduser(), Path(a.fase_A).expanduser()

    corridas = [("A3", "solo real (MMS)", f) for f in sorted((fA / "A3").glob("mms-1b-all_real_s*.json"))]
    corridas += [("A1", "solo real (Whisper small)", f) for f in sorted((fA / "A1").glob("whisper-small_real_s*.json"))]
    for esc in ESCENARIOS:
        corridas += [(esc, ESCENARIOS[esc], f) for f in sorted((fC / esc).glob("*.json"))]

    filas, pares, cache = [], [], {}
    for esc, desc, js in corridas:
        nombre = js.stem
        semilla = int(nombre.rsplit("_s", 1)[1])
        if esc == "C4":
            desc = f"{desc} {nombre.split('ambos')[1].split('_s')[0]}"
        d = leer(js.with_suffix(".csv"), ref)
        cache[(esc, semilla)] = d
        filas.append(dict(escenario=esc, descripcion=desc, modelo=nombre, semilla=semilla, **cer_por_conjunto(d)))
        base = {"A3": None, "A1": None, "C3w": "A1"}.get(esc, "A3")
        if base and (base, semilla) in cache:
            dif, lo, hi = pareado(d, cache[(base, semilla)])
            pares.append(dict(escenario=esc, descripcion=desc, modelo=nombre, semilla=semilla, base=base,
                              dif_cer_no_vistos=round(dif, 2), ic_inf=round(lo, 2), ic_sup=round(hi, 2),
                              excluye_0=bool(hi < 0 or lo > 0)))
        if esc == "C6" and ("C2", semilla) in cache:            # mismo TTS, otro texto: ¿importa el dominio del texto?
            dif, lo, hi = pareado(d, cache[("C2", semilla)])
            pares.append(dict(escenario=esc, descripcion=desc, modelo=nombre, semilla=semilla, base="C2",
                              dif_cer_no_vistos=round(dif, 2), ic_inf=round(lo, 2), ic_sup=round(hi, 2),
                              excluye_0=bool(hi < 0 or lo > 0)))
    t = pd.DataFrame(filas)
    t.round(2).to_csv(fC / "tabla_C_por_semilla.csv", index=False)
    cols = CONJUNTOS + ["no_vistos_sin_tildes"]
    g = t.groupby(["escenario", "descripcion"], sort=False)
    res = g[cols].mean().round(2).astype(str) + " ± " + g[cols].std().round(2).astype(str)
    res.insert(0, "semillas", g.size())
    res.to_csv(fC / "tabla_C.csv")
    p = pd.DataFrame(pares)
    p.to_csv(fC / "pareado_C.csv", index=False)
    pd.set_option("display.width", 250)
    print("CER (%), media ± DE entre semillas\n" + res.to_string())
    print("\nDiferencia pareada en no_vistos (escenario − base, misma semilla; negativo = mejora)\n" + p.to_string(index=False))


if __name__ == "__main__":
    main()
