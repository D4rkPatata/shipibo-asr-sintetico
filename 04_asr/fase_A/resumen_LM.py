"""Resumen del modelo de lenguaje por escenario (media de las 3 semillas) y prueba de robustez del LM con AmericasNLP.

Robustez: parte del workshop (test_otros) usa frases de plantilla parecidas a las de AmericasNLP
(p. ej. "enra mia boti atipanyamake" vs "enra mia meniti atipanyamake"). Se repite la comparación en no_vistos sin
los clips cuya transcripción tiene similitud >= 0.8 (difflib) con alguna línea de AmericasNLP.

Diferencias pareadas (bootstrap 5,000, misma corrida y mismos logits): tts − greedy, tts_anlp − greedy y
tts_anlp − tts (lo que agrega AmericasNLP sobre el LM del texto del TTS).

Uso:  python resumen_LM.py --referencias ~/tesis/evaluacion_v1/referencias.csv --lm_dir ~/tesis/fase_LM \
          --anlp ~/tesis/datos/americasnlp_detalle.csv
"""
import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluar import normalizar
from tabla_C import errores, leer

ESCENARIO = [("mms-1b-all_real", "A3 solo real"), ("mms_solosint", "C0 solo sintético"),("mms_tacotron", "C1 + Tacotron2"), ("mms_mmstts-anlp", "C6 + MMS-TTS texto AmericasNLP"),
             ("mms_mmstts", "C2 + MMS-TTS"), ("mms_ambos0.25", "C4 los dos TTS 25 %"), ("mms_ambos0.5", "C4 los dos TTS 50 %"),
             ("mms_ambos", "C3 los dos TTS"), ("mms_aug", "C5 + aumento"),
             ("mms_pseudo1", "D1 pseudo pliegue 1"), ("mms_pseudo2", "D2 pseudo pliegue 2"), ("mms_pseudotodo", "D todo")]


def escenario(nombre):
    base = re.sub(r"_s\d+$", "", nombre)
    return next(e for k, e in ESCENARIO if base == k)


def cer_mask(d, m):
    e, n = errores(d.ref[m], d.hyp[m])
    return 100 * e.sum() / n.sum()


def pareado(d, base, m, n_boot=5000, seed=0):
    e1, n = errores(d.ref[m], d.hyp[m])
    e0, _ = errores(base.ref[m], base.hyp[m])
    k = np.random.RandomState(seed).randint(0, len(n), size=(n_boot, len(n)))
    dif = 100 * (e1[k].sum(1) - e0[k].sum(1)) / n[k].sum(1)
    lo, hi = np.percentile(dif, [2.5, 97.5])
    return 100 * (e1.sum() - e0.sum()) / n.sum(), bool(hi < 0 or lo > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument("--anlp", required=True, help="americasnlp_detalle.csv (todas las líneas, text_norm)")
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    L = Path(a.lm_dir).expanduser()

    anlp = set(pd.read_csv(Path(a.anlp).expanduser(), keep_default_na=False).text_norm) - {""}
    por_largo = {}
    for x in anlp:
        por_largo.setdefault(len(x.split()), []).append(x)

    def similitud(t):
        n = len(t.split())
        return max((difflib.SequenceMatcher(None, t, c).ratio() for k in (n - 1, n, n + 1) for c in por_largo.get(k, [])),
                   default=0)
    nv = ref.split.isin(["test_otros", "test_control"])
    ref["sim_anlp"] = 0.0
    ref.loc[nv, "sim_anlp"] = ref.loc[nv, "ref"].map(similitud)
    parecido = nv & (ref.sim_anlp >= 0.8)
    print(f"clips de no_vistos parecidos (>= 0.8) a AmericasNLP: {int(parecido.sum())} "
          f"({ref[parecido].split.value_counts().to_dict()}, fuentes {ref[parecido].source.value_counts().to_dict()})")
    mascaras = {"no_vistos": nv.values, "test_control": (ref.split == "test_control").values,
                "no_vistos_sin_parecidos": (nv & ~parecido).values}

    filas, pares = [], []
    for g in sorted((L / "greedy").glob("*.csv")):
        dec = {"greedy": leer(g, ref)}
        for lm in ("tts", "tts_anlp"):
            if (L / lm / g.name).exists():
                dec[lm] = leer(L / lm / g.name, ref)
        if len(dec) < 3:
            continue
        esc = escenario(g.stem)
        for nombre, d in dec.items():
            filas.append(dict(escenario=esc, corrida=g.stem, decodificado=nombre,
                              **{k: cer_mask(d, m) for k, m in mascaras.items()}))
        for a_, b_ in [("tts", "greedy"), ("tts_anlp", "greedy"), ("tts_anlp", "tts")]:
            for k, m in mascaras.items():
                dif, sig = pareado(dec[a_], dec[b_], m)
                pares.append(dict(escenario=esc, corrida=g.stem, comparacion=f"{a_} − {b_}", conjunto=k, dif=dif, sig=sig))

    t = pd.DataFrame(filas)
    orden = [e for _, e in ESCENARIO if e in set(t.escenario)]
    r = t.groupby(["escenario", "decodificado"])[list(mascaras)].agg(["mean", "std"]).round(2)
    r = r.reindex(pd.MultiIndex.from_product([orden, ["greedy", "tts", "tts_anlp"]])).dropna(how="all")
    p = pd.DataFrame(pares)
    rp = p.groupby(["escenario", "comparacion", "conjunto"], sort=False).agg(
        dif_media=("dif", "mean"), semillas_sig=("sig", "sum"), semillas=("sig", "size")).round(2).reset_index()
    t.round(2).to_csv(L / "resumen_LM_por_corrida.csv", index=False)
    r.to_csv(L / "resumen_LM.csv")
    rp.to_csv(L / "resumen_LM_pareado.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nCER (%), media y DE entre semillas\n" + r.to_string())
    print("\nDiferencias pareadas (media entre semillas; semillas_sig = con IC 95 % que excluye 0)\n"
          + rp.pivot_table(index=["escenario", "comparacion"], columns="conjunto", values="dif_media", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
