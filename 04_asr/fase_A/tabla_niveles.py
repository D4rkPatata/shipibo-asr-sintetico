"""Tabla de la prueba 2: LM de palabra, subpalabra y carácter (lm_niveles.py) frente a greedy, en no_vistos.

Para cada escenario decodificado (todas las corridas MMS con logits) y texto del LM (tts, tts_anlp): CER y WER estándar y relajados (metricas_relajadas.py),
% de palabras acertadas según dónde aparecen (train real / solo texto del LM / solo AmericasNLP / nueva, como en
analizar_errores.py) y diferencias pareadas por semilla contra el LM de palabras (mismo decodificador) y contra greedy.

Uso:  python tabla_niveles.py --referencias ~/tesis/evaluacion_v1/referencias.csv --niveles ~/tesis/fase_LM_niveles \
          --greedy ~/tesis/fase_LM/greedy --datos ~/tesis/datos --lm_texto ~/tesis/lm
"""
import argparse
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

from evaluar import normalizar
from metricas_relajadas import METRICAS
from resumen_LM import escenario

NO_VISTOS = ["test_otros", "test_control"]
CLAVE = ["CER estandar", "CER relajado", "WER estandar", "WER segmentacion", "WER relajado"]


def acierto_palabras(refs, hyps, origen):
    o = jiwer.process_words(list(refs), list(hyps))
    ok = {}
    for r, al in zip(o.references, o.alignments):
        for a in al:
            for k in range(a.ref_start_idx, a.ref_end_idx):
                g = origen(r[k])
                n, b = ok.get(g, (0, 0))
                ok[g] = (n + 1, b + (a.type == "equal"))
    return {g: 100 * b / n for g, (n, b) in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--niveles", required=True)
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--lm_texto", required=True)
    a = ap.parse_args()
    N, G, D, T = (Path(x).expanduser() for x in (a.niveles, a.greedy, a.datos, a.lm_texto))
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref = ref[ref.split.isin(NO_VISTOS)].reset_index(drop=True)
    ref["ref"] = ref.text_norm.map(normalizar)

    split = pd.read_csv(D / "split_v1.1.csv")
    meta = pd.read_csv(D / "dataset_v1.1" / "metadata.csv", keep_default_na=False)
    vocab = lambda textos: {w for x in textos for w in normalizar(x).split()}
    v_train = vocab(meta[meta.id.isin(split[split.split == "train"].id)].text_norm)
    v_lm = vocab((T / "texto_tts_train.txt").read_text(encoding="utf-8").splitlines())
    v_anlp = vocab((T / "texto_anlp.txt").read_text(encoding="utf-8").splitlines())
    origen = lambda w: ("1 train real" if w in v_train else "2 solo texto LM" if w in v_lm
                        else "3 solo AmericasNLP" if w in v_anlp else "4 nueva")

    def leer(csv):
        p = pd.read_csv(csv, keep_default_na=False).set_index("id").prediccion_cruda
        return [normalizar(p[i]) for i in ref.id]

    filas, errores = [], {}
    corridas = sorted({c.stem for c in N.glob("*_palabra/*_s[0-9]*.csv")})
    for corrida in corridas:
        if True:
            esc, s = escenario(corrida), corrida.rsplit("_s", 1)[1]
            variantes = {("-", "greedy"): G / f"{corrida}.csv"}
            for texto in ("tts", "tts_anlp"):
                for nivel in ("palabra", "subpalabra", "caracter"):
                    variantes[(texto, nivel)] = N / f"{texto}_{nivel}" / f"{corrida}.csv"
            for (texto, nivel), csv in variantes.items():
                if not csv.exists():
                    continue
                hyp = leer(csv)
                por = {}
                for m in CLAVE:
                    en = np.array([METRICAS[m](r, h) for r, h in zip(ref.ref, hyp)])
                    por[m] = (en[:, 0], en[:, 1])
                errores[(esc, s, texto, nivel)] = por
                filas.append(dict(escenario=esc, semilla=s, texto=texto, nivel=nivel,
                                  **{m: 100 * e.sum() / n.sum() for m, (e, n) in por.items()},
                                  **acierto_palabras(ref.ref, hyp, origen)))
    t = pd.DataFrame(filas)
    t.round(2).to_csv(N / "tabla_niveles_por_corrida.csv", index=False)
    cols = CLAVE + sorted(c for c in t.columns if c[0].isdigit())
    g = t.groupby(["escenario", "texto", "nivel"], sort=False)
    r = g[cols].mean().round(2)
    r.insert(0, "semillas", g.size())
    r.to_csv(N / "tabla_niveles.csv")

    k = np.random.RandomState(0).randint(0, len(ref), size=(5000, len(ref)))
    pares = []
    for (esc, s, texto, nivel), por in errores.items():
        if texto == "-":
            continue
        for base_t, base_n, nombre in [(texto, "palabra", "vs LM palabra"), ("-", "greedy", "vs greedy")]:
            if (base_t, base_n) == (texto, nivel) or (esc, s, base_t, base_n) not in errores:
                continue
            for m in CLAVE:
                (x1, n), (x0, _) = por[m], errores[(esc, s, base_t, base_n)][m]
                dif = 100 * (x1[k].sum(1) - x0[k].sum(1)) / n[k].sum(1)
                lo, hi = np.percentile(dif, [2.5, 97.5])
                pares.append(dict(escenario=esc, texto=texto, nivel=nivel, comparacion=nombre, semilla=s, metrica=m,
                                  dif=100 * (x1.sum() - x0.sum()) / n.sum(), sig=bool(hi < 0 or lo > 0)))
    p = pd.DataFrame(pares)
    p.to_csv(N / "pareado_niveles_por_semilla.csv", index=False)
    rp = p.groupby(["escenario", "texto", "nivel", "comparacion", "metrica"], sort=False).agg(
        dif=("dif", "mean"), sig=("sig", "sum"), n=("sig", "size"))
    rp["texto_"] = rp.dif.round(2).astype(str) + " (" + rp.sig.astype(str) + "/" + rp.n.astype(str) + ")"
    tp = rp.texto_.unstack("metrica")[CLAVE]
    tp.to_csv(N / "pareado_niveles.csv")

    pd.set_option("display.width", 300)
    texto = ("no_vistos, media de 3 semillas. CER/WER en %; columnas 1–4: % de palabras de la referencia acertadas según "
             "dónde aparecen\n" + r.to_string()
             + "\n\nDiferencias pareadas (media; semillas con IC 95 % que excluye 0)\n" + tp.to_string())
    (N / "tabla_niveles.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
