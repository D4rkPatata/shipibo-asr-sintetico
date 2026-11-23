"""Prueba 1: CER/WER relajados. ¿Cambian las conclusiones si no se cuentan como error las diferencias de convención?

Métricas (cada una aplica la misma transformación a referencia y predicción):
  estandar      CER / WER de siempre (la métrica principal no cambia)
  sin_tildes    sin tildes (la ortografía oficial no marca acento)
  minedu        sin tildes + grafías no oficiales a su equivalente del alfabeto oficial de 19 grafías
                (RM 303-2015-MINEDU: a e i o b ch j k m n p r s sh t ts w x y): qu→k, c (no ch)→k, hu+vocal→w,
                vocal+u+vocal→w, v→b, z→s, u→o. Iguala escrituras antiguas (aquinti/akinti) sin romper préstamos (se aplica a ambos lados)
  sin_espacios  CER sin espacios (no cuenta dónde se cortan las palabras)
  segmentacion  error de palabra tolerante a la segmentación: una palabra de la referencia está bien si todas sus
                letras se reconocieron (aunque el modelo la una a otra o la parta) y no se le insertaron letras
                (las inserciones en el borde entre palabras se atribuyen a la palabra anterior)
  relajado      minedu + tolerante a la segmentación (WER) y minedu + sin espacios (CER)

Sistemas: A0 (sin entrenar), A1/A2/C3w (Whisper), todas las corridas MMS con greedy y los dos LM, Omnilingual 7B.
Se reporta no_vistos (media de semillas) y diferencias pareadas (bootstrap 5,000) para las comparaciones clave.

Uso:  python metricas_relajadas.py --referencias ~/tesis/evaluacion_v1/referencias.csv --tesis ~/tesis \
          --salida ~/tesis/fase_E/metricas_relajadas
"""
import argparse
import re
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

from evaluar import normalizar, sin_tildes
from resumen_LM import escenario

NO_VISTOS = ["test_otros", "test_control"]
VOC = "aeio"


def minedu(t):
    t = sin_tildes(t)
    t = t.replace("qu", "k")
    t = re.sub(r"c(?!h)", "k", t)
    t = re.sub(rf"hu(?=[{VOC}u])", "w", t)
    t = re.sub(rf"(?<=[{VOC}])u(?=[{VOC}])", "w", t)          # vocal + u + vocal: jauaquibo -> jawakibo
    return t.replace("v", "b").replace("z", "s").replace("u", "o")


def cer_par(r, h):
    """(errores, caracteres de la referencia) a nivel de caracteres."""
    if not r:
        return len(h), 0
    o = jiwer.process_characters(r, h)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def wer_par(r, h):
    if not r:
        return len(h.split()), 0
    o = jiwer.process_words(r, h)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def seg_par(r, h):
    """Palabras de la referencia no reconocidas por completo, alineando las letras sin espacios."""
    palabras = r.split()
    if not palabras:
        return 0, 0
    dueño, inicio = [], set()
    for k, w in enumerate(palabras):
        inicio.add(len(dueño))
        dueño += [k] * len(w)
    rr, hh = "".join(palabras), h.replace(" ", "")
    mal = set()
    if not hh:
        return len(palabras), len(palabras)
    for a in jiwer.process_characters(rr, hh).alignments[0]:
        if a.type == "insert":
            p = a.ref_start_idx                           # en el borde, se le atribuye a la palabra anterior
            mal.add(dueño[p] if p < len(dueño) and (p not in inicio or p == 0) else dueño[p - 1])
        elif a.type != "equal":
            mal.update(dueño[p] for p in range(a.ref_start_idx, a.ref_end_idx))
    return len(mal), len(palabras)


METRICAS = {
    "CER estandar": lambda r, h: cer_par(r, h),
    "CER sin_tildes": lambda r, h: cer_par(sin_tildes(r), sin_tildes(h)),
    "CER minedu": lambda r, h: cer_par(minedu(r), minedu(h)),
    "CER sin_espacios": lambda r, h: cer_par(r.replace(" ", ""), h.replace(" ", "")),
    "CER relajado": lambda r, h: cer_par(minedu(r).replace(" ", ""), minedu(h).replace(" ", "")),
    "WER estandar": lambda r, h: wer_par(r, h),
    "WER sin_tildes": lambda r, h: wer_par(sin_tildes(r), sin_tildes(h)),
    "WER minedu": lambda r, h: wer_par(minedu(r), minedu(h)),
    "WER segmentacion": lambda r, h: seg_par(r, h),
    "WER relajado": lambda r, h: seg_par(minedu(r), minedu(h)),
}


def sistemas(T):
    """(familia, escenario, decodificado, semilla, csv)."""
    out = []
    for f, n in [("mms-1b-all", "MMS sin entrenar"), ("omniASR_LLM_1B_v2", "Omnilingual LLM 1B sin entrenar"),
                 ("whisper-medium", "Whisper medium sin entrenar")]:
        out.append(("A0", n, "greedy", "-", T / "fase_A" / "A0" / f"{f}.csv"))
    for carpeta, n in [("fase_A/A1", "A1 Whisper small real"), ("fase_A/A2", "A2 Whisper medium real"),
                       ("fase_C/C3w", "C3w Whisper small + los dos TTS")]:
        for c in sorted((T / carpeta).glob("whisper-*_s*.csv")):
            out.append(("Whisper", n, "greedy", c.stem.rsplit("_s", 1)[1], c))
    for dec in ("greedy", "tts", "tts_anlp"):
        for c in sorted((T / "fase_LM" / dec).glob("*_s[0-9]*.csv")):
            out.append(("MMS", escenario(c.stem), dec, c.stem.rsplit("_s", 1)[1], c))
    for f, n in [("omniASR_LLM_7B_v2", "Omnilingual 7B sin entrenar"),
                 ("omniASR_LLM_7B_v2_shp_lora_real", "Omnilingual 7B LoRA real (asesor)")]:
        out.append(("7B", n, "propio", "1234", T / "omni7b" / f"{f}.csv"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--tesis", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    T, sal = Path(a.tesis).expanduser(), Path(a.salida).expanduser()
    sal.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref = ref[ref.split.isin(NO_VISTOS)].reset_index(drop=True)
    ref["ref"] = ref.text_norm.map(normalizar)

    # ejemplo de lo que cambia la regla minedu (para revisar a mano)
    cambia = sorted({(w, minedu(w)) for t in ref.ref for w in t.split() if minedu(w) != sin_tildes(w)})
    pd.DataFrame(cambia, columns=["palabra", "minedu"]).to_csv(sal / "palabras_minedu.csv", index=False)

    errores = {}   # (escenario, dec, semilla) -> {metrica: (e[], n[])}
    filas = []
    for fam, esc, dec, sem, csv in sistemas(T):
        p = pd.read_csv(csv, keep_default_na=False).set_index("id").prediccion_cruda
        hyp = [normalizar(p[i]) for i in ref.id]
        por = {}
        for m, f in METRICAS.items():
            en = np.array([f(r, h) for r, h in zip(ref.ref, hyp)])
            por[m] = (en[:, 0], en[:, 1])
        errores[(esc, dec, sem)] = por
        filas.append(dict(familia=fam, escenario=esc, decodificado=dec, semilla=sem,
                          **{m: 100 * e.sum() / n.sum() for m, (e, n) in por.items()}))
    t = pd.DataFrame(filas)
    t.round(2).to_csv(sal / "por_corrida.csv", index=False)
    g = t.groupby(["familia", "escenario", "decodificado"], sort=False)[list(METRICAS)].mean().round(2)
    g.insert(0, "semillas", t.groupby(["familia", "escenario", "decodificado"], sort=False).size())
    g.to_csv(sal / "resumen.csv")

    # comparaciones clave, pareadas por semilla
    comparaciones = [
        ("A3 solo real", "greedy", "A1 Whisper small real", "greedy", "MMS afinado vs Whisper small"),
        ("C3 los dos TTS", "greedy", "A3 solo real", "greedy", "sintético (C3) vs solo real, sin LM"),
        ("C6 + MMS-TTS texto AmericasNLP", "tts_anlp", "A3 solo real", "tts_anlp", "C6 vs solo real, con LM"),
        ("A3 solo real", "tts", "A3 solo real", "greedy", "LM vs sin LM (A3)"),
        ("C3w Whisper small + los dos TTS", "greedy", "A1 Whisper small real", "greedy", "sintético con Whisper"),
        ("Omnilingual 7B LoRA real (asesor)", "propio", "A3 solo real", "tts", "7B vs final (s2024 + LM)"),
    ]
    rng = np.random.RandomState(0)
    k = rng.randint(0, len(ref), size=(5000, len(ref)))
    pares = []
    for e1, d1, e0, d0, nombre in comparaciones:
        for sem in ("1234", "2024", "7"):
            s1 = "1234" if e1.startswith("Omnilingual") else sem
            if e1.startswith("Omnilingual") and sem != "2024":
                continue                                       # el 7B (1 corrida) contra el modelo final s2024
            if (e1, d1, s1) not in errores or (e0, d0, sem) not in errores:
                continue
            for m in METRICAS:
                (x1, n), (x0, _) = errores[(e1, d1, s1)][m], errores[(e0, d0, sem)][m]
                dif = 100 * (x1[k].sum(1) - x0[k].sum(1)) / n[k].sum(1)
                lo, hi = np.percentile(dif, [2.5, 97.5])
                pares.append(dict(comparacion=nombre, semilla=sem, metrica=m, dif=100 * (x1.sum() - x0.sum()) / n.sum(),
                                  sig=bool(hi < 0 or lo > 0)))
    p = pd.DataFrame(pares)
    p.to_csv(sal / "pareado_por_semilla.csv", index=False)
    rp = p.groupby(["comparacion", "metrica"], sort=False).agg(dif=("dif", "mean"), sig=("sig", "sum"), n=("sig", "size"))
    rp["texto"] = rp.dif.round(2).astype(str) + " (" + rp.sig.astype(str) + "/" + rp.n.astype(str) + ")"
    tabla_p = rp.texto.unstack("metrica")[list(METRICAS)]
    tabla_p.to_csv(sal / "pareado.csv")

    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 200)
    texto = ("CER/WER (%) en no_vistos, media de semillas\n" + g.to_string()
             + "\n\nDiferencias pareadas (media entre semillas; entre paréntesis, semillas con IC 95 % que excluye 0)\n"
             + tabla_p.to_string()
             + f"\n\nPalabras de no_vistos que cambia la regla minedu: {len(cambia)} (palabras_minedu.csv)")
    (sal / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
