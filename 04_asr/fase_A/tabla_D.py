"""Tabla de la Fase D (D25): pseudoetiquetas con validación cruzada por video, contra A3 (misma semilla).

Predicción "cruzada" por semilla: en test_control, cada clip lo transcribe el modelo entrenado con el OTRO pliegue
(D1 = pseudoetiquetas del pliegue 1 evalúa el control del pliegue 2, y al revés); los clips de videos sin pool y el
resto de conjuntos (dev, test_sk001, test_otros) los transcribe D1. Así ningún clip de evaluación lo transcribe un
modelo que oyó su video. D_todo (todas las pseudoetiquetas) se reporta aparte: su test_control está adaptado.

Se hace con y sin LM (salida de decodificar_lm.py), comparando contra A3 decodificado igual.

Uso:  python tabla_D.py --referencias ~/tesis/evaluacion_v1/referencias.csv --fase_D ~/tesis/fase_D \
          --fase_A ~/tesis/fase_A --fase_LM ~/tesis/fase_LM --pseudo ~/tesis/datos/pseudo_v2/todo.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluar import normalizar
from tabla_C import CONJUNTOS, cer_por_conjunto, leer, pareado

SEMILLAS = [1234, 2024, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--fase_D", required=True)
    ap.add_argument("--fase_A", required=True)
    ap.add_argument("--fase_LM", required=True)
    ap.add_argument("--pseudo", required=True, help="todo.csv de pseudoetiquetar_v2.py (pliegue de cada clip)")
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    fD, fA, fL = (Path(x).expanduser() for x in (a.fase_D, a.fase_A, a.fase_LM))
    pliegue = pd.read_csv(Path(a.pseudo).expanduser(), keep_default_na=False).set_index("id").pliegue
    pl = ref.id.map(pliegue).fillna(0).astype(int)
    usa_d2 = (ref.split == "test_control") & (pl == 1)      # control del pliegue 1 -> modelo del pliegue 2
    (fD / "cruzado").mkdir(exist_ok=True)

    fuentes = {"greedy": lambda n, esc: (fA / "A3" / f"{n}.csv") if esc == "A3" else fD / esc / f"{n}.csv"}
    for lm in ("tts", "tts_anlp"):
        fuentes[lm] = lambda n, esc, lm=lm: fL / lm / f"{n}.csv"

    filas, pares = [], []
    for dec, ruta in fuentes.items():
        for s in SEMILLAS:
            rutas = dict(A3=ruta(f"mms-1b-all_real_s{s}", "A3"), D1=ruta(f"mms_pseudo1_s{s}", "D1"),
                         D2=ruta(f"mms_pseudo2_s{s}", "D2"), Dtodo=ruta(f"mms_pseudotodo_s{s}", "Dtodo"))
            if not all(p.exists() for p in rutas.values()):
                print(f"falta {dec} s{s}: {[k for k, p in rutas.items() if not p.exists()]}")
                continue
            p1 = pd.read_csv(rutas["D1"], keep_default_na=False).set_index("id").prediccion_cruda
            p2 = pd.read_csv(rutas["D2"], keep_default_na=False).set_index("id").prediccion_cruda
            cruz = ref[["id"]].assign(prediccion_cruda=[p2[i] if u else p1[i] for i, u in zip(ref.id, usa_d2)])
            cruz_csv = fD / "cruzado" / f"{dec}_s{s}.csv"
            cruz.to_csv(cruz_csv, index=False)
            base = leer(rutas["A3"], ref)
            for nombre, csv in [("A3 solo real", rutas["A3"]), ("D cruzado (pseudo de videos no evaluados)", cruz_csv),
                                ("D todo (test_control adaptado)", rutas["Dtodo"])]:
                d = leer(csv, ref)
                filas.append(dict(decodificado=dec, modelo=nombre, semilla=s, **cer_por_conjunto(d)))
                if nombre != "A3 solo real":
                    dif, lo, hi = pareado(d, base)
                    pares.append(dict(decodificado=dec, modelo=nombre, semilla=s, base="A3", dif_cer_no_vistos=round(dif, 2),
                                      ic_inf=round(lo, 2), ic_sup=round(hi, 2), excluye_0=bool(hi < 0 or lo > 0)))
    t = pd.DataFrame(filas)
    t.round(2).to_csv(fD / "tabla_D_por_semilla.csv", index=False)
    cols = CONJUNTOS + ["no_vistos_sin_tildes"]
    g = t.groupby(["decodificado", "modelo"], sort=False)
    res = g[cols].mean().round(2).astype(str) + " ± " + g[cols].std().round(2).astype(str)
    res.insert(0, "semillas", g.size())
    res.to_csv(fD / "tabla_D.csv")
    p = pd.DataFrame(pares)
    p.to_csv(fD / "pareado_D.csv", index=False)
    pd.set_option("display.width", 250)
    print("CER (%), media ± DE entre semillas\n" + res.to_string())
    print("\nDiferencia pareada en no_vistos contra A3 (mismo decodificado y semilla; negativo = mejora)\n" + p.to_string(index=False))


if __name__ == "__main__":
    main()
