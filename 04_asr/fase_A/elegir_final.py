"""Fase E: elige la configuración final solo con dev (escenario × decodificado, media de las 3 semillas) y la semilla
del modelo final (la de menor CER en dev dentro de esa configuración). Recién después reporta test.

Candidatas: todas las corridas MMS con LM (A3, C1–C6, D1, D2, D todo) en greedy, LM texto TTS y LM TTS+AmericasNLP.
Ojo: alpha/beta/unk del LM se ajustaron en el mismo dev, así que el dev de los decodificados con LM es algo optimista
frente al greedy; no cambia la comparación entre escenarios con el mismo LM.

Uso:  python elegir_final.py --referencias ~/tesis/evaluacion_v1/referencias.csv --lm_dir ~/tesis/fase_LM
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluar import normalizar
from resumen_LM import escenario
from tabla_C import CONJUNTOS, cer_por_conjunto, leer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--lm_dir", required=True)
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    L = Path(a.lm_dir).expanduser()

    filas = []
    for dec in ("greedy", "tts", "tts_anlp"):
        for csv in sorted((L / dec).glob("*_s[0-9]*.csv")):         # corridas, no tabla.csv
            filas.append(dict(escenario=escenario(csv.stem), decodificado=dec, corrida=csv.stem,
                              **cer_por_conjunto(leer(csv, ref))))
    t = pd.DataFrame(filas)
    t.round(3).to_csv(L / "final_por_corrida.csv", index=False)

    g = t.groupby(["escenario", "decodificado"])
    r = g[CONJUNTOS].mean().assign(dev_de=g.dev.std(), semillas=g.size()).sort_values("dev")
    r.round(2).to_csv(L / "final_ranking_dev.csv")
    pd.set_option("display.width", 250)
    print("Ranking por CER en dev (media de semillas); test solo para informar\n" + r.round(2).to_string())

    esc, dec = r.index[0]
    sub = t[(t.escenario == esc) & (t.decodificado == dec)].sort_values("dev")
    print(f"\nConfiguración final (dev): {esc} + {dec}")
    print(f"Modelo final (semilla de menor dev): {sub.corrida.iloc[0]}\n" + sub.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
