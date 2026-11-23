"""Une las dos mitades de Tacotron2 (Colab: orden[:11779]; servidor: orden[11779:]) en una sola carpeta.

Comprueba que no se repita ninguna frase, que sumen 23,558, que la semilla de cada frase sea la de comun.semilla_de
y que exista cada wav. Los wavs se enlazan (hard link) en vez de copiarse: no ocupa espacio extra y los originales
quedan intactos. El audio sigue a 22,050 Hz; el remuestreo a 16 kHz va en el filtrado.

Uso:  python unir_tacotron.py --colab ~/tesis/tts/tacotron_colab/audio_generado \
          --servidor ~/tesis/tts/tacotron_parte2 --salida ~/tesis/tts/tacotron
"""
import argparse
import os
from pathlib import Path

import pandas as pd

import comun as K

N_TOTAL = 23558


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colab", required=True)
    ap.add_argument("--servidor", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    partes = []
    for raiz, corrida in [(a.colab, "colab_parte1"), (a.servidor, "servidor_parte2")]:
        raiz = Path(raiz).expanduser()
        m = pd.read_csv(raiz / "manifest.csv")
        m["corrida"] = corrida
        m["raiz_origen"] = str(raiz)
        partes.append(m)
    m = pd.concat(partes, ignore_index=True)
    assert m.id.is_unique, "hay frases repetidas entre las dos mitades"
    assert len(m) == N_TOTAL, f"{len(m)} frases, se esperaban {N_TOTAL}"
    assert (m.seed == m.id.map(K.semilla_de)).all(), "semillas distintas a comun.semilla_de"

    salida = Path(a.salida).expanduser()
    for r in m.itertuples():
        origen, destino = Path(r.raiz_origen) / r.archivo, salida / r.archivo
        assert origen.is_file(), f"falta {origen}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if not destino.exists():
            os.link(origen, destino)
    m.drop(columns="raiz_origen").sort_values("id").to_csv(salida / "manifest.csv", index=False)
    print(m.groupby("corrida").estado.value_counts().to_string())
    print(f"{len(m)} frases -> {salida} | horas ok: {m[m.estado == 'ok'].dur_s.sum() / 3600:.2f}")


if __name__ == "__main__":
    main()
