"""Arma el conjunto de evaluación fijo de la tesis: los 4 tests en una sola carpeta.

  dev, test_sk001, test_otros  ← dataset_v1.1 + split_v1.1.csv
  test_control                 ← no_pareado_v1 (clips con transcripción humana)

Salida (en SHIPIBO_DATOS):
  evaluacion_v1/wavs/<id>.wav    16 kHz mono (copia exacta, sin reprocesar)
  evaluacion_v1/referencias.csv  id, split, file_name, text_norm, source, speaker_group, duration_s
  evaluacion_v1/audios.csv       lo mismo SIN el texto: lo que se entrega a terceros (evaluación ciega)
  evaluacion_v1.zip              solo wavs + audios.csv + LEEME.md (sin referencias)

Uso:  python preparar_tests.py
"""
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SHIPIBO_DATOS", r"D:\tesis\Code"))
OUT = ROOT / "evaluacion_v1"


def main():
    m = pd.read_csv(ROOT / "dataset_v1.1" / "metadata.csv").merge(pd.read_csv(ROOT / "split_v1.1.csv"), on="id")
    real = m[m.split.isin(["dev", "test_sk001", "test_otros"])].assign(raiz=str(ROOT / "dataset_v1.1"))
    c = pd.read_csv(ROOT / "no_pareado_v1" / "metadata_control.csv", keep_default_na=False)
    c = c.assign(split="test_control", speaker_group="NP-" + c.subsource, raiz=str(ROOT / "no_pareado_v1"))
    cols = ["id", "split", "file_name", "text_norm", "source", "speaker_group", "duration_s", "raiz"]
    t = pd.concat([real[cols], c[cols]], ignore_index=True)
    assert t.id.is_unique and (t.text_norm.str.len() > 0).all()

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "wavs").mkdir(parents=True)
    for r in t.itertuples():
        shutil.copy2(Path(r.raiz) / r.file_name, OUT / "wavs" / f"{r.id}.wav")
    t["file_name"] = "wavs/" + t.id + ".wav"
    t = t.drop(columns="raiz")
    t.to_csv(OUT / "referencias.csv", index=False, encoding="utf-8")
    t.drop(columns="text_norm").to_csv(OUT / "audios.csv", index=False, encoding="utf-8")

    r = t.groupby("split").agg(clips=("id", "size"), minutos=("duration_s", lambda s: round(s.sum() / 60, 1)),
                               palabras=("text_norm", lambda s: int(s.str.split().str.len().sum())))
    r = r.reindex(["dev", "test_sk001", "test_otros", "test_control"])
    print(r.to_string())
    tabla = "| partición | clips | minutos | palabras |\n|---|---|---|---|\n" + "\n".join(
        f"| {s} | {x.clips} | {x.minutos} | {x.palabras} |" for s, x in r.iterrows())
    (OUT / "LEEME.md").write_text(LEEME.format(tabla=tabla, total=len(t),
                                              minutos=round(t.duration_s.sum() / 60, 1)), encoding="utf-8")
    with zipfile.ZipFile(ROOT / "evaluacion_v1.zip", "w", zipfile.ZIP_STORED) as z:
        for f in sorted((OUT / "wavs").glob("*.wav")):
            z.write(f, f"evaluacion_v1/wavs/{f.name}")
        for f in ("audios.csv", "LEEME.md"):
            z.write(OUT / f, f"evaluacion_v1/{f}")
    print(f"{len(t)} clips -> {ROOT / 'evaluacion_v1.zip'} ({(ROOT / 'evaluacion_v1.zip').stat().st_size / 1e6:.0f} MB)")


LEEME = """# Conjunto de evaluación — ASR en Shipibo-Konibo (tesis PUCP, D. Malca)

{total} clips, {minutos} min. Audio WAV, 16 kHz, mono, PCM 16-bit, de 1 a 23 s (todos < 40 s).

{tabla}

| partición | qué es |
|---|---|
| dev | frases de la hablante del corpus SK001 (Menéndez & Gómez, 2025), no vistas en entrenamiento |
| test_sk001 | ídem, conjunto de prueba |
| test_otros | grupos de hablante de YouTube y de un workshop, no vistos en entrenamiento |
| test_control | audio de campañas de salud y electorales (MINCUL, PNUD), cuentos (MIDIS Cuna Más) y relatos religiosos (Global Recordings Network), transcrito por un hablante nativo |

`audios.csv`: id, split, file_name, source, speaker_group, duration_s. **No incluye las transcripciones**: la
evaluación es ciega; las predicciones se puntúan después con las referencias.

Uso exclusivamente académico; el audio de YouTube, SoundCloud y GRN no se redistribuye.
"""

if __name__ == "__main__":
    main()
