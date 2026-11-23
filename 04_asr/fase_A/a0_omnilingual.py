"""Fase A0 (sin entrenar): transcribe el conjunto de evaluación con Omnilingual ASR de Meta (2025).

  omniASR_CTC_1B_v2    wav2vec 2.0 + CTC (no recibe idioma)
  omniASR_LLM_1B_v2    wav2vec 2.0 + decodificador LLM, condicionado a shipibo (shp_Latn)
  omniASR_LLM_7B_v2    ídem, 7B (~17 GB de VRAM en inferencia)

Requiere un entorno propio (fairseq2 fija su versión de PyTorch):  pip install omnilingual-asr

Salida: <salida>/<modelo>.csv (id, prediccion_cruda) + <modelo>.json (versiones).

Uso:  python a0_omnilingual.py --modelo omniASR_LLM_7B_v2 --eval evaluacion_v1 --salida predicciones
"""
import argparse
import importlib.metadata as md
import json
import platform
from pathlib import Path

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, help="p. ej. omniASR_CTC_1B_v2, omniASR_LLM_1B_v2, omniASR_LLM_7B_v2")
    ap.add_argument("--eval", required=True, help="carpeta evaluacion_v1 (con audios.csv y wavs/)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    ev = Path(a.eval).expanduser()
    audios = pd.read_csv(ev / "audios.csv")
    assert audios.duration_s.max() < 40, "Omnilingual solo acepta audio de menos de 40 s"
    rutas = [str(ev / f) for f in audios.file_name]
    pipe = ASRInferencePipeline(model_card=a.modelo)
    usa_idioma = "_LLM_" in a.modelo
    salida = []
    for i in range(0, len(rutas), a.batch):
        trozo = rutas[i:i + a.batch]
        kw = {"lang": ["shp_Latn"] * len(trozo)} if usa_idioma else {}
        salida += pipe.transcribe(trozo, batch_size=len(trozo), **kw)
        print(f"  {min(i + a.batch, len(rutas))}/{len(rutas)}", flush=True)

    out = Path(a.salida).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": audios.id, "prediccion_cruda": [str(s).strip() for s in salida]}).to_csv(
        out / f"{a.modelo}.csv", index=False)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (out / f"{a.modelo}.json").write_text(json.dumps(dict(
        modelo=a.modelo, idioma="shp_Latn" if usa_idioma else "(CTC: sin idioma)", gpu=gpu, torch=torch.__version__,
        omnilingual_asr=md.version("omnilingual-asr"), fairseq2=md.version("fairseq2"),
        python=platform.python_version(), clips=len(audios)), indent=2))
    print(f"{a.modelo}: {len(salida)} transcripciones -> {out}")


if __name__ == "__main__":
    main()
