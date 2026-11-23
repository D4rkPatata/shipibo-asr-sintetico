"""Fase A0 (sin entrenar): transcribe el conjunto de evaluación con modelos de Hugging Face.

  whisper-small, whisper-medium   no conocen shipibo: token de idioma español, como en el ajuste fino
  mms-1b-all                      wav2vec 2.0 + CTC de Meta, con su adaptador de shipibo (shp)

Salida: <salida>/<modelo>.csv (id, prediccion_cruda) + <modelo>.json (versiones). Decodificación greedy.

Uso:  python a0_hf.py --modelo mms-1b-all --eval ~/tesis/evaluacion_v1 --salida ~/tesis/fase_A/A0
"""
import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

MODELOS = {"whisper-small": "openai/whisper-small", "whisper-medium": "openai/whisper-medium",
           "mms-1b-all": "facebook/mms-1b-all"}


def lotes(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i + n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=MODELOS, required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    from huggingface_hub import model_info
    import transformers
    repo = MODELOS[a.modelo]
    rev = model_info(repo).sha
    ev = Path(a.eval).expanduser()
    audios = pd.read_csv(ev / "audios.csv")
    device = torch.device("cuda")
    leer = lambda t: [sf.read(ev / f, dtype="float32")[0] for f in t.file_name]
    salida = []

    if a.modelo.startswith("whisper"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained(repo, revision=rev)
        model = WhisperForConditionalGeneration.from_pretrained(repo, revision=rev, torch_dtype=torch.float16).to(device).eval()
        for t in lotes(audios, a.batch):
            x = proc.feature_extractor(leer(t), sampling_rate=16000, return_tensors="pt").input_features
            ids = model.generate(x.to(device, torch.float16), language="spanish", task="transcribe",
                                 max_new_tokens=128, num_beams=1)
            salida += proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
    else:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        proc = AutoProcessor.from_pretrained(repo, revision=rev, target_lang="shp")
        model = Wav2Vec2ForCTC.from_pretrained(repo, revision=rev, target_lang="shp",
                                               ignore_mismatched_sizes=True).to(device).eval()
        model.load_adapter("shp")
        for t in lotes(audios, a.batch):
            x = proc(leer(t), sampling_rate=16000, return_tensors="pt", padding=True)
            logits = model(x.input_values.to(device), attention_mask=x.attention_mask.to(device)).logits
            salida += proc.batch_decode(logits.argmax(-1))

    out = Path(a.salida).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": audios.id, "prediccion_cruda": [s.strip() for s in salida]}).to_csv(
        out / f"{a.modelo}.csv", index=False)
    (out / f"{a.modelo}.json").write_text(json.dumps(dict(
        modelo=repo, revision=rev, decodificacion="greedy", idioma="spanish" if "whisper" in a.modelo else "adaptador shp",
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
        python=platform.python_version(), clips=len(audios)), indent=2))
    print(f"{a.modelo}: {len(salida)} transcripciones -> {out}")


if __name__ == "__main__":
    main()
