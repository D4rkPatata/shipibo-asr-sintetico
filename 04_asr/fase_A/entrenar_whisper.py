"""Ajuste fino de Whisper con el protocolo fijo de la tesis (Fase A y escenarios C/D).

Protocolo (igual en todas las corridas, ver docs/decisiones.md D10 y D16):
  - presupuesto fijo de pasos (sin parada temprana), lote efectivo 32, LR 1e-5, 200 pasos de calentamiento,
    SpecAugment, bf16, token de idioma español
  - evaluación en dev cada 100 pasos; se conserva el checkpoint de menor CER en dev
  - al terminar transcribe evaluacion_v1 (greedy, igual que A0) para puntuarlo con evaluar.py

Uso (A1, solo real):
  python entrenar_whisper.py --modelo openai/whisper-small --semilla 1234 \
      --datos ~/tesis/datos/dataset_v1.1 --split ~/tesis/datos/split_v1.1.csv \
      --eval ~/tesis/evaluacion_v1 --salida ~/tesis/fase_A/A1 --nombre whisper-small_real_s1234
"""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import asr_core as C

REVISIONES = {"openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
              "openai/whisper-medium": "abdf7c39ab9d0397620ccaea8974cc764cd0953e"}


def metricas_cer(processor):
    """Como C.metricas_trainer, pero también devuelve CER: es la métrica de selección (D16)."""
    import jiwer
    tok = processor.tokenizer

    def f(pred):
        ids = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)
        lab = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        hyps = [C.normalizar(t) for t in tok.batch_decode(ids, skip_special_tokens=True)]
        refs = [C.normalizar(t) for t in tok.batch_decode(lab, skip_special_tokens=True)]
        par = [(r, h) for r, h in zip(refs, hyps) if r]
        r, h = [p[0] for p in par], [p[1] for p in par]
        return {"wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--datos", required=True, help="carpeta con metadata.csv y wavs/")
    ap.add_argument("--split", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--pasos", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--acum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_cada", type=int, default=100)
    ap.add_argument("--agregado", nargs="*", default=[], help="carpetas con metadata.csv (sintético filtrado o aumento)")
    ap.add_argument("--frac_agregado", type=float, default=1.0, help="fracción de frases del agregado (C4)")
    ap.add_argument("--prop_real", type=float, default=0.5, help="proporción de clips reales en cada lote (D10)")
    a = ap.parse_args()

    import transformers
    from huggingface_hub import model_info
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    rev = REVISIONES.get(a.modelo) or model_info(a.modelo).sha
    set_seed(a.semilla)
    datos = Path(a.datos).expanduser()
    meta = pd.read_csv(datos / "metadata.csv").merge(pd.read_csv(Path(a.split).expanduser()), on="id")
    train, dev = meta[meta.split == "train"], meta[meta.split == "dev"]
    real_train = train
    if a.agregado:      # Fase C: lista fija y barajada, 50 % real / 50 % agregado en cada micro-lote
        agregado = C.leer_agregado(a.agregado, a.frac_agregado)
        train = C.orden_mezcla(real_train.assign(file_name=[str(datos / f) for f in real_train.file_name]),
                               agregado, a.pasos * a.batch * a.acum, a.batch, a.prop_real, a.semilla)

    processor = WhisperProcessor.from_pretrained(a.modelo, revision=rev, language="spanish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.modelo, revision=rev)
    model.generation_config.language, model.generation_config.task = "spanish", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.config.apply_spec_augment, model.config.mask_time_prob, model.config.mask_feature_prob = True, 0.05, 0.05

    run = Path(a.salida).expanduser() / a.nombre
    run.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(run / "checkpoints"), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.acum,
        learning_rate=a.lr, warmup_steps=min(200, a.pasos // 10), max_steps=a.pasos, lr_scheduler_type="linear",
        gradient_checkpointing=True, bf16=True, eval_strategy="steps", eval_steps=a.eval_cada, save_steps=a.eval_cada,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        per_device_eval_batch_size=32, predict_with_generate=True, generation_max_length=128, logging_steps=25,
        report_to="none", seed=a.semilla, data_seed=a.semilla, dataloader_num_workers=4)
    trainer = (C.en_orden(Seq2SeqTrainer) if a.agregado else Seq2SeqTrainer)(
        model=model, args=args, train_dataset=C.ClipsASR(train, datos, processor),
        eval_dataset=C.ClipsASR(dev, datos, processor), data_collator=C.Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=metricas_cer(processor), processing_class=processor.feature_extractor)
    t0 = time.time()
    ultimo = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
    trainer.train(resume_from_checkpoint=ultimo)
    horas = (time.time() - t0) / 3600
    trainer.save_model(str(run / "modelo_final"))
    processor.save_pretrained(str(run / "modelo_final"))
    pd.DataFrame(trainer.state.log_history).to_csv(run / "historial.csv", index=False)

    # --- transcribir el conjunto de evaluación fijo (mismo decodificado que A0)
    model = trainer.model.to(torch.float16).eval()
    ev = Path(a.eval).expanduser()
    audios = pd.read_csv(ev / "audios.csv")
    pred = []
    with torch.no_grad():
        for i in range(0, len(audios), 16):
            t = audios.iloc[i:i + 16]
            x = processor.feature_extractor([sf.read(ev / f, dtype="float32")[0] for f in t.file_name],
                                            sampling_rate=16000, return_tensors="pt").input_features
            ids = model.generate(x.to(model.device, torch.float16), language="spanish", task="transcribe",
                                 max_new_tokens=128, num_beams=1)
            pred += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    salida = Path(a.salida).expanduser()
    pd.DataFrame({"id": audios.id, "prediccion_cruda": [p.strip() for p in pred]}).to_csv(salida / f"{a.nombre}.csv", index=False)
    (salida / f"{a.nombre}.json").write_text(json.dumps(dict(
        modelo=a.modelo, revision=rev, semilla=a.semilla, datos=str(datos), train_clips=len(real_train),
        train_horas=round(real_train.duration_s.sum() / 3600, 3),
        agregado=a.agregado, frac_agregado=a.frac_agregado, prop_real=a.prop_real if a.agregado else 1.0,
        agregado_clips=len(agregado) if a.agregado else 0,
        agregado_horas=round(agregado.duration_s.sum() / 3600, 3) if a.agregado else 0, pasos=a.pasos, lote_efectivo=a.batch * a.acum, lr=a.lr,
        mejor_checkpoint=Path(trainer.state.best_model_checkpoint).name, cer_dev_mejor=trainer.state.best_metric,
        horas_entrenamiento=round(horas, 2), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        transformers=transformers.__version__, python=platform.python_version()), indent=2))
    shutil.rmtree(run / "checkpoints", ignore_errors=True)       # el modelo final ya está guardado
    print(f"{a.nombre}: listo en {horas:.2f} h | mejor {trainer.state.best_model_checkpoint} CER dev {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
