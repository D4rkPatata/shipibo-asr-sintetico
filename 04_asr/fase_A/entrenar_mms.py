"""Ajuste fino de MMS-1b-all (wav2vec 2.0 + CTC) partiendo de su adaptador de shipibo (Fase A3 y escenarios C/D).

Se entrena solo el adaptador shp (capas de adaptador + cabeza CTC, ~2.5 M parámetros), con el resto congelado:
es la receta de Meta para MMS y no destruye lo que el modelo ya sabe (A0: CER 12.5 en voces no vistas sin entrenar).

Protocolo: igual que entrenar_whisper.py (D16) salvo lo propio de la arquitectura:
  - mismos 2,000 pasos, lote efectivo 32, bf16, evaluación en dev cada 100 pasos, checkpoint de menor CER en dev
  - LR 1e-3 (receta del adaptador MMS; 1e-5 es para ajustar Whisper completo), 200 pasos de calentamiento
  - SpecAugment en tiempo (mask_time_prob 0.05)
  - al terminar transcribe evaluacion_v1 con argmax, igual que a0_hf.py

Uso (A3, solo real):
  python entrenar_mms.py --semilla 1234 --datos ~/tesis/datos/dataset_v1.1 --split ~/tesis/datos/split_v1.1.csv \
      --eval ~/tesis/evaluacion_v1 --salida ~/tesis/fase_A/A3 --nombre mms-1b-all_real_s1234
"""
import argparse
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import asr_core as C

REPO, REVISION = "facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"   # la misma de A0


class ClipsCTC(torch.utils.data.Dataset):
    def __init__(self, df, raiz, processor):
        self.filas = list(df[["file_name", "text_norm"]].itertuples(index=False))
        self.raiz, self.p = Path(raiz), processor

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, i):
        archivo, texto = self.filas[i]
        y, sr = sf.read(self.raiz / archivo, dtype="float32")
        assert sr == C.SR, f"{archivo}: {sr} Hz"
        return {"input_values": self.p.feature_extractor(y, sampling_rate=C.SR).input_values[0],
                "labels": self.p.tokenizer(texto).input_ids}       # dígitos (23 en train) -> <unk>


@dataclass
class CollatorCTC:
    processor: object

    def __call__(self, feats):
        batch = self.processor.feature_extractor.pad([{"input_values": f["input_values"]} for f in feats],
                                                     return_attention_mask=True, return_tensors="pt")
        lab = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        batch["labels"] = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        return batch


def metricas_cer(processor):
    import jiwer
    tok = processor.tokenizer

    def f(pred):
        ids = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)   # relleno entre lotes
        lab = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        hyps = [C.normalizar(t) for t in tok.batch_decode(ids)]
        refs = [C.normalizar(t) for t in tok.batch_decode(lab, group_tokens=False)]
        par = [(r, h) for r, h in zip(refs, hyps) if r]
        r, h = [p[0] for p in par], [p[1] for p in par]
        return {"wer": round(100 * jiwer.wer(r, h), 2), "cer": round(100 * jiwer.cer(r, h), 2)}
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", type=int, required=True)
    ap.add_argument("--datos", required=True, help="carpeta con metadata.csv y wavs/")
    ap.add_argument("--split", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--pasos", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--acum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_cada", type=int, default=100)
    ap.add_argument("--agregado", nargs="*", default=[], help="carpetas con metadata.csv (sintético filtrado o aumento)")
    ap.add_argument("--frac_agregado", type=float, default=1.0, help="fracción de frases del agregado (C4)")
    ap.add_argument("--prop_real", type=float, default=0.5, help="proporción de clips reales en cada lote (D10)")
    a = ap.parse_args()

    import transformers
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(a.semilla)
    datos = Path(a.datos).expanduser()
    meta = pd.read_csv(datos / "metadata.csv").merge(pd.read_csv(Path(a.split).expanduser()), on="id")
    train, dev = meta[meta.split == "train"], meta[meta.split == "dev"]
    real_train = train
    if a.agregado:      # Fase C: lista fija y barajada, 50 % real / 50 % agregado en cada micro-lote
        agregado = C.leer_agregado(a.agregado, a.frac_agregado)
        train = C.orden_mezcla(real_train.assign(file_name=[str(datos / f) for f in real_train.file_name]),
                               agregado, a.pasos * a.batch * a.acum, a.batch, a.prop_real, a.semilla)

    processor = AutoProcessor.from_pretrained(REPO, revision=REVISION, target_lang="shp")
    model = Wav2Vec2ForCTC.from_pretrained(REPO, revision=REVISION, target_lang="shp", ignore_mismatched_sizes=True,
                                           ctc_loss_reduction="mean", ctc_zero_infinity=True, mask_time_prob=0.05,
                                           layerdrop=0.0)
    model.load_adapter("shp")
    model.freeze_base_model()
    entrenables = model._get_adapters()                         # adaptadores de cada capa + lm_head
    for p in entrenables.values():
        p.requires_grad = True
    n_entrenables = sum(p.numel() for p in entrenables.values())

    run = Path(a.salida).expanduser() / a.nombre
    run.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(run / "checkpoints"), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.acum,
        learning_rate=a.lr, warmup_steps=min(200, a.pasos // 10), max_steps=a.pasos, lr_scheduler_type="linear",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True,
        eval_strategy="steps", eval_steps=a.eval_cada, save_steps=a.eval_cada, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        per_device_eval_batch_size=16, logging_steps=25, report_to="none", seed=a.semilla, data_seed=a.semilla,
        dataloader_num_workers=4)
    trainer = (C.en_orden(Trainer) if a.agregado else Trainer)(
        model=model, args=args, train_dataset=ClipsCTC(train, datos, processor), eval_dataset=ClipsCTC(dev, datos, processor),
        data_collator=CollatorCTC(processor), compute_metrics=metricas_cer(processor),
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(-1),
        processing_class=processor.feature_extractor)
    t0 = time.time()
    ultimo = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
    inicial = None if ultimo else trainer.evaluate()["eval_cer"]   # sin entrenar: debe coincidir con A0 en dev
    trainer.train(resume_from_checkpoint=ultimo)
    horas = (time.time() - t0) / 3600
    pd.DataFrame(trainer.state.log_history).to_csv(run / "historial.csv", index=False)
    # solo se guarda el adaptador (10 MB): el resto es MMS-1b-all intacto en la revisión fijada
    from safetensors.torch import save_file
    (run / "modelo_final").mkdir(exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in trainer.model._get_adapters().items()},
              str(run / "modelo_final" / "adapter.shp.safetensors"))
    processor.save_pretrained(str(run / "modelo_final"))

    # --- transcribir el conjunto de evaluación fijo (mismo decodificado que A0)
    model = trainer.model.float().eval()
    ev = Path(a.eval).expanduser()
    audios = pd.read_csv(ev / "audios.csv")
    pred = []
    with torch.no_grad():
        for i in range(0, len(audios), 16):
            t = audios.iloc[i:i + 16]
            x = processor([sf.read(ev / f, dtype="float32")[0] for f in t.file_name], sampling_rate=16000,
                          return_tensors="pt", padding=True)
            logits = model(x.input_values.to(model.device), attention_mask=x.attention_mask.to(model.device)).logits
            pred += processor.batch_decode(logits.argmax(-1))
    salida = Path(a.salida).expanduser()
    pd.DataFrame({"id": audios.id, "prediccion_cruda": [p.strip() for p in pred]}).to_csv(salida / f"{a.nombre}.csv", index=False)
    (salida / f"{a.nombre}.json").write_text(json.dumps(dict(
        modelo=REPO, revision=REVISION, adaptador_inicial="shp", parametros_entrenables=n_entrenables,
        semilla=a.semilla, datos=str(datos), train_clips=len(real_train), train_horas=round(real_train.duration_s.sum() / 3600, 3),
        agregado=a.agregado, frac_agregado=a.frac_agregado, prop_real=a.prop_real if a.agregado else 1.0,
        agregado_clips=len(agregado) if a.agregado else 0,
        agregado_horas=round(agregado.duration_s.sum() / 3600, 3) if a.agregado else 0,
        pasos=a.pasos, lote_efectivo=a.batch * a.acum, lr=a.lr,
        cer_dev_inicial=inicial, mejor_checkpoint=Path(trainer.state.best_model_checkpoint).name, cer_dev_mejor=trainer.state.best_metric,
        horas_entrenamiento=round(horas, 2), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        transformers=transformers.__version__, python=platform.python_version()), indent=2))
    shutil.rmtree(run / "checkpoints", ignore_errors=True)
    print(f"{a.nombre}: listo en {horas:.2f} h | mejor {trainer.state.best_model_checkpoint} CER dev {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
