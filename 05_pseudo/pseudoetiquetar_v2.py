"""Fase D (D25): pseudoetiqueta el audio no pareado con el mejor modelo según dev (MMS-1b-all afinado + LM) y arma
la validación cruzada por video.

Maestro: la configuración MMS + LM de menor CER en dev (A3 s2024 + LM del texto del TTS, CER dev 3.22).
Se transcriben los 740 clips: los 481 del pool y, solo para estimar la calidad de las pseudoetiquetas, los 259 del
control humano (que nunca entran a entrenamiento, D07).

Filtros (los 4 de pseudoetiquetar.ipynb, más el acuerdo de dos jueces de la Fase B):
  1 vacio         sin texto
  2 repeticiones  compression ratio > 2.4, una palabra 4+ veces seguidas o una frase de 2–4 palabras 3+ veces
  3 velocidad     caracteres/s fuera de p0.5–p99.5 del audio pareado real (dataset_v1.1)
  4 jueces        CER sin tildes entre el maestro y Whisper medium afinado (A2) > 40 %
  5 confianza     fuera del 70 % con mayor confianza (log-prob medio del argmax por frame), entre los que pasan 1–4

Validación cruzada: los 32 videos se reparten en dos pliegues equilibrados por fuente (minutos del pool). El modelo
entrenado con las pseudoetiquetas de un pliegue se evalúa en el control del otro: ningún clip del control lo evalúa
un modelo que oyó su video.

Salida (<salida>/): todo.csv (740 clips con predicción, jueces, filtros y pliegue), pliegues.csv (video -> pliegue),
carpetas pliegue1/, pliegue2/, todo/ con metadata.csv + enlace wavs/ (formato de --agregado), resumen.json.

Uso:  python pseudoetiquetar_v2.py --no_pareado ~/tesis/datos/no_pareado_v1 --maestro ~/tesis/fase_A/A3/mms-1b-all_real_s2024 \
          --lm ~/tesis/lm/lm_tts.arpa --alpha 0.25 --beta 1.0 --unk -5 \
          --juez2 ~/tesis/fase_A/A2/whisper-medium_real_s1234/modelo_final --datos ~/tesis/datos/dataset_v1.1 \
          --salida ~/tesis/datos/pseudo_v2
"""
import argparse
import json
import re
import sys
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path.home() / "tesis" / "fase_A" / "codigo"))
import asr_core as C                                     # noqa: E402
from decodificar_lm import etiquetas, logits_de          # noqa: E402

P_BAJO, P_ALTO, TOP_PCT, COMP_MAX, UMBRAL_JUECES = 0.5, 99.5, 70, 2.4, 40


def sin_tildes(t):
    t = unicodedata.normalize("NFD", t)
    return unicodedata.normalize("NFC", "".join(c for c in t if c not in "́̀"))


def repetitivo(t):
    if not t:
        return False
    if len(t) > 20 and len(t.encode()) / len(zlib.compress(t.encode())) > COMP_MAX:
        return True
    p = t.split()
    if any(len(set(p[i:i + 4])) == 1 for i in range(len(p) - 3)):
        return True
    for n in (2, 3, 4):
        for i in range(len(p) - 3 * n + 1):
            if p[i:i + n] == p[i + n:i + 2 * n] == p[i + 2 * n:i + 3 * n]:
                return True
    return False


@torch.no_grad()
def whisper(carpeta, rutas):
    """Mismo decodificado que A2 y que el juez de la Fase B: greedy, token de español, fp16."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(carpeta)
    model = WhisperForConditionalGeneration.from_pretrained(carpeta, torch_dtype=torch.float16).to("cuda").eval()
    out = []
    for i in range(0, len(rutas), 16):
        x = proc.feature_extractor([sf.read(r, dtype="float32")[0] for r in rutas[i:i + 16]], sampling_rate=16000,
                                   return_tensors="pt").input_features
        ids = model.generate(x.to("cuda", torch.float16), language="spanish", task="transcribe", max_new_tokens=128,
                             num_beams=1)
        out += proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return out


def pliegues(pool):
    """Videos -> pliegue 1/2, equilibrando por fuente los minutos del pool (determinista)."""
    asign = {}
    for fuente, g in pool.groupby("source"):
        minutos = g.groupby("subsource").duration_s.sum().sort_values(ascending=False)
        carga = {1: 0.0, 2: 0.0}
        for video, m in minutos.items():
            k = 1 if carga[1] <= carga[2] else 2
            asign[video], carga[k] = k, carga[k] + m
    return asign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_pareado", required=True)
    ap.add_argument("--maestro", required=True, help="carpeta de la corrida MMS (con modelo_final/)")
    ap.add_argument("--lm", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--unk", type=float, required=True)
    ap.add_argument("--juez2", required=True)
    ap.add_argument("--datos", required=True, help="dataset_v1.1 (velocidad del habla real)")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    from pyctcdecode import build_ctcdecoder

    npd, sal = Path(a.no_pareado).expanduser(), Path(a.salida).expanduser()
    sal.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(npd / "metadata.csv", keep_default_na=False)
    ctl = pd.read_csv(npd / "metadata_control.csv", keep_default_na=False).set_index("id")
    meta["control"] = meta.id.isin(ctl.index)
    meta["ref_control"] = meta.id.map(ctl.text_norm).fillna("")

    # --- maestro: logits, LM y confianza
    proc, lps = logits_de(Path(a.maestro).expanduser(), npd, meta)
    dec = build_ctcdecoder(etiquetas(proc), kenlm_model_path=str(Path(a.lm).expanduser()), alpha=a.alpha, beta=a.beta,
                           unk_score_offset=a.unk)
    meta["pred"] = [C.normalizar(dec.decode(lp, beam_width=64)) for lp in lps]
    meta["confianza"] = [float(lp.max(-1).mean()) for lp in lps]
    torch.cuda.empty_cache()
    meta["pred_whisper"] = [C.normalizar(t) for t in whisper(a.juez2, [str(npd / f) for f in meta.file_name])]

    import jiwer
    cer = lambda r, h: 100 * jiwer.cer(sin_tildes(r), sin_tildes(h)) if r else np.nan
    meta["cer_jueces"] = [cer(p, w) if p else np.nan for p, w in zip(meta.pred, meta.pred_whisper)]

    # --- filtros
    real = pd.read_csv(Path(a.datos).expanduser() / "metadata.csv")
    cps_real = real.text_norm.str.len() / real.duration_s
    lo, hi = np.percentile(cps_real, [P_BAJO, P_ALTO])
    meta["cps"] = meta.pred.str.len() / meta.duration_s.astype(float)
    meta["filtro"] = "ok"
    meta.loc[meta.pred == "", "filtro"] = "1 vacio"
    meta.loc[(meta.filtro == "ok") & meta.pred.map(repetitivo), "filtro"] = "2 repeticiones"
    meta.loc[(meta.filtro == "ok") & ~meta.cps.between(lo, hi), "filtro"] = "3 velocidad"
    meta.loc[(meta.filtro == "ok") & (meta.cer_jueces > UMBRAL_JUECES), "filtro"] = "4 jueces"
    # el corte de confianza se fija solo con el pool (el control no decide nada)
    pasa = (meta.filtro == "ok") & ~meta.control
    corte = np.percentile(meta.loc[pasa, "confianza"], 100 - TOP_PCT)
    meta.loc[(meta.filtro == "ok") & (meta.confianza < corte), "filtro"] = "5 confianza"

    # --- pliegues por video
    pool = meta[~meta.control]
    asign = pliegues(pool)
    meta["pliegue"] = meta.subsource.map(asign).fillna(0).astype(int)   # 0: video sin clips en el pool
    pd.DataFrame(sorted(asign.items()), columns=["video", "pliegue"]).to_csv(sal / "pliegues.csv", index=False)
    meta.to_csv(sal / "todo.csv", index=False)

    acept = meta[(meta.filtro == "ok") & ~meta.control]
    for nombre, sub in [("todo", acept), ("pliegue1", acept[acept.pliegue == 1]), ("pliegue2", acept[acept.pliegue == 2])]:
        d = sal / nombre
        d.mkdir(exist_ok=True)
        if not (d / "wavs").exists():
            (d / "wavs").symlink_to(npd / "wavs")
        sub.assign(transcription=sub.pred, text_norm=sub.pred)[
            ["id", "file_name", "transcription", "text_norm", "source", "subsource", "speaker_group", "duration_s",
             "confianza", "cer_jueces", "pliegue"]].to_csv(d / "metadata.csv", index=False)

    # --- calidad esperada: el control pasa por los mismos filtros (umbral de confianza ya fijado con el pool)
    c = meta[meta.control]
    ca = c[c.filtro == "ok"]
    resumen = dict(
        maestro=str(a.maestro), lm=str(a.lm), alpha=a.alpha, beta=a.beta, unk=a.unk, cps_rango=[round(lo, 2), round(hi, 2)],
        corte_confianza=round(float(corte), 4),
        pool_por_filtro=meta[~meta.control].filtro.value_counts().to_dict(),
        aceptados=len(acept), aceptados_min=round(acept.duration_s.astype(float).sum() / 60, 1),
        por_pliegue={k: dict(clips=int((acept.pliegue == k).sum()),
                             min=round(acept[acept.pliegue == k].duration_s.astype(float).sum() / 60, 1)) for k in (1, 2)},
        control_cer_todos=round(100 * jiwer.cer(list(c.ref_control), list(c.pred)), 2),
        control_cer_aceptados=round(100 * jiwer.cer(list(ca.ref_control), list(ca.pred)), 2) if len(ca) else None,
        control_wer_aceptados=round(100 * jiwer.wer(list(ca.ref_control), list(ca.pred)), 2) if len(ca) else None,
        control_pct_aceptado=round(100 * len(ca) / len(c), 1),
        control_por_pliegue={k: int((c.pliegue == k).sum()) for k in (0, 1, 2)})
    (sal / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
