# Parámetros de referencia y dónde queda registrado cada uno

Cada notebook guarda sus parámetros reales en un `config.json` al ejecutarse; estos son los valores de la corrida de la tesis.

| paso | registro automático | parámetros |
|---|---|---|
| generación TTS | `TESIS/audio_generado/config.json` | SEED 1234 · MAX_PALABRAS 20 · MIN_PALABRAS 2 · MAX_DUR_S 20 · STOP_THRESHOLD 0.5 · FUERZA_DENOISER 35 · OUTPUT_SR 22050 · FOCO_MIN 0.30 · COBERTURA_MIN 0.85 · RATIO_DUR 0.4–2.5 · sha256 del texto y de ambos modelos |
| filtrado TTS | `audio_generado/filtrado/config_filtrado.json` | DUR_MIN_S 0.5 · RMS_MIN_DB −50 · MESETA_MAX_PCT 0.1 · VOZ_MIN_PCT 50 · QC Tacotron sí · seg/carácter p1–p99 de SK001 |
| ASR semilla | `TESIS/asr/whisper_small_base/config.json` | whisper-small @973afd2 · token de idioma *spanish* · SEED 1234 · LR 1e-5 · BATCH 16 × ACUM 2 · WARMUP 200 · MAX_STEPS 2000 · EVAL_CADA 100 · PACIENCIA 5 · SpecAugment (0.05 / 0.05) |
| cierre | `whisper_small_base/modelo_final/origen.json` | checkpoint elegido y paso |
| aumento | `aug_v1/stats.json` y columna `semilla` de `metadata_aug.csv` | SEED 1234 · velocidad ×0.9 / ×1.1 · SNR U(10, 20) dB · OpenSLR 28 pointsource_noises |
| pseudoetiquetado | `TESIS/asr/pseudo_v1/config.json` | MODO top · TOP_PCT 70 · COMP_MAX 2.4 · REP 4 (palabra) / 3 (frase) · cps p0.5–p99.5 · greedy · MAX_TOKENS 128 |
| consolidación | `dataset_v2.0.zip` → `stats.json`, `chequeos.csv` | sha256 de cada fuente · INCLUIR_PSEUDO |
