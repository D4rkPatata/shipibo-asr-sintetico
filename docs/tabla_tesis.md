# Tabla para la tesis: script → entrada → salida → parámetros clave

Lista para copiar en la sección de implementación. El diagrama está en el [README](../README.md#flujo).

| Etapa | Script | Entrada | Salida | Parámetros clave |
|---|---|---|---|---|
| Corpus pareado | `build_dataset_v1.py` | SK001, clips de YouTube, workshop | `dataset_v1.0` (3,327 clips, 4.64 h) | 16 kHz mono PCM16; workshop con verificación auditiva |
| Corpus no pareado | `extract.py`, `cut.py`, `build_no_pareado_v1.py` | 19 grabaciones públicas | `no_pareado_v1` (740 clips, 56.4 min) | Silero VAD; clips de 2–6 s; revisión manual |
| Partición | `train_asr.ipynb` | `dataset_v1.0`, texto del TTS | `split_v1.csv` | semilla 1234; dev 5 %, test 10 % de SK001; test_otros por grupos de hablante |
| Síntesis | `generation.ipynb` | Tacotron2 + HiFi-GAN, 22,240 oraciones | audio sintético + manifiesto | semilla 1234 por frase; ≤ 20 palabras; QC de atención |
| Filtrado del sintético | `filtrar_sintetico.ipynb` | audio sintético | `metadata_tts.csv` | voz ≥ 50 %; seg/carácter en p1–p99 de SK001; sin saturación |
| Aumento | `build_aug_v1.py` | train de `dataset_v1.0` | `aug_v1` (8,118 clips, 11.5 h) | velocidad ×0.9 / ×1.1; ruido MUSAN a SNR 10–20 dB; semilla 1234 |
| ASR semilla | `train_asr.ipynb`, `cerrar_asr_semilla.ipynb` | `dataset_v1.0` (train) | `modelo_final` | Whisper small; LR 1e-5; batch 32; SpecAugment; mejor WER en dev |
| Pseudoetiquetado | `pseudoetiquetar.ipynb` | `no_pareado_v1`, ASR semilla | `metadata_pseudo.csv` | top 70 % por log-prob; anti-bucles; cps p0.5–p99.5 |
| Consolidación | `consolidar_v2.ipynb` | todo lo anterior | `dataset_v2.0` + tabla de horas | 5 chequeos de fuga; sha256 del zip |
