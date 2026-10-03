# 02 · Síntesis de voz (TTS)

## `generation.ipynb` (Colab, GPU)

Tacotron2 (checkpoint `SKF2`) + HiFi-GAN (`g_00000000`) del trabajo de Menéndez y Erasmo, cargados como en su notebook
de inferencia, sobre la versión actual de PyTorch.

- **Texto**: `shp_apto_tts.txt`, 22,240 oraciones (el 95.2 % aparece tal cual en el corpus de Bustamante et al., 2020). Las de más de
  20 palabras se parten en frases de ≤ 20 palabras (primero por puntuación): **23,558 frases** sintetizables.
- **Reproducible**: semilla por frase = SEED + crc32(id); la semilla del denoiser se fija antes de construirlo;
  orden de proceso barajado con semilla, así que `N_MAX = k` siempre toma las mismas k frases.
- **Reanudable**: `manifest.csv` es la lista de lo hecho; escritura atómica (`.part` → wav). Si cambian SEED, el texto
  o los modelos, se niega a mezclar corridas en la misma carpeta.
- **Control de calidad por frase** (solo marca, no borra): no se detuvo, atención difusa, no llegó al final del texto,
  duración anómala respecto a 10.31 caracteres/s de SK001.

## `filtrar_sintetico.ipynb` (Colab, CPU)

Filtros en orden; cada clip se cuenta en el primero que lo descarta:

| # | filtro | descarta si |
|---|---|---|
| 0 | no generado / fuga | sin wav, o su texto contiene una frase de dev/test |
| 1a | vacío | < 0.5 s o RMS < −50 dBFS |
| 1b | saturado | > 0.1 % de muestras en mesetas pegadas al pico |
| 1c | silencio | voz < 50 % (Silero VAD) |
| 1d | QC de atención | el estado de la generación no es `ok` |
| 2 | duración vs texto | segundos de **voz** por carácter fuera de p1–p99 de SK001 |

La ida y vuelta con el ASR (CER contra el texto) queda **pospuesta**: el ASR
semilla (WER 52 % / CER 14 % en voces nuevas) mezclaría sus errores con los del TTS.

Salida: `audio_generado/metadata_tts.csv`, `filtrado/resumen_filtrado.csv`, `detalle_filtrado.csv`, `config_filtrado.json`.