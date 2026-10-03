# 04 · ASR semilla

## `train_asr.ipynb` (Colab, GPU)

Whisper small afinado con el corpus real v1.0. Whisper no trae shipibo: se usa el token de español (misma grafía latina).

**Partición `split_v1.csv`** (se crea una vez y la usan todos los pasos):

| partición | contenido |
|---|---|
| train | resto de SK001 + mitad de los grupos de hablante de YouTube y workshop |
| dev | 5 % de SK001 (elige el checkpoint) |
| test_sk001 | 10 % de SK001: mismo hablante, frases no vistas |
| test_otros | la otra mitad de los grupos: voces no vistas |

Ninguna frase de dev/test está en train ni contenida en el texto del TTS (213 frases de SK001 lo estaban y se excluyeron
de la evaluación). Entrenamiento: LR 1e-5, batch efectivo 32, SpecAugment, hasta 2000 pasos con parada temprana por
WER en dev. Reanudable desde el último checkpoint.

## `cerrar_asr_semilla.ipynb` (Colab, GPU)

Toma el mejor checkpoint según `trainer_state.json` (o el de menor WER entre los que existan), crea `modelo_final/`
y evalúa dev, test_sk001 y test_otros: WER y CER de corpus con IC 95 % (bootstrap sobre clips), con y sin tildes,
junto al zero-shot.

Salida: `tabla_resultados.csv`, `resultados.json`, `pred_*.csv`, `curvas.png`, `modelo_final/origen.json`.