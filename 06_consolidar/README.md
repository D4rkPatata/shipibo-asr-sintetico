# 06 · Corpus ampliado v2.0

`consolidar_v2.ipynb` (Colab, CPU) junta todo en un solo manifiesto y lo empaqueta en `dataset_v2.0.zip`
(audio a 16 kHz mono en `wavs/<tipo>/`).

| tipo | split |
|---|---|
| real (SK001, workshop, YouTube) | train / dev / test_sk001 / test_otros, según `split_v1.csv` |
| real (control, transcripción humana) | test_control |
| tts | train |
| aug | train |
| pseudo (opcional) | train |

**Chequeos** (si uno falla, no escribe nada; probado inyectando fugas a propósito):

1. ninguna frase de dev/test contenida en el texto del TTS;
2. el aumento sale solo de ids de train;
3. el no pareado no comparte fuente (institución o canal) con test_otros;
4. las pseudoetiquetas no incluyen clips del control;
5. ids únicos, archivos existentes, ningún clip sin texto.

Salida en `TESIS/asr/`: `dataset_v2.0.zip` (con `metadata.csv`, `metadata_hf_train.csv`, `chequeos.csv`, `stats.json`),
`dataset_v2.0.zip.sha256` y `stats_v2.0.md` (horas por tipo y partición, para la tesis).
Cada experimento elige sus datos filtrando la columna `tipo`.
