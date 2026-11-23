# 04 · fase_A: entrenamiento, evaluación y comparación de escenarios (Objetivo 3)

Scripts de Python que corren en un servidor Linux con GPU (en la tesis, dos NVIDIA RTX A6000 de 48 GB). Los
`cola_*.sh` encadenan entrenamiento, decodificación y tablas de cada fase, y esperan a que haya una GPU libre
(`esperar_gpu.sh`). Las decisiones que citan los comentarios (D10, D24…) están en
[docs/decisiones.md](../../docs/decisiones.md).

Rutas del servidor que usan las colas (se cambian al inicio de cada `.sh`):

| ruta | contenido |
|---|---|
| `~/tesis/datos/` | `dataset_v1.1/`, `split_v1.1.csv`, `no_pareado_v1/`, `sintetico_v1/` (los dos TTS filtrados), `aug_v1/`, `pseudo_v2/` |
| `~/tesis/evaluacion_v1/` | conjunto de evaluación fijo (881 clips: dev, test_sk001, test_otros, test_control) |
| `~/tesis/fase_A/`, `fase_C/`, `fase_D/`, `fase_LM/`, `fase_E/` | una carpeta por corrida: configuración, historial de CER en dev, `modelo_final/` y predicciones |
| `~/tesis/lm/` | textos y modelos de lenguaje (`lm_tts.arpa`, `lm_tts_anlp.arpa`) |

## Orden de ejecución

| fase | qué hace | cómo se lanza |
|---|---|---|
| 0 | arma el conjunto de evaluación fijo | `python preparar_tests.py` |
| A0 | modelos sin ajuste fino (Whisper, MMS-1b-all, Omnilingual 1B) | `python a0_hf.py`, `python a0_omnilingual.py` |
| A | Whisper small (A1), Whisper medium (A2) y MMS-1b-all (A3) solo con audio real | `bash cola_A.sh`, `bash cola_A2A3.sh` |
| C | C1–C5: real + cada tipo de dato generado (50 % real por lote) | `bash cola_C.sh 0` y `bash cola_C.sh 1` (un trabajador por GPU) |
| C6 + LM | real + MMS-TTS sobre AmericasNLP; decodificación con LM de todas las corridas MMS | `bash cola_C6_LM.sh` |
| C0 | solo audio sintético (sin audio real) | `bash cola_C0.sh` |
| D | pseudoetiquetas (versión final) con validación cruzada por video | `bash cola_D.sh` |
| E | elección del modelo final solo con dev, análisis de errores y métricas relajadas | `python elegir_final.py`, `python analizar_errores.py`, `python metricas_relajadas.py` |

Los argumentos exactos de cada script están en su `--help` y en las colas, que muestran las llamadas usadas en la tesis.
Cada `cola_*.sh` deja su log en `~/tesis/<fase>/logs/` y es reanudable: una corrida con `.json` final no se repite.

## Scripts

| script | qué hace |
|---|---|
| `asr_core.py` | núcleo común: normalización de texto, carga de datos y mezcla real + agregado por lote |
| `entrenar_mms.py` | ajuste fino de MMS-1b-all (solo adaptador shp + cabeza CTC, LR 1e-3, 2,000 pasos, lote 32) |
| `entrenar_whisper.py` | ajuste fino de Whisper con el mismo protocolo (LR 1e-5) |
| `evaluar.py` | CER y WER de corpus, con y sin tildes, con IC 95 % por bootstrap |
| `lm_ngram.py` | modelo de lenguaje de 4-gramas Kneser-Ney en formato ARPA |
| `decodificar_lm.py` | búsqueda en haz (64) con LM; pesos elegidos solo con dev |
| `lm_niveles.py` | LM de palabra, subpalabra (SentencePiece) y carácter con un decodificador propio |
| `tabla_C.py`, `tabla_D.py`, `tabla_LM.py`, `tabla_niveles.py`, `resumen_LM.py` | tablas por fase con bootstrap pareado (5,000 remuestreos) |
| `elegir_final.py` | elige la configuración final solo con dev y recién después reporta test |
| `analizar_errores.py` | errores del modelo final: segmentación, confusiones, palabras según su origen |
| `metricas_relajadas.py` | CER normalizado al alfabeto oficial, CER sin espacios y WER tolerante a la segmentación |

## Entorno

Python 3.12 en un entorno virtual (`~/venvs/asr`) con `torch`, `transformers`, `accelerate`, `jiwer`, `pyctcdecode`,
`kenlm` (solo el módulo de Python) y `sentencepiece`. Las versiones exactas quedan registradas en el `config.json`
de cada corrida.
