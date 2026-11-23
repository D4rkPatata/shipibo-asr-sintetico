# Guía de ejecución

Pasos 0 a 6: etapa preliminar (corpus v2.0, en local y Colab). Paso 7: etapa final en el servidor.

Orden exacto para reproducir el corpus v2.0 desde cero. `TESIS/` es la carpeta de Drive del proyecto
(en la corrida original, `Mi unidad/TESISOTA/TESIS/`); `SHIPIBO_DATOS` es la carpeta local de datos
(por defecto `D:\tesis\Code`).

## 0. Preparar

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
set SHIPIBO_DATOS=D:\tesis\Code          # PowerShell: $env:SHIPIBO_DATOS = "D:\tesis\Code"
```

En Drive, `TESIS/modelo_tts/` debe tener `SKF2` (Tacotron2), `g_00000000` (HiFi-GAN) y `shp_apto_tts.txt`.

## 1. Corpus (local)

| paso | comando | revisa |
|---|---|---|
| 1.1 | `python 01_corpus/build_dataset_v1.py` | `dataset_v1.0/stats.md`, `excluidos.csv` |
| 1.2 | `python 01_corpus/recoleccion/extract.py <carpeta con links.txt>` | un WAV por enlace |
| 1.3 | `python 01_corpus/recoleccion/cut.py <carpeta> -o <salida>` | `manifest.csv` de cada audio |
| 1.4 | revisión manual: columna `revision` de cada `seguimiento.xlsx` (ok / eliminar / edicion) | — |
| 1.5 | `python 01_corpus/build_no_pareado_v1.py` | `no_pareado_v1/stats.md` |

Subir a Drive `TESIS/asr/`: `dataset_v1.0.zip` y `no_pareado_v1.zip`.

## 2. ASR semilla (Colab, GPU)

1. `04_asr/train_asr.ipynb` → crea `TESIS/asr/split_v1.csv` y entrena. Si se corta, ejecutar todo de nuevo: reanuda.
2. `04_asr/cerrar_asr_semilla.ipynb` → `whisper_small_base/modelo_final/`, `tabla_resultados.csv`, `curvas.png`.

Bajar `TESIS/asr/split_v1.csv` a `SHIPIBO_DATOS` (lo usa el aumento).

## 3. Síntesis (Colab)

1. `02_tts/generation.ipynb` (GPU). Primera corrida con `N_MAX = 50` para escuchar; luego el valor final
   (la tesis usa 11,779 = 50 % de las 23,558 frases). Mismo `SALIDA` y `SEED` en todas las sesiones.
2. `02_tts/filtrar_sintetico.ipynb` (CPU) → `audio_generado/metadata_tts.csv` y `filtrado/resumen_filtrado.csv`.

## 4. Aumento (local)

```bash
python 03_aug/build_aug_v1.py --split %SHIPIBO_DATOS%\split_v1.csv
```

Subir `aug_v1.zip` a `TESIS/asr/`. La primera vez descarga los ruidos de OpenSLR 28 a `_cache/` (solo esa carpeta del zip).

## 5. Pseudoetiquetado (Colab, GPU)

`05_pseudo/pseudoetiquetar.ipynb` → `TESIS/asr/pseudo_v1/`. Revisar la curva `curva_confianza.png` y ajustar
`TOP_PCT` si conviene (no se retranscribe).

## 6. Consolidar (Colab, CPU)

`06_consolidar/consolidar_v2.ipynb` → `TESIS/asr/dataset_v2.0.zip`, `.sha256` y `stats_v2.0.md`.
Si un chequeo falla, el notebook se detiene sin escribir nada.

## Verificar

```powershell
Get-FileHash $env:SHIPIBO_DATOS\dataset_v1.0.zip -Algorithm SHA256
```

Comparar con [configs/huellas_sha256.txt](../configs/huellas_sha256.txt). La de `dataset_v2.0.zip` está en `TESIS/asr/dataset_v2.0.zip.sha256`.

## 7. Etapa final (servidor Linux con GPU)

Entorno: Python 3.12 en `~/venvs/asr` (ver [04_asr/fase_A/README.md](../04_asr/fase_A/README.md)). Copiar al servidor
`dataset_v1.1`, `split_v1.1.csv`, `no_pareado_v1`, `aug_v1`, el texto `shp_apto_tts.txt` y los modelos de Tacotron 2
y HiFi-GAN. Cada cola es reanudable y se lanza dentro de `tmux` para que siga si se corta la conexión.

| paso | comando | revisa |
|---|---|---|
| 7.1 | `python 02_tts/servidor/generar_tacotron.py …` y `python 02_tts/servidor/unir_tacotron.py …` | 23,558 frases de Tacotron 2, sin repetidas |
| 7.2 | `python 02_tts/servidor/generar_mms.py --texto shp_apto_tts.txt --salida ~/tesis/tts/mms` | 23,558 frases de MMS-TTS |
| 7.3 | `python 02_tts/texto_americasnlp.py …` y `generar_mms.py` sobre ese texto | `americasnlp_detalle.csv` (motivo de cada línea descartada) |
| 7.4 | `bash 02_tts/servidor/cola_filtrado.sh` | resumen del filtrado por etapa y por TTS |
| 7.5 | `python 04_asr/fase_A/preparar_tests.py …` | 881 clips en `evaluacion_v1/` |
| 7.6 | colas de `04_asr/fase_A/` en el orden de su README (A → C → C6 + LM → C0 → D → E) | `tabla_C.txt`, `tabla_LM.txt`, `tabla_D.txt`, `tabla_niveles.txt` |
| 7.7 | copiar el adaptador y el LM del modelo final a `07_demo/modelo/` y correr `07_demo/verificar.py` | CER y WER iguales a la evaluación |

Los argumentos de cada script están en su `--help`.
