# Datos sintéticos para ASR en Shipibo-Konibo

Código de la tesis *Generación de datos sintéticos para el entrenamiento de sistemas ASR en Shipibo-Konibo*
(Daniel Malca López, PUCP). Cubre el flujo completo: armado del corpus, síntesis de voz (TTS), aumento de audio,
ASR semilla, pseudoetiquetado y consolidación del corpus ampliado **v2.0**.

Los datos no viven en el repositorio (son GB de audio): los scripts locales leen y escriben en la carpeta
`SHIPIBO_DATOS` y los notebooks de Colab en Google Drive. Cada paso deja un `config.json` con sus parámetros,
versiones y huellas sha256, que funciona como registro automático del experimento.

## Flujo

```mermaid
flowchart TD
    subgraph C["01 · Corpus"]
        A1[SK001 · Clipped · Workshop] --> B1[build_dataset_v1.py]
        B1 --> D1[(dataset_v1.0.zip<br/>3,327 clips · 4.64 h)]
        A2[SoundCloud · YouTube · GRN] --> X1[recoleccion/extract.py] --> X2[recoleccion/cut.py<br/>Silero VAD] --> X3[revisión manual<br/>seguimiento.xlsx]
        X3 --> B2[build_no_pareado_v1.py]
        T1[CONTROL_SET<br/>transcripción humana] --> B2
        B2 --> D2[(no_pareado_v1.zip<br/>740 clips · 56.4 min)]
    end
    D1 --> S1[04 · train_asr.ipynb<br/>split_v1.csv + Whisper small]
    S1 --> S2[04 · cerrar_asr_semilla.ipynb]
    S2 --> M1[(ASR semilla<br/>modelo_final)]
    TX[shp_apto_tts.txt<br/>22,240 oraciones] --> G1[02 · generation.ipynb<br/>Tacotron2 + HiFi-GAN]
    G1 --> G2[02 · filtrar_sintetico.ipynb]
    G2 --> D3[(metadata_tts.csv)]
    D1 --> A3[03 · build_aug_v1.py<br/>×0.9 · ×1.1 · ruido 10–20 dB]
    S1 -. split_v1.csv .-> A3
    A3 --> D4[(aug_v1.zip<br/>8,118 clips · 11.5 h)]
    M1 --> P1[05 · pseudoetiquetar.ipynb]
    D2 --> P1
    P1 --> D5[(metadata_pseudo.csv)]
    D1 & D2 & D3 & D4 & D5 --> K1[06 · consolidar_v2.ipynb<br/>5 chequeos de fuga]
    K1 --> D6[(dataset_v2.0.zip<br/>+ tabla de horas por tipo)]
```

## Pasos

| # | script | dónde corre | entrada | salida | parámetros clave |
|---|---|---|---|---|---|
| 01 | [`build_dataset_v1.py`](01_corpus/build_dataset_v1.py) | local | SK001, Clipped, DatasetWorkshop | `dataset_v1.0/` + `.zip` | 16 kHz mono; solo workshop con check auditivo = SI |
| 01 | [`recoleccion/extract.py`](01_corpus/recoleccion/extract.py) | local | `links.txt` | un WAV por enlace | yt-dlp |
| 01 | [`recoleccion/cut.py`](01_corpus/recoleccion/cut.py) | local | audio largo | clips de 2–6 s + `manifest.csv` | Silero VAD, umbral 0.5, pausa de unión 0.40 s |
| 01 | [`build_no_pareado_v1.py`](01_corpus/build_no_pareado_v1.py) | local | `seguimiento.xlsx` ×3, `CONTROL_SET/transcripcion.xlsx` | `no_pareado_v1/` + `.zip` | excluye *eliminar* y *edicion* |
| 02 | [`generation.ipynb`](02_tts/generation.ipynb) | Colab GPU | Tacotron2 SKF2, HiFi-GAN, `shp_apto_tts.txt` | `audio_generado/` + `manifest.csv` | SEED 1234, ≤20 palabras/frase, `N_MAX` |
| 02 | [`filtrar_sintetico.ipynb`](02_tts/filtrar_sintetico.ipynb) | Colab CPU | `audio_generado/`, `dataset_v1.0.zip`, `split_v1.csv` | `metadata_tts.csv`, `resumen_filtrado.csv` | voz ≥ 50 %, seg/carácter en p1–p99 de SK001 |
| 03 | [`build_aug_v1.py`](03_aug/build_aug_v1.py) | local | `dataset_v1.0/`, `split_v1.csv` | `aug_v1/` + `.zip` | SEED 1234, ×0.9/×1.1, SNR U(10, 20) dB |
| 04 | [`train_asr.ipynb`](04_asr/train_asr.ipynb) | Colab GPU | `dataset_v1.0.zip`, `shp_apto_tts.txt` | `split_v1.csv`, checkpoints | whisper-small, LR 1e-5, batch 32, SpecAugment |
| 04 | [`cerrar_asr_semilla.ipynb`](04_asr/cerrar_asr_semilla.ipynb) | Colab GPU | checkpoints | `modelo_final/`, `tabla_resultados.csv` | mejor checkpoint por WER en dev |
| 05 | [`pseudoetiquetar.ipynb`](05_pseudo/pseudoetiquetar.ipynb) | Colab GPU | `no_pareado_v1.zip`, `modelo_final/` | `metadata_pseudo.csv`, `eval_control.json` | top 70 % por log-prob, cps p0.5–p99.5 |
| 06 | [`consolidar_v2.ipynb`](06_consolidar/consolidar_v2.ipynb) | Colab CPU | todo lo anterior | `dataset_v2.0.zip`, `stats_v2.0.md` | 5 chequeos; falla = no escribe |

Detalle de cada paso en el `README.md` de su carpeta. Orden de ejecución y rutas exactas: [docs/guia_ejecucion.md](docs/guia_ejecucion.md).

## Estructura

```
01_corpus/        corpus pareado v1.0 y corpus no pareado v1 (+ recolección y corte con VAD)
02_tts/           generación de voz sintética y su filtrado
03_aug/           aumento de audio offline (solo train)
04_asr/           ASR semilla: entrenamiento y cierre
05_pseudo/        pseudoetiquetado del no pareado, evaluado contra el control humano
06_consolidar/    corpus ampliado v2.0 con chequeos de fuga
configs/          huellas sha256 y parámetros de referencia
docs/             guía de ejecución y tabla para la tesis
```

Cada notebook es autocontenido: su celda 4 escribe en Colab los módulos que usa (`asr_core.py`, `tts_core.py`,
`tts_filtro.py`, `pseudo_core.py`) con `%%writefile`.

## Requisitos

- **Local** (Python 3.12): `pip install -r requirements.txt` y `ffmpeg` en el PATH (solo para `extract.py`).
- **Colab**: cada notebook instala sus versiones fijas en su celda 3:
  `transformers==5.17.0`, `accelerate==1.15.0`, `jiwer==4.0.0`, `silero-vad`, `gdown`.
- **Modelo ASR**: `openai/whisper-small`, revisión `973afd24965f72e36ca33b3055d56a652f456b4d`.
- **TTS**: `rmcpantoja/tacotron2@0b33108` y `justinjohn0306/hifi-gan@11a1788` (commits fijados en `generation.ipynb`).

## Datos y rutas

| dónde | qué |
|---|---|
| `SHIPIBO_DATOS` (local, por defecto `D:\tesis\Code`) | `SK001/`, `Clipped/`, `DatasetWorkshop-…/`, carpetas del no pareado, `CONTROL_SET/`; aquí se escriben `dataset_v1.0/`, `no_pareado_v1/`, `aug_v1/` y sus zips |
| Drive `TESIS/modelo_tts/` | `SKF2` (Tacotron2), `g_00000000` (HiFi-GAN), `shp_apto_tts.txt` |
| Drive `TESIS/asr/` | zips del corpus, `split_v1.csv`, `whisper_small_base/`, `pseudo_v1/`, `dataset_v2.0.zip` |
| Drive `TESIS/audio_generado/` | audio sintético, `manifest.csv`, `metadata_tts.csv`, `filtrado/` |

Los notebooks traen por defecto `MyDrive/TESIS/…`; si la carpeta está en otro lugar (p. ej. `MyDrive/TESISOTA/TESIS`)
se cambia en la celda 1.

## Reproducibilidad

- **Semillas fijas** (1234) en partición, generación TTS (semilla por frase derivada de su id), aumento
  (semilla por clip derivada de id + tipo) y entrenamiento. Regenerar da el mismo audio, verificado byte a byte.
- **Partición fija** `split_v1.csv`: la crea `train_asr` una vez y todos los pasos la reutilizan;
  ninguna frase de dev/test aparece en train ni en el texto del TTS.
- **Huellas sha256** de dataset, texto, modelos y partición en cada `config.json` y en
  [configs/huellas_sha256.txt](configs/huellas_sha256.txt). Para verificar un archivo:

  ```powershell
  Get-FileHash dataset_v1.0.zip -Algorithm SHA256      # o en Colab/Linux: sha256sum dataset_v1.0.zip
  ```
- **Reanudable**: generación, filtrado, pseudoetiquetado y entrenamiento continúan donde quedaron si Colab se desconecta.

## Licencias de los datos

Detalle por fuente en [01_corpus/recoleccion/fuentes.csv](01_corpus/recoleccion/fuentes.csv). En resumen: SK001 cedido por
Menéndez & Gómez (2025); workshop, grabación propia; YouTube y SoundCloud públicos con derechos reservados
(uso académico, sin redistribuir el audio); Palabras de Vida (Global Recordings Network); texto del TTS derivado de
Bustamante, Oncevay & Zariquiey (2020); ruidos del aumento de OpenSLR 28 / MUSAN (Apache 2.0).
