# 01 · Corpus

## `build_dataset_v1.py` → corpus pareado v1.0

Integra tres fuentes con transcripción humana y las deja en 16 kHz, mono, PCM16:

| fuente | carpeta en `SHIPIBO_DATOS` | criterio |
|---|---|---|
| SK001, Menéndez & Gómez (2025) | `SK001/` | se usa tal cual (ya curado) |
| YouTube (8 videos) | `Clipped/` | clips cortados y transcritos a mano |
| Workshop / curso NLP | `DatasetWorkshop-…/` | solo filas con *Paso check auditivo* = SI; usa la versión `_FIXED` si existe |

Salida: `dataset_v1.0/` (`wavs/`, `metadata.csv`, `metadata_hf.csv`, `excluidos.csv`, `stats.json`, `stats.md`).
Resultado de la tesis: 3,327 clips, 4.64 h, 18 grupos de hablante.

## `build_dataset_v1_1.py` → corpus pareado v1.1

v1.0 **congelada** (se copia byte a byte) + los clips del workshop aprobados después (`Transcriptions_final.xlsx`).
Cada clip nuevo va al split de su grupo de hablante en `split_v1.1.csv` (= `split_v1.csv` + filas nuevas); se saca
si su texto genera fuga (evaluación contenida en el TTS, o train con texto de evaluación).

Resultado: 3,331 clips, 4.641 h. Agregados 4 clips (21.1 s): `WS-g07-oracion_001_shipibo` (train) y los 3 trozos en que
se partió `grupo08_oracion21` (test_otros). Todo lo construido sobre v1.0 (aumento, ASR semilla, v2.0) sigue siendo válido.

## `recoleccion/` → audio no pareado

1. `extract.py <carpeta>`: descarga cada enlace de `<carpeta>/links.txt` (SoundCloud o YouTube) a su propia carpeta, en WAV.
2. `cut.py <carpeta> -o <salida>`: corta en clips de voz de 2–6 s con Silero VAD. Une regiones separadas por pausas
   cortas, parte las largas en su silencio interno más ancho, detecta contrafase L/R y deja un `manifest.csv` con
   tiempos, % de voz y estado para la revisión manual.
3. Revisión manual a oído en `seguimiento.xlsx` (columna `revision`: ok / eliminar / edicion).

`fuentes.csv`: todas las fuentes usadas, con enlace, duración, licencia y forma de acceso.

## `build_no_pareado_v1.py` → corpus no pareado v1

Junta los clips revisados de las tres carpetas (ministerio/PNUD, Cuna Más, Palabras de Vida), excluye *eliminar* y
*edicion*, y adjunta la transcripción humana del `CONTROL_SET` a los clips que la tienen.

Salida: `no_pareado_v1/` (`metadata.csv`, `metadata_control.csv`, `stats.md`) y `no_pareado_v1.zip`.
Resultado: 740 clips, 56.4 min; 259 con transcripción humana (control, 20.1 min) y 481 para pseudoetiquetar.
