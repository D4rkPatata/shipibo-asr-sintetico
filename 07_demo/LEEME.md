# Demo local del ASR en shipibo-konibo (R8)

Página web local (FastAPI) para **grabar o subir un audio en shipibo-konibo y ver su transcripción** con el modelo
final de la tesis. Muestra dos resultados: con modelo de lenguaje (el del modelo final) y solo con el modelo acústico,
para ver qué aporta el LM. Corre en CPU, sin GPU ni internet (salvo la primera descarga del modelo base).

Modelo por defecto (decisión D26, elegido solo con dev): MMS-1b-all + adaptador shp afinado con audio real (A3,
semilla 2024) + LM de palabras de 4-gramas sobre el texto TTS (alpha 0.25, beta 1.0, unk −5, haz 64).

## 1. Instalar (una vez)

Requisitos: Windows, Linux o macOS; 8 GB de RAM libres; ~6 GB de disco; [uv](https://docs.astral.sh/uv/).

```bat
instalar.bat
```

(o a mano: `uv venv .venv --python 3.12`, luego `torch==2.11.0` desde el índice de CPU de PyTorch y
`uv pip install --python .venv -r requirements.txt`).

Los archivos del modelo van en `modelo/` (ver `modelo/LEEME.md`). No están en el repositorio.

## 2. Usar

```bat
iniciar.bat
```

Se abre http://127.0.0.1:8000. La primera vez descarga MMS-1b-all (~4 GB) y tarda varios minutos; después carga en
~1–2 min. En la página:

1. **Grabar** (pide permiso para el micrófono) o **Subir audio** (WAV, FLAC, OGG o MP3; hasta 60 s).
2. **Transcribir**: en unos 2 s aparece la transcripción con LM y, debajo, la del modelo acústico solo.

Los audios no se guardan: se transcriben en memoria y se descartan.

También hay una API: `POST /transcribir` (campo `audio`) devuelve
`{"con_lm", "sin_lm", "duracion_s", "segundos_modelo", "segundos_lm"}`; `GET /modelo` describe el modelo cargado.

## 3. Cambiar de modelo

Todo lo que define el modelo está en `config.json` y `modelo/`; el código no cambia.

| para usar… | qué hacer |
|---|---|
| otro adaptador MMS (otra semilla o escenario: C4, C6, D…) | copiar su carpeta `modelo_final/` (`adapter.shp.safetensors` + `vocab.json`, `tokenizer_config.json`, `processor_config.json`) a `modelo/` (o a otra carpeta y poner su ruta en `"adaptador"`) |
| otro LM de palabras (p. ej. `lm_tts_anlp.arpa`) | copiar el `.arpa` y poner su ruta en `"lm"` |
| sin LM | `"lm": ""` |
| hiperparámetros del LM | `"alpha"`, `"beta"`, `"unk"`, `"haz"`: usar **los elegidos en dev para esa corrida** (`~/tesis/fase_LM/hiperparametros.csv`), no valores al azar |
| describirlo en la página | `"nombre"` |

Después: reiniciar `iniciar.bat` y correr `verificar.py` (abajo) para comprobar que da lo mismo que la evaluación.

Límites: el código sirve para adaptadores de **MMS-1b-all** (CTC) con LM de **palabras** en ARPA. Los LM de caracteres
o subpalabras de la prueba 2 (`lm_niveles.py`) usan otro decodificador y no están conectados a la demo.

## 4. Verificar (evidencia de funcionamiento)

`verificar.py` transcribe los audios de prueba con el mismo código que usa la página y compara con la evaluación de la
tesis (CER/WER y % de transcripciones idénticas a las del servidor):

```bat
.venv\Scripts\python verificar.py --audios prueba ^
    --esperado_lm prueba/tts/mms-1b-all_real_s2024.csv --esperado_greedy prueba/greedy/mms-1b-all_real_s2024.csv
```

`prueba/` tiene `referencias.csv`, los 428 audios de no_vistos (test_otros + test_control) y las predicciones de la
evaluación; no se publica (son datos del corpus). Resultado:

| | CER | WER | evaluación de la tesis (CER / WER) | transcripciones idénticas |
|---|---|---|---|---|
| con LM (modelo final) | 9.58 | 39.22 | 9.58 / 39.22 | 99.8 % |
| sin LM | 10.07 | 43.07 | 10.05 / 43.04 | 99.5 % |

428 audios (4.4 s de media), ~2.2 s de proceso por audio en CPU (Ryzen 7 5800H). Las pocas diferencias vienen de que
la evaluación procesó los audios en lotes de 16 con relleno y la demo de a uno (empates numéricos entre espacio y
blanco).

## 5. Cómo está hecho

| archivo | qué hace |
|---|---|
| `demo/transcriptor.py` | carga el modelo según `config.json`; audio → 16 kHz mono → logits → greedy y búsqueda en haz con LM (igual que `04_asr/fase_A/decodificar_lm.py`) |
| `demo/arpa_lm.py` | LM ARPA en Python puro que reemplaza a KenLM (que no tiene binarios para Windows); mismas probabilidades con backoff |
| `demo/app.py`, `demo/index.html` | servidor FastAPI y página (la grabación se convierte a WAV en el navegador, sin ffmpeg) |
| `verificar.py` | prueba funcional contra la evaluación de la tesis |

## 6. Consideraciones y límites

- **No es streaming**: transcribe un enunciado completo (hasta 60 s) en ~2 s en CPU, no palabra por palabra mientras
  se habla.
- **Licencia**: MMS-1b-all es CC-BY-NC 4.0 → la demo es solo para uso académico / no comercial.
- **Datos**: el adaptador y el LM derivan del corpus (SK001 y demás fuentes); no se publican hasta confirmar las
  licencias con el asesor. El LM contiene n-gramas de las frases del corpus.
- **Calidad esperada**: CER ≈ 3 % en la voz de entrenamiento (SK001) y ≈ 9.6 % en voces nuevas (WER ≈ 40 %; ≈ 26 %
  sin contar errores de segmentación). Funciona mejor con frases cortas, habla clara y poco ruido.
- **Privacidad**: el servidor escucha solo en 127.0.0.1 (no es accesible desde otras computadoras) y no guarda audio.
