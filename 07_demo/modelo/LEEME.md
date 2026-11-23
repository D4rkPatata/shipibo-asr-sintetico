# modelo/

Archivos del modelo que usa la demo (no se suben al repositorio: son derivados del corpus; ver ../LEEME.md).

| archivo | qué es | de dónde sale |
|---|---|---|
| `adapter.shp.safetensors` | adaptador shp afinado (~9 MB) | `~/tesis/fase_A/A3/mms-1b-all_real_s2024/modelo_final/` |
| `vocab.json`, `tokenizer_config.json`, `processor_config.json` | procesador (alfabeto, extractor de audio) | misma carpeta |
| `lm_tts.arpa` | LM de palabras, 4-gramas Kneser-Ney (texto TTS + train real) | `~/tesis/lm/lm_tts.arpa` |

El modelo base (MMS-1b-all, revisión fijada en `config.json`) no está aquí: se descarga de Hugging Face la primera vez.
