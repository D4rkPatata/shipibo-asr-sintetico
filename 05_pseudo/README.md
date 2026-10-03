# 05 · Pseudoetiquetado

`pseudoetiquetar.ipynb` (Colab, GPU) transcribe los 740 clips no pareados con el ASR semilla (greedy), guardando el
log-prob promedio por token de cada clip, y filtra el pool (los 481 clips que no son control):

| # | filtro | descarta si |
|---|---|---|
| 1 | vacío | sin texto |
| 2 | repeticiones | llegó al tope de tokens, *compression ratio* > 2.4, una palabra 4+ veces seguidas o una frase de 2–4 palabras 3+ veces |
| 3 | caracteres/s | fuera de p0.5–p99.5 del audio pareado real |
| 4 | confianza | fuera del 70 % con mayor log-prob (o bajo un umbral fijo) |

**Evaluación contra el control humano** (259 clips, nunca entran a `metadata_pseudo.csv`): WER/CER por fuente,
WER de los clips que pasarían los filtros (= error esperado de las pseudoetiquetas) y curva WER vs. % conservado.

Validaciones: el filtro de repeticiones no marca ninguna de las 3,586 transcripciones humanas del proyecto; con
p1–p99 el filtro 3 descartaba el 6.6 % de etiquetas humanas correctas (habla más rápida en spots de radio), con p0.5–p99.5 el 3.5 %.

Resultado de la corrida con el ASR semilla: WER 52.2 % / CER 14.1 % en el control; aceptados 327 clips (25.4 min) con
WER esperado 46.0 %. Para repetir con un ASR mejor: cambiar `ASR_DIR` y `SALIDA` (p. ej. `pseudo_v2`).