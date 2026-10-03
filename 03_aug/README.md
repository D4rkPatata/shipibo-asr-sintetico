# 03 · Aumento de audio

`build_aug_v1.py` genera offline, **solo sobre el split train**, tres versiones de cada clip real:

| versión | método |
|---|---|
| `sp0.9`, `sp1.1` | *speed perturbation* por remuestreo (cambia tempo y tono), receta de Ko et al. (2015) |
| `ruido` | ruido real de OpenSLR 28 `pointsource_noises` (extraído de MUSAN, Apache 2.0) a SNR uniforme en [10, 20] dB |

- Semilla por clip = crc32(SEED + id + tipo): no depende del orden; regenerar da archivos idénticos byte a byte.
- Si la mezcla pasa el pico, se baja todo junto (la SNR no cambia).
- La primera vez descarga solo la carpeta de ruidos del zip de 1.3 GB (lectura por rangos HTTP) a `_cache/`.

```bash
python build_aug_v1.py --split ruta/a/split_v1.csv
```

Salida: `aug_v1/metadata_aug.csv` (columna `origen_id` → clip real), `stats.md`, `aug_v1.zip`.
Resultado: 8,118 clips, 11.5 h. Verificado: SNR real a ≤ 0.02 dB de la pedida; duraciones ×1.111 / ×0.909.
