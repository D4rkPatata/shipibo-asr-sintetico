#!/usr/bin/env bash
# Fase C: MMS-1b-all (adaptador shp) con 50 % real / 50 % agregado por lote (D10), mismos 2,000 pasos que A3.
# Uso: cola_C.sh <trabajador>   (se lanzan dos, uno por GPU; cada trabajo lo toma el primero que llega)
# Orden = prioridad: C1–C3 primero, luego aumento, control con Whisper small y escalado.
set -u
cd ~/tesis/fase_A/codigo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; S=$D/sintetico_v1; EV=~/tesis/evaluacion_v1; F=~/tesis/fase_C
source ~/venvs/asr/bin/activate
mkdir -p $F/.tomados $F/logs

COMUN="--datos $D/dataset_v1.1 --split $D/split_v1.1.csv --eval $EV"
trabajos=()
for s in 1234 2024 7; do
  trabajos+=("C1|mms_tacotron_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $S/tacotron")
  trabajos+=("C2|mms_mmstts_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $S/mms")
  trabajos+=("C3|mms_ambos_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $S/tacotron $S/mms")
done
for s in 1234 2024 7; do
  trabajos+=("C5|mms_aug_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $D/aug_v1")
done
for s in 1234 2024 7; do
  trabajos+=("C3w|whisper-small_ambos_s$s|python entrenar_whisper.py --modelo openai/whisper-small --semilla $s $COMUN --agregado $S/tacotron $S/mms")
done
for frac in 0.25 0.5; do
  for s in 1234 2024 7; do
    trabajos+=("C4|mms_ambos${frac}_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $S/tacotron $S/mms --frac_agregado $frac")
  done
done

for t in "${trabajos[@]}"; do
  IFS='|' read -r esc nombre cmd <<< "$t"
  mkdir $F/.tomados/$nombre 2>/dev/null || continue          # atómico: otro trabajador ya lo tomó
  [ -f $F/$esc/$nombre.json ] && continue                     # ya terminado
  echo "[$(date '+%F %T')] trabajador $1: $esc $nombre"
  ./esperar_gpu.sh 24 $cmd --salida $F/$esc --nombre $nombre >> $F/logs/$nombre.log 2>&1 \
    || { echo "[$(date '+%F %T')] FALLÓ $nombre (ver logs/$nombre.log)"; rmdir $F/.tomados/$nombre; }
done
echo "[$(date '+%F %T')] trabajador $1 TERMINADO"
