#!/usr/bin/env bash
# Cola A3 (MMS-1b-all afinado, solo real, 3 semillas) y A2 (Whisper medium, solo real, 1 semilla).
set -u
cd ~/tesis/fase_A/codigo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; EV=~/tesis/evaluacion_v1
source ~/venvs/asr/bin/activate

# prueba corta del entrenamiento CTC (4 pasos)
./esperar_gpu.sh 24 python entrenar_mms.py --semilla 1234 --datos $D/dataset_v1.1 --split $D/split_v1.1.csv \
    --eval $EV --salida ~/tesis/fase_A/prueba --nombre prueba_mms --pasos 4 --eval_cada 2 || exit 1

# A3: MMS-1b-all (adaptador shp) afinado, 3 semillas
for s in 1234 2024 7; do
  ./esperar_gpu.sh 24 python entrenar_mms.py --semilla $s --datos $D/dataset_v1.1 --split $D/split_v1.1.csv \
      --eval $EV --salida ~/tesis/fase_A/A3 --nombre mms-1b-all_real_s$s
done
python evaluar.py --referencias $EV/referencias.csv --preds ~/tesis/fase_A/A3 --salida ~/tesis/fase_A/A3/tabla_A3.csv

# A2: Whisper medium, 1 semilla
./esperar_gpu.sh 30 python entrenar_whisper.py --modelo openai/whisper-medium --semilla 1234 --datos $D/dataset_v1.1 \
    --split $D/split_v1.1.csv --eval $EV --salida ~/tesis/fase_A/A2 --nombre whisper-medium_real_s1234
python evaluar.py --referencias $EV/referencias.csv --preds ~/tesis/fase_A/A2 --salida ~/tesis/fase_A/A2/tabla_A2.csv
echo "COLA A2/A3 TERMINADA"
