#!/usr/bin/env bash
# Cola de la Fase A en el servidor (una tarea detrás de otra, para no pelear por memoria con otros usuarios).
set -u
cd ~/tesis/fase_A/codigo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; EV=~/tesis/evaluacion_v1

# A0: Omnilingual CTC 1B (reintento con lote menor)
source ~/venvs/omni/bin/activate
./esperar_gpu.sh 8 python a0_omnilingual.py --modelo omniASR_CTC_1B_v2 --eval $EV --salida ~/tesis/fase_A/A0 --batch 4
deactivate

source ~/venvs/asr/bin/activate
# prueba corta del entrenamiento (4 pasos): valida todo el flujo antes de las corridas largas
./esperar_gpu.sh 14 python entrenar_whisper.py --modelo openai/whisper-small --semilla 1234 --datos $D/dataset_v1.1 \
    --split $D/split_v1.1.csv --eval $EV --salida ~/tesis/fase_A/prueba --nombre prueba --pasos 4 --eval_cada 2 || exit 1

# A1: Whisper small, solo audio real, 3 semillas
for s in 1234 2024 7; do
  ./esperar_gpu.sh 14 python entrenar_whisper.py --modelo openai/whisper-small --semilla $s --datos $D/dataset_v1.1 \
      --split $D/split_v1.1.csv --eval $EV --salida ~/tesis/fase_A/A1 --nombre whisper-small_real_s$s
done
python evaluar.py --referencias $EV/referencias.csv --preds ~/tesis/fase_A/A1 --salida ~/tesis/fase_A/A1/tabla_A1.csv
echo "COLA A TERMINADA"
