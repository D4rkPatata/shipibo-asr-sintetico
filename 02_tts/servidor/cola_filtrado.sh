#!/usr/bin/env bash
# Fase B: filtra y exporta a 16 kHz los dos TTS con los mismos criterios (filtrar_tts.py).
set -u
cd ~/tesis/tts/codigo
source ~/venvs/asr/bin/activate
D=~/tesis/datos; J=~/tesis/fase_A/A3/mms-1b-all_real_s1234/modelo_final
for t in tacotron mms; do
  ~/tesis/fase_A/codigo/esperar_gpu.sh 8 python filtrar_tts.py --tts $t --gen ~/tesis/tts/$t --salida $D/sintetico_v1/$t \
      --datos $D/dataset_v1.1 --split $D/split_v1.1.csv --referencias ~/tesis/evaluacion_v1/referencias.csv --juez $J --juez2 ~/tesis/fase_A/A2/whisper-medium_real_s1234/modelo_final || exit 1
done
echo "FILTRADO TERMINADO"
