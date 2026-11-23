#!/usr/bin/env bash
# Después de la Fase C (D23): C6 = real + MMS-TTS sobre el texto de AmericasNLP, y decodificación con LM de
# todas las corridas MMS (A3, C1–C6). Espera a que terminen los dos trabajadores de cola_C.sh.
set -u
source ~/venvs/asr/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; EV=~/tesis/evaluacion_v1; F=~/tesis/fase_C; A=~/tesis/fase_A; E=$A/codigo/esperar_gpu.sh
LOG=$F/logs; L=~/tesis/lm
LMS="tts=$L/lm_tts.arpa tts_anlp=$L/lm_tts_anlp.arpa"
COMUN="--datos $D/dataset_v1.1 --split $D/split_v1.1.csv --eval $EV"

while tmux has-session -t colaC1 2>/dev/null || tmux has-session -t colaC2 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] Fase C terminada"
cd $A/codigo && python tabla_C.py --referencias $EV/referencias.csv --fase_C $F --fase_A $A > $F/tabla_C.txt

# 1) LM sobre lo ya entrenado (en paralelo: la GPU solo se usa unos minutos por corrida para los logits)
( corridas=$(find $A/A3 $F/C1 $F/C2 $F/C3 $F/C5 $F/C4 -mindepth 1 -maxdepth 1 -type d -name "m*" | sort)   # carpetas, no .csv/.json (C4 lleva "." en el nombre)
  $E 6 python decodificar_lm.py --corridas $corridas --eval $EV --lm $LMS --salida ~/tesis/fase_LM --procesos 16 \
      > $LOG/lm_parte1.log 2>&1 ) &
LM1=$!

# 2) audio MMS-TTS del texto de AmericasNLP y filtrado con los dos jueces (mismos criterios que la Fase B)
cd ~/tesis/tts/codigo
$E 8 python generar_mms.py --texto $D/shp_americasnlp_tts.txt --salida ~/tesis/tts/mms_anlp > ~/tesis/tts/logs/mms_anlp.log 2>&1 || exit 1
$E 8 python filtrar_tts.py --tts mms --gen ~/tesis/tts/mms_anlp --salida $D/sintetico_v1/mms_anlp \
    --datos $D/dataset_v1.1 --split $D/split_v1.1.csv \
    --referencias $EV/referencias.csv --juez $A/A3/mms-1b-all_real_s1234/modelo_final \
    --juez2 $A/A2/whisper-medium_real_s1234/modelo_final > ~/tesis/tts/logs/filtrado_mms_anlp.log 2>&1 || exit 1
echo "[$(date '+%F %T')] mms_anlp filtrado"

# 3) C6, 3 semillas: dos en paralelo (una por GPU) y luego la tercera
cd $A/codigo
c6() { $E 24 python entrenar_mms.py --semilla $1 $COMUN --agregado $D/sintetico_v1/mms_anlp --salida $F/C6 \
          --nombre mms_mmstts-anlp_s$1 > $LOG/mms_mmstts-anlp_s$1.log 2>&1; }
c6 1234 & P1=$!; sleep 300; c6 2024 & P2=$!; wait $P1 $P2; c6 7
echo "[$(date '+%F %T')] C6 terminado"

# 4) LM sobre C6 y tablas
wait $LM1
$E 6 python decodificar_lm.py --corridas $(find $F/C6 -mindepth 1 -maxdepth 1 -type d | sort)--eval $EV --lm $LMS \
    --salida ~/tesis/fase_LM --procesos 16 > $LOG/lm_parte2.log 2>&1
python tabla_C.py --referencias $EV/referencias.csv --fase_C $F --fase_A $A > $F/tabla_C.txt
python tabla_LM.py --referencias $EV/referencias.csv --lm_dir ~/tesis/fase_LM > ~/tesis/fase_LM/tabla_LM.txt
echo "[$(date '+%F %T')] COLA C6 + LM TERMINADA"
