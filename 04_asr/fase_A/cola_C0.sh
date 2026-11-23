#!/usr/bin/env bash
# C0 (D30): solo sintético. MMS-1b-all (adaptador shp) entrenado solo con los dos TTS filtrados (sin audio real en el
# entrenamiento: --prop_real 0), mismos 2,000 pasos, lote y lr que C3. El mejor checkpoint se elige con el dev real,
# igual que en todos los escenarios. Después: LM de palabras (tts, tts_anlp), LM por nivel y tablas.
# Uso: tmux new -d -s colaC0 'bash cola_C0.sh > ~/tesis/fase_C/logs/cola_C0.log 2>&1'
set -u
source ~/venvs/asr/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; S=$D/sintetico_v1; EV=~/tesis/evaluacion_v1; F=~/tesis/fase_C; A=~/tesis/fase_A
E=$A/codigo/esperar_gpu.sh; LOG=$F/logs; L=~/tesis/lm
COMUN="--datos $D/dataset_v1.1 --split $D/split_v1.1.csv --eval $EV"
cd $A/codigo

c0() { [ -f $F/C0/mms_solosint_s$1.json ] && return 0
       $E 24 python entrenar_mms.py --semilla $1 $COMUN --agregado $S/tacotron $S/mms --prop_real 0 \
          --salida $F/C0 --nombre mms_solosint_s$1 > $LOG/mms_solosint_s$1.log 2>&1; }
c0 1234 & P1=$!; sleep 300; c0 2024 & P2=$!; wait $P1 $P2; c0 7
echo "[$(date '+%F %T')] C0 entrenado"

$E 6 python decodificar_lm.py --corridas $(find $F/C0 -mindepth 1 -maxdepth 1 -type d | sort) --eval $EV \
    --lm tts=$L/lm_tts.arpa tts_anlp=$L/lm_tts_anlp.arpa --salida ~/tesis/fase_LM --procesos 16 > $LOG/lm_C0.log 2>&1
python lm_niveles.py decodificar --corridas mms_solosint_s1234 mms_solosint_s2024 mms_solosint_s7 \
    --logits ~/tesis/fase_LM/logits --lm_dir $L --eval $EV --salida ~/tesis/fase_LM_niveles --procesos 16 \
    > $LOG/niveles_C0.log 2>&1
echo "[$(date '+%F %T')] C0 decodificado"

python tabla_C.py --referencias $EV/referencias.csv --fase_C $F --fase_A $A > $F/tabla_C.txt
python tabla_LM.py --referencias $EV/referencias.csv --lm_dir ~/tesis/fase_LM > ~/tesis/fase_LM/tabla_LM.txt
python tabla_niveles.py --referencias $EV/referencias.csv --niveles ~/tesis/fase_LM_niveles --greedy ~/tesis/fase_LM/greedy \
    --datos $D --lm_texto $L > ~/tesis/fase_LM_niveles/tabla.log 2>&1
python metricas_relajadas.py --referencias $EV/referencias.csv --tesis ~/tesis --salida ~/tesis/fase_E/metricas_relajadas \
    > ~/tesis/fase_E/metricas_relajadas/log.txt 2>&1
echo "[$(date '+%F %T')] COLA C0 TERMINADA"
