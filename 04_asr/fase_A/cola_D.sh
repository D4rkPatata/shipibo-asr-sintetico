#!/usr/bin/env bash
# Fase D (D25): pseudoetiquetas con el maestro elegido en dev (A3 s2024 + LM texto TTS) y validación cruzada por video.
#   D1 = real + pseudo del pliegue 1, D2 = real + pseudo del pliegue 2, Dtodo = real + todas (3 semillas cada uno).
#   Mezcla en proporción al tamaño (sin sobremuestrear: solo ~30 min de pseudoetiquetas), mismos 2,000 pasos que A3.
# Uso: cola_D.sh   (lanza dos trabajadores, uno por GPU)
set -u
cd ~/tesis/fase_A/codigo
source ~/venvs/asr/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=~/tesis/datos; EV=~/tesis/evaluacion_v1; A=~/tesis/fase_A; FD=~/tesis/fase_D; P=$D/pseudo_v2; L=~/tesis/lm
E=./esperar_gpu.sh
mkdir -p $FD/.tomados $FD/logs

# 1) pseudoetiquetas (maestro y LM fijados con dev, ver hiperparametros.csv)
if [ ! -f $P/resumen.json ]; then
  $E 16 python ~/tesis/pseudo/pseudoetiquetar_v2.py --no_pareado $D/no_pareado_v1 --maestro $A/A3/mms-1b-all_real_s2024 \
      --lm $L/lm_tts.arpa --alpha 0.25 --beta 1.0 --unk -5 --juez2 $A/A2/whisper-medium_real_s1234/modelo_final \
      --datos $D/dataset_v1.1 --salida $P > $FD/logs/pseudoetiquetar.log 2>&1 || exit 1
fi

# proporción de reales por lote = tamaño relativo (entrenar_mms.py redondea al micro-lote de 16)
prop() { python -c "
import pandas as pd
r = (pd.read_csv('$D/split_v1.1.csv').split == 'train').sum(); p = len(pd.read_csv('$P/$1/metadata.csv'))
print(round(r / (r + p), 4))"; }
COMUN="--datos $D/dataset_v1.1 --split $D/split_v1.1.csv --eval $EV"
trabajos=()
for s in 1234 2024 7; do
  trabajos+=("D1|mms_pseudo1_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $P/pliegue1 --prop_real $(prop pliegue1)")
  trabajos+=("D2|mms_pseudo2_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $P/pliegue2 --prop_real $(prop pliegue2)")
done
for s in 1234 2024 7; do
  trabajos+=("Dtodo|mms_pseudotodo_s$s|python entrenar_mms.py --semilla $s $COMUN --agregado $P/todo --prop_real $(prop todo)")
done

trabajador() {
  for t in "${trabajos[@]}"; do
    IFS='|' read -r esc nombre cmd <<< "$t"
    mkdir $FD/.tomados/$nombre 2>/dev/null || continue
    [ -f $FD/$esc/$nombre.json ] && continue
    echo "[$(date '+%F %T')] trabajador $1: $esc $nombre"
    $E 24 $cmd --salida $FD/$esc --nombre $nombre >> $FD/logs/$nombre.log 2>&1 \
      || { echo "[$(date '+%F %T')] FALLÓ $nombre"; rmdir $FD/.tomados/$nombre; }
  done
}
trabajador 1 & T1=$!; sleep 300; trabajador 2 & T2=$!; wait $T1 $T2
echo "[$(date '+%F %T')] entrenamiento D terminado"

# 2) LM sobre las corridas D (alpha/beta/unk elegidos en dev, como en D24) y tabla
$E 6 python decodificar_lm.py --corridas $(find $FD/D1 $FD/D2 $FD/Dtodo -mindepth 1 -maxdepth 1 -type d | sort) --eval $EV \
    --lm tts=$L/lm_tts.arpa tts_anlp=$L/lm_tts_anlp.arpa --salida ~/tesis/fase_LM --procesos 16 > $FD/logs/lm_D.log 2>&1
python tabla_D.py --referencias $EV/referencias.csv --fase_D $FD --fase_A $A --fase_LM ~/tesis/fase_LM \
    --pseudo $P/todo.csv > $FD/tabla_D.txt
echo "[$(date '+%F %T')] COLA D TERMINADA"
