#!/usr/bin/env bash
# Uso: esperar_gpu.sh <GB necesarios> <comando...>
# El servidor es compartido: espera a que alguna GPU tenga esa memoria libre, elige la de más memoria libre y corre el comando ahí.
# Si el comando falla (otro usuario ocupó la memoria a mitad de camino), espera 5 min y reintenta (hasta 4 veces).
# Los scripts son reanudables o cortos, así que reintentar no repite trabajo terminado.
necesita_mb=$(( $1 * 1024 )); shift
for intento in 1 2 3 4; do
  while true; do
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -nr | awk -F', ' -v n="$necesita_mb" '$2 >= n {print $1; exit}')
    [ -n "$gpu" ] && break
    sleep 60
  done
  echo "[$(date '+%F %T')] intento $intento, GPU $gpu libre: $*"
  CUDA_VISIBLE_DEVICES=$gpu "$@" && exit 0
  echo "[$(date '+%F %T')] falló (intento $intento); reintento en 5 min"
  sleep 300
done
echo "[$(date '+%F %T')] se agotaron los reintentos: $*"
exit 1
