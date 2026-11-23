"""Tacotron2 + HiFi-GAN (Menéndez y Erasmo): genera las frases que Colab no generó.

Mismo núcleo (tts_core.py), mismas frases, mismo orden y misma semilla por frase que generation.ipynb.
Colab generó las primeras 11,779 del orden barajado; aquí van las 11,779 restantes, así el total
cubre las 23,558 frases sin repetir ninguna. Se ejecuta en el servidor con GPU.

Uso:  python generar_tacotron.py --texto shp_apto_tts.txt --salida ~/tesis/tts/tacotron_parte2
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

# hparams.py del fork importa tensorflow solo para registrar mensajes: un módulo mínimo basta
_tf = types.ModuleType("tensorflow")
_tf.compat = types.SimpleNamespace(v1=types.SimpleNamespace(logging=types.SimpleNamespace(info=lambda *a, **k: None)))
sys.modules.setdefault("tensorflow", _tf)

import numpy as np
import torch

REPOS = {"tacotron2": ("https://github.com/rmcpantoja/tacotron2.git", "0b33108d515c8dfaef4cc6fc1c4691c9c419bfa4"),
         "hifi-gan": ("https://github.com/justinjohn0306/hifi-gan", "11a178893bad3df9a918b405e3cf9c48ee682487")}
MODELOS = {"tacotron": "1Hbun7QzSy5wXXQnM-7rd8XIqjp-_Cxkf",   # SKF2, ID público del notebook de inferencia
           "hifigan": "1-FC41PwOvFmoacWgMlrIgC6LwaI0sT5l"}    # g_00000000
MAX_DUR_S, STOP_THRESHOLD, FUERZA_DENOISER = 20, 0.5, 35.0
CPS_SK001 = 10.31
MAX_PASOS = int(MAX_DUR_S * 22050 / 256)
QC = (0.30, 0.85, (0.4, 2.5))                                   # foco, cobertura, ratio de duración
CAMPOS = ["id", "archivo", "texto", "texto_original", "n_palabras", "dur_s", "seed", "pasos", "toco_max", "foco",
          "cobertura_fin", "dur_esperada", "ratio_dur", "estado", "t_gen_s", "corrida"]


def preparar(base):
    """Clona los repos en el commit fijado y baja los dos checkpoints (una sola vez)."""
    for carpeta, (url, commit) in REPOS.items():
        d = base / carpeta
        if not d.is_dir():
            subprocess.run(["git", "clone", "-q", url, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", commit], check=True)
    for p in (base / "hifi-gan", base / "tacotron2"):          # mismo orden que el notebook
        sys.path.append(str(p))
    import gdown
    rutas = {}
    for k, fid in MODELOS.items():
        destino = base / "modelos" / k
        destino.mkdir(parents=True, exist_ok=True)
        ya = [p for p in destino.iterdir() if p.is_file()]
        rutas[k] = ya[0] if ya else Path(gdown.download(id=fid, output=str(destino) + "/", quiet=True))
    return rutas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--base", default=str(Path.home() / "tesis" / "tts"))
    ap.add_argument("--limite", type=int, default=0, help="solo para probar: primeras N frases pendientes")
    a = ap.parse_args()
    base = Path(a.base).expanduser()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import comun as K
    C = K.C
    rutas = preparar(base)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    tacotron, hp = C.cargar_tacotron(rutas["tacotron"], device, MAX_PASOS, STOP_THRESHOLD)
    hifigan, h, denoiser = C.cargar_hifigan(rutas["hifigan"], base / "hifi-gan" / "config_v1.json", device, K.SEED)

    todas = K.frases(a.texto)
    objetivo = todas[K.N_COLAB:]                                   # las que Colab no generó
    salida = Path(a.salida).expanduser()
    cfg = dict(script="generar_tacotron.py", corrida="servidor_parte2", SEED=K.SEED, MAX_PALABRAS=K.MAX_PALABRAS,
               MIN_PALABRAS=K.MIN_PALABRAS, MAX_DUR_S=MAX_DUR_S, STOP_THRESHOLD=STOP_THRESHOLD,
               FUERZA_DENOISER=FUERZA_DENOISER, OUTPUT_SR=22050, n_frases_total=len(todas),
               frases_de_esta_corrida=f"orden barajado [{K.N_COLAB}:{len(todas)}]", n_objetivo=len(objetivo),
               gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
               python=platform.python_version(), sha256_texto=K.sha256(a.texto),
               sha256_tacotron=K.sha256(rutas["tacotron"]), sha256_hifigan=K.sha256(rutas["hifigan"]),
               commits={k: v[1] for k, v in REPOS.items()}, n_symbols=hp.n_symbols)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    m = K.Manifiesto(salida, CAMPOS)
    pendientes = [f for f in objetivo if f[0] not in m.hechos]
    if a.limite:
        pendientes = pendientes[:a.limite]
    print(f"{len(objetivo)} frases objetivo | {len(m.hechos)} ya hechas | {len(pendientes)} pendientes", flush=True)
    from tqdm import tqdm
    for id_, original, texto in tqdm(pendientes, mininterval=30):
        seed = K.semilla_de(id_)
        t0 = time.time()
        fila = dict(id=id_, archivo="", texto=texto, texto_original=original, n_palabras=len(texto.split()),
                    seed=seed, corrida="servidor_parte2")
        try:
            y, q = C.sintetizar(texto, seed, tacotron, hifigan, denoiser, device, MAX_PASOS, CPS_SK001, FUERZA_DENOISER)
            fila.update({c: q.get(c) for c in CAMPOS if c in q})
            fila["estado"] = C.estado_qc(q, *QC)
            if y is not None:
                fila["archivo"] = K.guardar_wav(salida, id_, y, 22050)
                fila["dur_s"] = round(len(y) / 22050, 3)
        except Exception as e:                                       # una frase rota no detiene las demás
            fila["estado"] = f"error: {type(e).__name__}: {e}"[:200]
            torch.cuda.empty_cache()
        fila["t_gen_s"] = round(time.time() - t0, 3)
        m.escribir(fila)
    m.cerrar()
    print("listo", flush=True)


if __name__ == "__main__":
    main()
