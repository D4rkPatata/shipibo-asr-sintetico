"""Meta MMS-TTS para shipibo (facebook/mms-tts-shp, VITS): genera las 23,558 frases.

Mismas frases y mismo orden que Tacotron2, así las dos voces se comparan sobre el mismo texto.
VITS tiene partes aleatorias (duración y ruido): se fija la semilla de cada frase, derivada de su id,
para que regenerar dé el mismo audio. Salida a 16 kHz (la nativa del modelo), pico a -1 dBFS como Tacotron2.
El tokenizador del modelo pasa a minúsculas y descarta la puntuación (no está en su vocabulario).

Uso:  python generar_mms.py --texto shp_apto_tts.txt --salida ~/tesis/tts/mms
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

MODELO = "facebook/mms-tts-shp"
CAMPOS = ["id", "archivo", "texto", "texto_original", "texto_modelo", "n_palabras", "dur_s", "seed", "estado",
          "t_gen_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--limite", type=int, default=0, help="solo para probar: primeras N frases pendientes")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import comun as K
    from huggingface_hub import model_info
    from transformers import AutoTokenizer, VitsModel, set_seed

    revision = model_info(MODELO).sha                          # se fija y se registra la versión exacta
    tok = AutoTokenizer.from_pretrained(MODELO, revision=revision)
    device = torch.device("cuda")
    modelo = VitsModel.from_pretrained(MODELO, revision=revision).to(device).eval()
    sr = modelo.config.sampling_rate
    vocab = set(tok.get_vocab())

    todas = K.frases(a.texto)
    salida = Path(a.salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    import transformers
    cfg = dict(script="generar_mms.py", modelo=MODELO, revision=revision, SEED=K.SEED, MAX_PALABRAS=K.MAX_PALABRAS,
               MIN_PALABRAS=K.MIN_PALABRAS, OUTPUT_SR=sr, n_frases=len(todas), gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
               sha256_texto=K.sha256(a.texto), licencia="CC-BY-NC-4.0")
    (salida / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    m = K.Manifiesto(salida, CAMPOS)
    pendientes = [f for f in todas if f[0] not in m.hechos]
    if a.limite:
        pendientes = pendientes[:a.limite]
    print(f"{len(todas)} frases | {len(m.hechos)} ya hechas | {len(pendientes)} pendientes", flush=True)
    from tqdm import tqdm
    for id_, original, texto in tqdm(pendientes, mininterval=30):
        seed = K.semilla_de(id_)
        t0 = time.time()
        entrada = tok(texto, return_tensors="pt")
        fila = dict(id=id_, archivo="", texto=texto, texto_original=original, n_palabras=len(texto.split()),
                    seed=seed, texto_modelo="".join(c for c in texto.lower() if c in vocab))  # lo que el modelo lee
        try:
            if entrada.input_ids.shape[1] < 3:
                fila["estado"] = "sin_texto"
            else:
                set_seed(seed)
                with torch.no_grad():
                    y = modelo(**entrada.to(device)).waveform[0].float().cpu().numpy()
                y = K.pico_menos_1db(y)
                fila["archivo"] = K.guardar_wav(salida, id_, y, sr)
                fila["dur_s"] = round(len(y) / sr, 3)
                fila["estado"] = "ok"
        except Exception as e:
            fila["estado"] = f"error: {type(e).__name__}: {e}"[:200]
            torch.cuda.empty_cache()
        fila["t_gen_s"] = round(time.time() - t0, 3)
        m.escribir(fila)
    m.cerrar()
    print("listo", flush=True)


if __name__ == "__main__":
    main()
