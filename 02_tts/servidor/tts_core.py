# ---- nucleo de sintesis: se prueba localmente y se copia tal cual al notebook ----
import csv, hashlib, json, os, random, re, time
import numpy as np
import torch

# ============================ TEXTO ============================
# Caracteres que el Tacotron2 no conoce (text/symbols.py). Sin mapear, se descartan en silencio.
_MAPA = {"—": ", ", "–": ", ", "/": " ",               # guiones largos y barra -> pausa/espacio
         '"': "", "“": "", "”": "", "‘": "", "’": ""}  # comillas -> nada
_CORTE_FUERTE = re.compile(r"(?<=[.!?])\s+")
_CORTE_SUAVE = re.compile(r"(?<=[,:;])\s+")


def normalizar(linea):
    t = linea.strip()
    for a, b in _MAPA.items():
        t = t.replace(a, b)
    # ';' es el token de fin de frase del modelo: uno interno podria cortarlo antes de tiempo
    t = t.rstrip(";").replace(";", ",")
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return t


def _n(t):
    return len(t.split())


def partir(texto, max_pal):
    """Parte en <= max_pal palabras: primero en . ! ?, luego en , : ;, y en ultimo caso por palabras."""
    if _n(texto) <= max_pal:
        return [texto]
    salida = []
    for trozo in _CORTE_FUERTE.split(texto):
        if _n(trozo) <= max_pal:
            salida.append(trozo)
            continue
        actual = ""
        for sub in _CORTE_SUAVE.split(trozo):
            cand = (actual + " " + sub).strip()
            if _n(cand) <= max_pal:
                actual = cand
            else:
                if actual:
                    salida.append(actual)
                actual = sub
                while _n(actual) > max_pal:            # sin puntuacion util: corte por palabras
                    p = actual.split()
                    salida.append(" ".join(p[:max_pal]))
                    actual = " ".join(p[max_pal:])
        if actual:
            salida.append(actual)
    return [s.strip(" ,") for s in salida if s.strip(" ,")]


def construir_items(ruta_txt, max_pal, min_pal):
    """Devuelve [(id, texto_original, texto_tts)] con ids estables ligados a la linea de origen."""
    items = []
    with open(ruta_txt, encoding="utf-8-sig") as f:
        for i, linea in enumerate(f):
            if not linea.strip():
                continue
            base = normalizar(linea)
            for j, trozo in enumerate(partir(base, max_pal)):
                if _n(trozo) >= min_pal:
                    items.append((f"L{i:05d}_{j:02d}", linea.strip(), trozo))
    return items


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================ SEMILLAS ============================
def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ============================ MODELOS ============================
def cargar_tacotron(ruta, device, max_decoder_steps, gate_threshold):
    from hparams import create_hparams
    from model import Tacotron2
    hp = create_hparams()
    hp.sampling_rate = 22050
    hp.max_decoder_steps = max_decoder_steps
    hp.gate_threshold = gate_threshold
    m = Tacotron2(hp).to(device)
    if ruta is not None:
        # weights_only=False: desde torch 2.6 el default es True y rechaza estos checkpoints
        ck = torch.load(ruta, map_location=device, weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        if any("mi." in k for k in sd):
            raise RuntimeError("Checkpoint MMI: no compatible con este Tacotron2.")
        m.load_state_dict(sd)
    m.decoder.max_decoder_steps = max_decoder_steps
    m.decoder.gate_threshold = gate_threshold
    return m.eval(), hp


def cargar_hifigan(ruta, ruta_config, device, seed_denoiser):
    from env import AttrDict
    from models import Generator
    from denoiser import Denoiser
    with open(ruta_config) as f:
        h = AttrDict(json.load(f))
    g = Generator(h).to(device)
    if ruta is not None:
        ck = torch.load(ruta, map_location=device, weights_only=False)
        g.load_state_dict(ck["generator"] if "generator" in ck else ck)
    g.eval()
    g.remove_weight_norm()
    # El Denoiser 'normal' estima su sesgo con torch.randn: sin semilla, cada corrida da otro audio.
    fijar_semilla(seed_denoiser)
    den = Denoiser(g, mode="normal")
    return g, h, den


# ============================ SINTESIS ============================
MAX_WAV_VALUE = 32768.0
MIN_FRAMES = 8          # 8 frames x 256 hop = 2048 muestras: margen sobre las 1024 del STFT


def _qc(align, n_pasos, max_pasos, dur_s, texto, cps):
    a = align.float().cpu().numpy()[0]                   # (T_dec, T_enc)
    foco = float(a.max(axis=1).mean())                   # 1.0 = atencion perfectamente enfocada
    fin = float(a[-1].argmax() / max(a.shape[1] - 1, 1)) # ~1.0 si la atencion llego al final del texto
    esperado = len(texto.replace(" ", "")) / cps          # duracion esperada segun la tasa de SK001
    return dict(pasos=n_pasos, toco_max=int(n_pasos >= max_pasos), foco=round(foco, 3),
                cobertura_fin=round(fin, 3), dur_esperada=round(esperado, 2),
                ratio_dur=round(dur_s / esperado, 3) if esperado > 0 else None)


def estado_qc(q, foco_min, cobertura_min, ratio_rango):
    f = []
    if q.get("muy_corto"):
        f.append("muy_corto")
    if q["toco_max"]:
        f.append("no_se_detuvo")
    if q["foco"] < foco_min:
        f.append("atencion_difusa")
    if q["cobertura_fin"] < cobertura_min:
        f.append("no_llego_al_final")
    r = q["ratio_dur"]
    if r is not None and not (ratio_rango[0] <= r <= ratio_rango[1]):
        f.append("duracion_anomala")
    return "ok" if not f else ",".join(f)


@torch.no_grad()
def sintetizar(texto, seed, tacotron, hifigan, denoiser, device, max_pasos, cps,
               fuerza_denoiser=35.0):
    from text import text_to_sequence
    fijar_semilla(seed)                                  # por frase: reproducible y reanudable
    t = texto if texto.endswith(";") else texto + ";"    # token de fin, igual que el notebook original
    seq = torch.LongTensor(text_to_sequence(t, ["basic_cleaners"]))[None, :].to(device)
    _, mel, _, align = tacotron.inference(seq)
    if mel.shape[-1] < MIN_FRAMES:
        # el decoder se detuvo casi de inmediato (falla de atencion). El STFT del denoiser
        # necesita > 1024 muestras y crashearia la corrida entera: se reporta y se sigue.
        q = _qc(align, mel.shape[-1], max_pasos, mel.shape[-1] * 256 / 22050, texto, cps)
        q["muy_corto"] = 1
        return None, q
    y = hifigan(mel.float()).squeeze() * MAX_WAV_VALUE
    y = denoiser(y.view(1, -1), strength=fuerza_denoiser)[:, 0]
    y = y.cpu().numpy().reshape(-1) / MAX_WAV_VALUE
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    if pico > 0:
        y = y * (10 ** (-1 / 20) / pico)                 # pico a -1 dBFS
    dur = len(y) / 22050
    return y.astype(np.float32), _qc(align, mel.shape[-1], max_pasos, dur, texto, cps)