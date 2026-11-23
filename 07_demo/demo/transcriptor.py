"""Núcleo de la demo: carga el modelo final según config.json y transcribe un audio.

Reproduce el decodificado de la evaluación (04_asr/fase_A/decodificar_lm.py): MMS-1b-all en la revisión fijada, con
el adaptador shp afinado; log-softmax de los logits; greedy y búsqueda en haz con el LM (alpha, beta y unk elegidos
en dev). Para cambiar de modelo basta con reemplazar los archivos de modelo/ y editar config.json (ver LEEME.md).
"""
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

logging.getLogger("pyctcdecode").setLevel(logging.ERROR)   # avisos esperados: sin kenlm (se usa arpa_lm) y alfabeto
RAIZ = Path(__file__).resolve().parent.parent
SR = 16000


def leer_config(ruta=RAIZ / "config.json"):
    c = json.loads(Path(ruta).read_text(encoding="utf-8"))
    for k in ("adaptador", "lm"):
        if c.get(k):
            p = Path(c[k])
            c[k] = str(p if p.is_absolute() else RAIZ / p)
    return c


def cargar_audio(datos):
    """bytes o ruta (wav, flac, ogg, mp3) -> float32 mono a 16 kHz."""
    y, sr = sf.read(io.BytesIO(datos) if isinstance(datos, (bytes, bytearray)) else datos, dtype="float32",
                    always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype("float32")
    return y


class Transcriptor:
    def __init__(self, config=None):
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.c = config or leer_config()
        if self.c.get("hilos_cpu"):
            torch.set_num_threads(int(self.c["hilos_cpu"]))
        ad = Path(self.c["adaptador"])
        self.proc = AutoProcessor.from_pretrained(str(ad), target_lang="shp")
        self.model = Wav2Vec2ForCTC.from_pretrained(self.c["modelo_base"], revision=self.c["revision_base"],
                                                    target_lang="shp", ignore_mismatched_sizes=True)
        self.model.load_adapter("shp")
        pesos = load_file(str(ad / "adapter.shp.safetensors"))
        sobran = set(pesos) - set(self.model.state_dict())
        if sobran:
            raise ValueError(f"El adaptador no corresponde a MMS-1b-all: {sorted(sobran)[:3]}")
        self.model.load_state_dict(pesos, strict=False)
        self.model.eval()
        self.decodificador = self._decodificador() if self.c.get("lm") else None

    def _etiquetas(self):
        vocab = self.proc.tokenizer.get_vocab()
        lab = [None] * len(vocab)
        for t, i in vocab.items():
            lab[i] = t
        lab = [" " if t == "|" else t for t in lab]
        lab[self.proc.tokenizer.pad_token_id] = ""          # el blanco de CTC en MMS es <pad>
        return lab

    def _decodificador(self):
        from pyctcdecode.alphabet import Alphabet
        from pyctcdecode.decoder import BeamSearchDecoderCTC
        from pyctcdecode.language_model import LanguageModel

        from arpa_lm import ArpaLM, unigramas_arpa
        lm = LanguageModel(ArpaLM(self.c["lm"]), unigramas_arpa(self.c["lm"]), alpha=float(self.c["alpha"]),
                           beta=float(self.c["beta"]), unk_score_offset=float(self.c["unk"]))
        return BeamSearchDecoderCTC(Alphabet.build_alphabet(self._etiquetas()), lm)

    @torch.no_grad()
    def log_probs(self, y):
        x = self.proc(y, sampling_rate=SR, return_tensors="pt")
        logits = self.model(x.input_values).logits[0]
        return torch.log_softmax(logits.float(), -1).numpy()

    def transcribir(self, y):
        dur = len(y) / SR
        if dur > self.c["max_segundos"]:
            raise ValueError(f"El audio dura {dur:.0f} s; el máximo es {self.c['max_segundos']} s.")
        if dur < 0.3:
            raise ValueError("El audio es demasiado corto.")
        t0 = time.time()
        lp = self.log_probs(y)
        t1 = time.time()
        sin_lm = self.proc.decode(lp.argmax(-1)).strip()
        con_lm = self.decodificador.decode(lp, beam_width=int(self.c["haz"])).strip() if self.decodificador else None
        t2 = time.time()
        return dict(sin_lm=sin_lm, con_lm=con_lm, duracion_s=round(dur, 2), segundos_modelo=round(t1 - t0, 2),
                    segundos_lm=round(t2 - t1, 2))
