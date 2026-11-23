"""Decodifica modelos MMS-1b-all ya entrenados con búsqueda en haz + modelo de lenguaje de n-gramas (pyctcdecode).

No se reentrena nada: se calculan una vez los logits de evaluacion_v1 para cada corrida (mismos lotes de 16 y fp32
que la transcripción del entrenamiento, así el argmax coincide con el .csv de la corrida) y se decodifican:
  - greedy (control: casi idéntico a la corrida; difiere en empates numéricos espacio/blanco en ~8 % de los clips,
    con el mismo CER) -> <salida>/greedy/, base de la comparación pareada
  - con cada LM, con alpha (peso del LM) y beta (bono por palabra) elegidos **solo con dev** (CER mínimo)

Salida, por LM: <salida>/<lm>/<corrida>.csv (id, prediccion_cruda), igual que las demás predicciones, más
<salida>/hiperparametros.csv con lo elegido en dev.

Uso:  python decodificar_lm.py --corridas ~/tesis/fase_A/A3/mms-1b-all_real_s1234 ... --eval ~/tesis/evaluacion_v1 \
          --lm tts=~/tesis/lm/lm_tts.arpa tts_anlp=~/tesis/lm/lm_tts_anlp.arpa --salida ~/tesis/fase_LM
"""
import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import asr_core as C

REPO, REVISION = "facebook/mms-1b-all", "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
ALPHAS, BETAS = [0.1, 0.25, 0.5, 0.75, 1.0], [0.0, 1.0, 2.0, 3.0]
UNKS = [-10.0, -5.0, -2.0]      # penalización a palabras fuera del LM (~17 % de las palabras de evaluación)
HAZ = 64


@torch.no_grad()
def logits_de(corrida, ev, audios):
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(REPO, revision=REVISION, target_lang="shp")
    model = Wav2Vec2ForCTC.from_pretrained(REPO, revision=REVISION, target_lang="shp", ignore_mismatched_sizes=True)
    model.load_adapter("shp")
    pesos = load_file(str(corrida / "modelo_final" / "adapter.shp.safetensors"))
    assert not set(pesos) - set(model.state_dict())
    model.load_state_dict(pesos, strict=False)
    model = model.to("cuda").eval()
    salida = []
    for i in range(0, len(audios), 16):
        t = audios.iloc[i:i + 16]
        x = proc([sf.read(ev / f, dtype="float32")[0] for f in t.file_name], sampling_rate=16000,
                 return_tensors="pt", padding=True)
        lg = model(x.input_values.to("cuda"), attention_mask=x.attention_mask.to("cuda")).logits
        largos = model._get_feat_extract_output_lengths(x.attention_mask.sum(-1)).tolist()
        lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
        salida += [lp[k, :n] for k, n in enumerate(largos)]
    return proc, salida


def etiquetas(proc):
    vocab = proc.tokenizer.get_vocab()
    lab = [None] * len(vocab)
    for t, i in vocab.items():
        lab[i] = t
    lab = [" " if t == "|" else t for t in lab]
    lab[proc.tokenizer.pad_token_id] = ""                         # el blanco de CTC en MMS es <pad>
    return lab


_dec = {}


def _iniciar(lab, lms):
    from pyctcdecode import build_ctcdecoder
    for nombre, ruta in lms.items():
        _dec[nombre] = build_ctcdecoder(lab, kenlm_model_path=ruta)


def _decodificar(args):
    nombre, alpha, beta, unk, lp = args
    d = _dec[nombre]
    d.reset_params(alpha=alpha, beta=beta, unk_score_offset=unk)
    return d.decode(lp, beam_width=HAZ)


def cer(refs, hyps):
    import jiwer
    return 100 * jiwer.cer([C.normalizar(r) for r in refs], [C.normalizar(h) for h in hyps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridas", nargs="+", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--lm", nargs="+", required=True, help="nombre=ruta.arpa")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--procesos", type=int, default=24)
    a = ap.parse_args()

    ev, sal = Path(a.eval).expanduser(), Path(a.salida).expanduser()
    lms = dict(x.split("=", 1) for x in a.lm)
    lms = {k: str(Path(v).expanduser()) for k, v in lms.items()}
    audios = pd.read_csv(ev / "audios.csv")
    ref = pd.read_csv(ev / "referencias.csv", keep_default_na=False).set_index("id").loc[audios.id]
    es_dev = (ref.split == "dev").values
    hp_ruta = sal / "hiperparametros.csv"
    hp = pd.read_csv(hp_ruta).to_dict("records") if hp_ruta.exists() else []
    lab = None

    for corrida in [Path(c).expanduser() for c in a.corridas]:
        nombre = corrida.name
        cache = sal / "logits" / f"{nombre}.npz"
        if cache.exists():
            lps = list(np.load(cache, allow_pickle=True)["lp"])
            if lab is None:
                from transformers import AutoProcessor
                lab = etiquetas(AutoProcessor.from_pretrained(REPO, revision=REVISION, target_lang="shp"))
        else:
            proc, lps = logits_de(corrida, ev, audios)
            lab = etiquetas(proc)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, lp=np.array(lps, dtype=object))
            torch.cuda.empty_cache()

        # control: greedy sobre los mismos logits = la transcripción de la corrida
        from pyctcdecode import build_ctcdecoder
        greedy = [build_ctcdecoder(lab).decode(lp, beam_width=1) for lp in lps]
        previo = pd.read_csv(corrida.parent / f"{nombre}.csv", keep_default_na=False).set_index("id").loc[audios.id]
        iguales = np.mean([C.normalizar(x) == C.normalizar(y) for x, y in zip(greedy, previo.prediccion_cruda)])
        print(f"{nombre}: greedy coincide con la corrida en {100 * iguales:.1f} % de los clips", flush=True)
        (sal / "greedy").mkdir(parents=True, exist_ok=True)       # base de la comparación pareada: mismos logits
        pd.DataFrame({"id": audios.id, "prediccion_cruda": [g.strip() for g in greedy]}).to_csv(
            sal / "greedy" / f"{nombre}.csv", index=False)

        with Pool(a.procesos, initializer=_iniciar, initargs=(lab, lms)) as pool:
            for lm in lms:
                if any(h["corrida"] == nombre and h["lm"] == lm for h in hp):
                    continue
                dev_lps = [lp for lp, d in zip(lps, es_dev) if d]
                mejor = None
                for al, be, un in itertools.product(ALPHAS, BETAS, UNKS):
                    hyp = pool.map(_decodificar, [(lm, al, be, un, lp) for lp in dev_lps], chunksize=4)
                    c = cer(ref.text_norm[es_dev], hyp)
                    if mejor is None or c < mejor[0]:
                        mejor = (c, al, be, un)
                c_dev, al, be, un = mejor
                hyp = pool.map(_decodificar, [(lm, al, be, un, lp) for lp in lps], chunksize=4)
                (sal / lm).mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"id": audios.id, "prediccion_cruda": [h.strip() for h in hyp]}).to_csv(
                    sal / lm / f"{nombre}.csv", index=False)
                cer_greedy_dev = cer(ref.text_norm[es_dev], [g for g, d in zip(greedy, es_dev) if d])
                hp.append(dict(corrida=nombre, lm=lm, alpha=al, beta=be, unk=un, haz=HAZ, cer_dev_greedy=round(cer_greedy_dev, 2),
                               cer_dev_lm=round(c_dev, 2), greedy_igual_corrida_pct=round(100 * iguales, 1)))
                pd.DataFrame(hp).to_csv(hp_ruta, index=False)
                print(f"  {lm}: alpha {al} beta {be} unk {un} | CER dev greedy {cer_greedy_dev:.2f} -> LM {c_dev:.2f}", flush=True)


if __name__ == "__main__":
    main()
