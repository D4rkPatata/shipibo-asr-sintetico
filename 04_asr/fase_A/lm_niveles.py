"""Prueba 2: modelo de lenguaje a nivel de palabra, subpalabra y carácter para MMS (CTC), con un decodificador propio.

Motivo: el análisis de errores mostró que el LM de palabras empeora las palabras nunca vistas (17 % del test): solo
conoce palabras completas. Un LM de subpalabras o de caracteres puede formar palabras nuevas (lengua aglutinante).

Para que lo único que cambie sea la unidad:
  - mismos textos que D24 (texto TTS + train real; y lo mismo + AmericasNLP), sin frases de evaluación
  - mismo suavizado (Kneser-Ney interpolado, lm_ngram.py) y el mismo decodificador para los tres niveles:
    búsqueda en haz CTC por prefijos (haz 64, como D24), peso del LM alpha y bono por palabra beta elegidos solo con dev
  - palabra: 4-gramas (el LM de D24); la palabra se puntúa al cerrarse (espacio o fin), con penalización unk
    para palabras fuera del vocabulario (también elegida en dev)
  - subpalabra: SentencePiece unigram entrenado con el mismo texto; la palabra cerrada se segmenta y se puntúan sus
    piezas
  - carácter: cada letra se puntúa al emitirse (el espacio es la unidad ▁)
  Tamaño de vocabulario de subpalabras y orden de cada n-grama: menor perplejidad por palabra en el texto de dev.

Subcomandos:
  preparar     entrena los LM (subpalabra y carácter) y elige orden/vocabulario con dev → <lm_dir>/niveles/
  decodificar  decodifica corridas (logits ya guardados por decodificar_lm.py) → <salida>/<texto>_<nivel>/<corrida>.csv

Uso:  python lm_niveles.py preparar --lm_dir ~/tesis/lm --eval ~/tesis/evaluacion_v1
      python lm_niveles.py decodificar --corridas mms-1b-all_real_s1234 ... --logits ~/tesis/fase_LM/logits \
          --lm_dir ~/tesis/lm --eval ~/tesis/evaluacion_v1 --salida ~/tesis/fase_LM_niveles
"""
import argparse
import itertools
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from evaluar import normalizar

TEXTOS = {"tts": ["texto_tts_train.txt"], "tts_anlp": ["texto_tts_train.txt", "texto_anlp.txt"]}
ORDENES = {"caracter": [4, 5, 6], "subpalabra": [4, 6]}   # KenLM (PyPI) admite hasta orden 6
VOCAB_SP = [500, 2000]
ESP = "▁"
LN10 = math.log(10)
HAZ, PODA = 64, 10.0
ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.0, 1.0, 2.0, 3.0]
UNKS = {"palabra": [-10.0, -5.0, -2.0], "subpalabra": [0.0], "caracter": [0.0]}


# ------------------------------------------------------------------------------------------------ preparar

def a_caracteres(linea):
    return " ".join(ESP if c == " " else c for c in linea.strip())


def logprob_total(arpa, lineas):
    import kenlm
    m = kenlm.Model(str(arpa))
    return sum(m.score(l, bos=True, eos=True) for l in lineas)        # log10


def preparar(a):
    import sentencepiece as spm
    L = Path(a.lm_dir).expanduser()
    N = L / "niveles"
    N.mkdir(exist_ok=True)
    ev = pd.read_csv(Path(a.eval).expanduser() / "referencias.csv", keep_default_na=False)
    dev = [normalizar(t) for t in ev[ev.split == "dev"].text_norm]
    n_pal = sum(len(t.split()) + 1 for t in dev)                       # + </s>
    lm_ngram = Path(__file__).with_name("lm_ngram.py")
    elegido = {}
    for texto, archivos in TEXTOS.items():
        lineas = [normalizar(l) for f in archivos for l in open(L / f, encoding="utf-8")]
        lineas = [l for l in lineas if l]
        (N / f"{texto}.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
        candidatos = []
        # carácter
        (N / f"{texto}_car.txt").write_text("\n".join(a_caracteres(l) for l in lineas) + "\n", encoding="utf-8")
        for o in ORDENES["caracter"]:
            arpa = N / f"{texto}_car_o{o}.arpa"
            if not arpa.exists():
                subprocess.run([sys.executable, str(lm_ngram), "--orden", str(o), "--salida", str(arpa),
                                str(N / f"{texto}_car.txt")], check=True)
            lp = logprob_total(arpa, [a_caracteres(t) for t in dev])
            candidatos.append(dict(nivel="caracter", orden=o, vocab=None, arpa=str(arpa), ppl_palabra=10 ** (-lp / n_pal)))
        # subpalabra
        for v in VOCAB_SP:
            pref = N / f"{texto}_sp{v}"
            if not pref.with_suffix(".model").exists():
                spm.SentencePieceTrainer.train(input=str(N / f"{texto}.txt"), model_prefix=str(pref), vocab_size=v,
                                               model_type="unigram", character_coverage=1.0, bos_id=-1, eos_id=-1,
                                               unk_id=0, num_threads=16, minloglevel=2)
            sp = spm.SentencePieceProcessor(model_file=str(pref.with_suffix(".model")))
            seg = lambda t: " ".join(sp.encode(t, out_type=str))
            (N / f"{texto}_sp{v}.txt").write_text("\n".join(seg(l) for l in lineas) + "\n", encoding="utf-8")
            for o in ORDENES["subpalabra"]:
                arpa = N / f"{texto}_sp{v}_o{o}.arpa"
                if not arpa.exists():
                    subprocess.run([sys.executable, str(lm_ngram), "--orden", str(o), "--salida", str(arpa),
                                    str(N / f"{texto}_sp{v}.txt")], check=True)
                lp = logprob_total(arpa, [seg(t) for t in dev])
                candidatos.append(dict(nivel="subpalabra", orden=o, vocab=v, arpa=str(arpa), sp=str(pref.with_suffix(".model")),
                                       ppl_palabra=10 ** (-lp / n_pal)))
        # palabra (el de D24) como referencia de perplejidad
        arpa = L / f"lm_{texto}.arpa"
        candidatos.append(dict(nivel="palabra", orden=4, vocab=None, arpa=str(arpa),
                               ppl_palabra=10 ** (-logprob_total(arpa, dev) / n_pal)))
        c = pd.DataFrame(candidatos).assign(texto=texto)
        print(c[["texto", "nivel", "vocab", "orden", "ppl_palabra"]].round(1).to_string(index=False), flush=True)
        c.to_csv(N / f"candidatos_{texto}.csv", index=False)
        for nivel, g in c.groupby("nivel"):
            elegido[f"{texto}_{nivel}"] = g.sort_values("ppl_palabra").iloc[0].dropna().to_dict()
    (N / "elegidos.json").write_text(json.dumps(elegido, indent=2, ensure_ascii=False))
    print(json.dumps(elegido, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ decodificar

class LM:
    """Puntúa un prefijo de texto incrementalmente según el nivel. Devuelve log natural."""

    def __init__(self, cfg):
        import kenlm
        self.nivel = cfg["nivel"]
        self.m = kenlm.Model(cfg["arpa"])
        self.sp = None
        if self.nivel == "subpalabra":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=cfg["sp"])
        if self.nivel == "palabra":
            self.vocab = set()
            for linea in open(cfg["arpa"], encoding="utf-8"):
                if linea.startswith("\\2-grams"):
                    break
                p = linea.split("\t")
                if len(p) >= 2:
                    self.vocab.add(p[1].strip())

    def inicio(self):
        import kenlm
        s = kenlm.State()
        self.m.BeginSentenceWrite(s)
        return s

    def _tokens(self, s, toks):
        import kenlm
        lp = 0.0
        for t in toks:
            o = kenlm.State()
            lp += self.m.BaseScore(s, t, o)
            s = o
        return lp * LN10, s

    def caracter(self, s, c):
        """Para el nivel carácter: cada letra (o ▁)."""
        return self._tokens(s, [ESP if c == " " else c])

    def palabra(self, s, w, unk):
        """Para palabra y subpalabra: al cerrarse una palabra."""
        if self.nivel == "palabra":
            lp, s = self._tokens(s, [w])
            return lp + (unk if w not in self.vocab else 0.0), s
        return self._tokens(s, self.sp.encode(w, out_type=str))

    def fin(self, s):
        return self._tokens(s, ["</s>"])[0]


def decodificar_uno(lp, lab, lm, alpha, beta, unk):
    """Búsqueda en haz CTC por prefijos con fusión superficial del LM. lp: (T, V) log-probabilidades."""
    blanco = lab.index("")
    validos = [i for i, c in enumerate(lab) if c and not c.startswith("<")]
    NEG = -1e30
    lse = np.logaddexp
    # haz: prefijo -> [pb, pnb, lm_acum, estado_lm, palabra_en_curso, n_palabras]
    haz = {"": [0.0, NEG, 0.0, lm.inicio(), "", 0]}
    for t in range(lp.shape[0]):
        fila = lp[t]
        tope = fila.max()
        cand = [i for i in validos if fila[i] > tope - PODA]
        nuevo = {}

        def entrada(pref, base):
            if pref not in nuevo:
                nuevo[pref] = [NEG, NEG] + base[2:]
            return nuevo[pref]

        for pref, (pb, pnb, lmv, st, enc, nw) in haz.items():
            tot = lse(pb, pnb)
            e = entrada(pref, haz[pref])
            e[0] = lse(e[0], tot + fila[blanco])
            for i in cand:
                c, p = lab[i], fila[i]
                if c == " " and (not pref or pref.endswith(" ")):
                    e[0] = lse(e[0], tot + p)                       # espacio repetido o inicial: no cambia el texto
                    continue
                ult = pref[-1] if pref else None
                if c == ult:
                    e[1] = lse(e[1], pnb + p)                       # repetición sin blanco: se colapsa
                    if pb <= NEG / 2:
                        continue
                    previo = pb
                else:
                    previo = tot
                nuevo_pref = pref + c
                if nuevo_pref in nuevo:
                    f = nuevo[nuevo_pref]
                else:
                    if lm.nivel == "caracter":
                        inc, st2 = lm.caracter(st, c)
                        f = [NEG, NEG, lmv + inc, st2, "", nw + (c == " ")]
                    elif c == " ":
                        inc, st2 = lm.palabra(st, enc, unk)
                        f = [NEG, NEG, lmv + inc, st2, "", nw + 1]
                    else:
                        f = [NEG, NEG, lmv, st, enc + c, nw]
                    nuevo[nuevo_pref] = f
                f[1] = lse(f[1], previo + p)

        def puntaje(item):
            pref, (pb, pnb, lmv, _, _, nw) = item
            return lse(pb, pnb) + alpha * lmv + beta * nw
        haz = dict(sorted(nuevo.items(), key=puntaje, reverse=True)[:HAZ])

    def final(item):
        pref, (pb, pnb, lmv, st, enc, nw) = item
        extra = 0.0
        if lm.nivel == "caracter":
            if pref and not pref.endswith(" "):
                nw += 1
            extra = lm.fin(st)
        else:
            if enc:
                inc, st = lm.palabra(st, enc, unk)
                extra += inc
                nw += 1
            extra += lm.fin(st)
        return lse(pb, pnb) + alpha * (lmv + extra) + beta * nw
    return max(haz.items(), key=final)[0].strip()


_g = {}


def _iniciar(lab, cfgs):
    _g["lab"] = lab
    _g["lm"] = {k: LM(v) for k, v in cfgs.items()}


def _tarea(args):
    clave, alpha, beta, unk, lp = args
    return decodificar_uno(lp, _g["lab"], _g["lm"][clave], alpha, beta, unk)


def cer(refs, hyps):
    import jiwer
    return 100 * jiwer.cer([normalizar(r) for r in refs], [normalizar(h) for h in hyps])


def decodificar(a):
    from decodificar_lm import REPO, REVISION, etiquetas
    from transformers import AutoProcessor
    lab = etiquetas(AutoProcessor.from_pretrained(REPO, revision=REVISION, target_lang="shp"))
    L, sal = Path(a.lm_dir).expanduser(), Path(a.salida).expanduser()
    elegidos = json.loads((L / "niveles" / "elegidos.json").read_text())
    cfgs = {k: v for k, v in elegidos.items() if k in a.lm}
    ev = Path(a.eval).expanduser()
    audios = pd.read_csv(ev / "audios.csv")
    ref = pd.read_csv(ev / "referencias.csv", keep_default_na=False).set_index("id").loc[audios.id]
    es_dev = (ref.split == "dev").values
    hp_ruta = sal / "hiperparametros.csv"
    sal.mkdir(parents=True, exist_ok=True)
    hp = pd.read_csv(hp_ruta).to_dict("records") if hp_ruta.exists() else []
    with Pool(a.procesos, initializer=_iniciar, initargs=(lab, cfgs)) as pool:
        for corrida in a.corridas:
            lps = list(np.load(Path(a.logits).expanduser() / f"{corrida}.npz", allow_pickle=True)["lp"])
            dev_lps = [lp for lp, d in zip(lps, es_dev) if d]
            for clave, cfg in cfgs.items():
                if (sal / clave / f"{corrida}.csv").exists():
                    continue
                mejor = None
                for al, be, un in itertools.product(ALPHAS, BETAS, UNKS[cfg["nivel"]]):
                    hyp = pool.map(_tarea, [(clave, al, be, un, lp) for lp in dev_lps], chunksize=2)
                    c = cer(ref.text_norm[es_dev], hyp)
                    if mejor is None or c < mejor[0]:
                        mejor = (c, al, be, un)
                c_dev, al, be, un = mejor
                hyp = pool.map(_tarea, [(clave, al, be, un, lp) for lp in lps], chunksize=2)
                (sal / clave).mkdir(exist_ok=True)
                pd.DataFrame({"id": audios.id, "prediccion_cruda": hyp}).to_csv(sal / clave / f"{corrida}.csv", index=False)
                hp.append(dict(corrida=corrida, lm=clave, alpha=al, beta=be, unk=un, haz=HAZ, cer_dev=round(c_dev, 2)))
                pd.DataFrame(hp).to_csv(hp_ruta, index=False)
                print(f"{corrida} {clave}: alpha {al} beta {be} unk {un} | CER dev {c_dev:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--lm_dir", required=True)
    p.add_argument("--eval", required=True)
    d = sub.add_parser("decodificar")
    d.add_argument("--corridas", nargs="+", required=True, help="nombres de corrida (con logits guardados)")
    d.add_argument("--logits", required=True)
    d.add_argument("--lm_dir", required=True)
    d.add_argument("--eval", required=True)
    d.add_argument("--lm", nargs="+", default=[f"{t}_{n}" for t in TEXTOS for n in ("palabra", "subpalabra", "caracter")])
    d.add_argument("--salida", required=True)
    d.add_argument("--procesos", type=int, default=16)
    a = ap.parse_args()
    preparar(a) if a.cmd == "preparar" else decodificar(a)


if __name__ == "__main__":
    main()
