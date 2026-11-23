"""Modelo de lenguaje de n-gramas de palabras con suavizado Kneser-Ney interpolado, escrito en formato ARPA
(lo lee kenlm / pyctcdecode). Reemplaza a lmplz de KenLM, que no está instalado en el servidor.

Kneser-Ney interpolado (Chen & Goodman, 1998) con un descuento por orden, D = n1 / (n1 + 2 n2):
  orden más alto:  conteos reales            p(w|h) = max(c(hw) - D, 0) / c(h) + γ(h) p(w|h')
  órdenes menores: conteos de continuación   (en cuántos contextos distintos aparece cada n-grama)
  γ(h) = D · N1+(h·) / c(h), que en ARPA es el peso de retroceso (backoff) de h.
Unigramas interpolados con la uniforme, para dejar masa a <unk>.

Uso:  python lm_ngram.py --orden 4 --salida lm.arpa texto1.txt texto2.txt ...   (una oración por línea, ya normalizada)
"""
import argparse
import math
from collections import Counter, defaultdict


def leer(rutas):
    for r in rutas:
        for linea in open(r, encoding="utf-8"):
            p = linea.split()
            if p:
                yield ["<s>"] + p + ["</s>"]


def entrenar(oraciones, orden):
    conteo = [None] + [Counter() for _ in range(orden)]          # conteo[n][ngrama]
    for o in oraciones:
        for n in range(1, orden + 1):
            for i in range(len(o) - n + 1):
                conteo[n][tuple(o[i:i + n])] += 1
    # conteos que usa cada orden: reales en el más alto, de continuación en los demás
    usar = [None] * (orden + 1)
    usar[orden] = conteo[orden]
    for n in range(1, orden):
        cont = Counter()
        for g in conteo[n + 1]:
            cont[g[1:]] += 1
        for g in conteo[n]:                                       # los que empiezan con <s> no tienen izquierda
            if g[0] == "<s>":
                cont[g] = conteo[n][g]
        usar[n] = cont

    probs, bows = [None] + [dict() for _ in range(orden)], [None] + [dict() for _ in range(orden)]
    vocab = [g[0] for g in conteo[1]]
    # unigramas
    c1 = usar[1]
    n1, n2 = sum(1 for v in c1.values() if v == 1), sum(1 for v in c1.values() if v == 2)
    d = n1 / (n1 + 2 * n2) if n1 + n2 else 0.5
    total = sum(v for g, v in c1.items() if g != ("<s>",))
    gamma = d * sum(1 for g in c1 if g != ("<s>",)) / total
    uniforme = 1 / len(vocab)                                      # palabras sin <s>, más <unk>
    for g, v in c1.items():
        if g != ("<s>",):
            probs[1][g] = max(v - d, 0) / total + gamma * uniforme
    probs[1][("<unk>",)] = gamma * uniforme
    probs[1][("<s>",)] = 0.0                                       # ARPA: log10 = -99
    # órdenes 2..N
    for n in range(2, orden + 1):
        c = usar[n]
        n1, n2 = sum(1 for v in c.values() if v == 1), sum(1 for v in c.values() if v == 2)
        d = n1 / (n1 + 2 * n2) if n1 + n2 else 0.5
        tot_h, tipos_h = defaultdict(int), defaultdict(int)
        for g, v in c.items():
            tot_h[g[:-1]] += v
            tipos_h[g[:-1]] += 1
        for h in tot_h:
            bows[n - 1][h] = d * tipos_h[h] / tot_h[h]
        for g, v in c.items():
            h = g[:-1]
            probs[n][g] = max(v - d, 0) / tot_h[h] + bows[n - 1][h] * prob_menor(probs, bows, g[1:])
    return probs, bows


def prob_menor(probs, bows, g):
    """p(g) en el orden len(g), retrocediendo si el n-grama no se vio (para interpolar con el orden menor)."""
    n = len(g)
    if g in probs[n]:
        return probs[n][g]
    if n == 1:
        return probs[1][("<unk>",)]
    return bows[n - 1].get(g[:-1], 1.0) * prob_menor(probs, bows, g[1:])


def escribir_arpa(probs, bows, orden, ruta):
    log = lambda p: -99.0 if p <= 0 else math.log10(p)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in range(1, orden + 1):
            f.write(f"ngram {n}={len(probs[n])}\n")
        for n in range(1, orden + 1):
            f.write(f"\n\\{n}-grams:\n")
            for g, p in probs[n].items():
                linea = f"{log(p):.6f}\t{' '.join(g)}"
                if n < orden and g in bows[n]:
                    linea += f"\t{math.log10(bows[n][g]):.6f}"
                f.write(linea + "\n")
        f.write("\n\\end\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("textos", nargs="+")
    ap.add_argument("--orden", type=int, default=4)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    probs, bows = entrenar(list(leer(a.textos)), a.orden)
    escribir_arpa(probs, bows, a.orden, a.salida)
    print(f"{a.salida}: " + ", ".join(f"{n}-gramas {len(probs[n])}" for n in range(1, a.orden + 1)))


if __name__ == "__main__":
    main()
