"""Modelo de lenguaje ARPA en Python puro, compatible con pyctcdecode, para no depender de KenLM.

KenLM (la librería que usó la evaluación en el servidor) no trae binarios para Windows y compilarla exige Visual
Studio. pyctcdecode solo le pide a KenLM cinco cosas (State, BeginSentenceWrite, NullContextWrite, BaseScore y `in`),
así que aquí se implementan con la misma fórmula de backoff de un ARPA:

    log10 p(w | h) = log10 p(w | h)                         si el n-grama (h, w) está en el ARPA
                   = backoff(h) + log10 p(w | h sin su primera palabra)   si no (backoff 0 si h no está)

Las palabras fuera del vocabulario se puntúan como <unk>, igual que KenLM. Da las mismas probabilidades que KenLM
(la verificación de la demo compara las transcripciones con las del servidor: ver verificar.py).
"""
import pyctcdecode.language_model as _plm


class State:
    __slots__ = ("h",)

    def __init__(self):
        self.h = ()


class ArpaLM:
    def __init__(self, ruta):
        self.path = str(ruta).encode("utf-8")
        self.prob, self.backoff = {}, {}
        n = 0
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                if linea.startswith("\\") and linea.endswith("-grams:"):
                    n = int(linea[1:linea.index("-")])
                    continue
                if n == 0 or linea.startswith("\\"):
                    continue
                partes = linea.split("\t")
                ngrama = tuple(partes[1].split(" "))
                self.prob[ngrama] = float(partes[0])
                if len(partes) == 3:
                    self.backoff[ngrama] = float(partes[2])
        self.order = n
        self.unigramas = {g[0] for g in self.prob if len(g) == 1}

    def __contains__(self, palabra):
        return palabra in self.unigramas

    def BeginSentenceWrite(self, estado):
        estado.h = ("<s>",)

    def NullContextWrite(self, estado):
        estado.h = ()

    def BaseScore(self, entrada, palabra, salida):
        w = palabra if palabra in self.unigramas else "<unk>"
        h = entrada.h[-(self.order - 1):] if self.order > 1 else ()
        puntaje = 0.0
        for k in range(len(h), -1, -1):              # del contexto más largo al más corto
            contexto = h[len(h) - k:]
            p = self.prob.get(contexto + (w,))
            if p is not None:
                puntaje += p
                break
            puntaje += self.backoff.get(contexto, 0.0)
        salida.h = (entrada.h + (w,))[-(self.order - 1):] if self.order > 1 else ()
        return puntaje


def unigramas_arpa(ruta):
    """Lo mismo que pyctcdecode.load_unigram_set_from_arpa, pero leyendo en UTF-8 (en Windows abre en cp1252)."""
    out, en_1 = set(), False
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea == "\\1-grams:":
                en_1 = True
            elif linea == "\\2-grams:":
                break
            if en_1 and linea:
                partes = linea.split("\t")
                if len(partes) == 3:
                    out.add(partes[1])
    return out


class _KenlmMinimo:          # lo único que pyctcdecode usa del módulo kenlm al decodificar
    State = State


_plm.kenlm = _KenlmMinimo
