"""Demo local: FastAPI con una página para subir o grabar un audio en shipibo-konibo y ver su transcripción.

Uso (desde 07_demo/):  .venv\\Scripts\\python demo\\app.py      → abrir http://127.0.0.1:8000
Los audios no se guardan: se transcriben en memoria y se descartan.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcriptor import Transcriptor, cargar_audio  # noqa: E402

AQUI = Path(__file__).resolve().parent
EJEMPLOS = AQUI.parent / "ejemplos"
app = FastAPI(title="Demo ASR shipibo-konibo")
print("Cargando el modelo (la primera vez descarga MMS-1b-all, ~4 GB)...", flush=True)
T = Transcriptor()
print("Modelo listo:", T.c["nombre"], flush=True)


@app.get("/")
def inicio():
    return FileResponse(AQUI / "index.html")


@app.get("/modelo")
def modelo():
    return {k: T.c[k] for k in ("nombre", "alpha", "beta", "unk", "haz", "max_segundos")} | {"con_lm": bool(T.c.get("lm"))}


def _ejemplos():
    """Audios de ejemplo con su transcripción real (ejemplos/ejemplos.json); solo los que tienen su audio."""
    f = EJEMPLOS / "ejemplos.json"
    if not f.exists():
        return []
    return [e for e in json.loads(f.read_text(encoding="utf-8")) if (EJEMPLOS / e["archivo"]).exists()]


@app.get("/ejemplos")
def ejemplos():
    return [{k: e[k] for k in ("id", "titulo", "referencia", "duracion_s")} for e in _ejemplos()]


@app.get("/ejemplos/{id_ejemplo}/audio")
def audio_ejemplo(id_ejemplo: str):
    for e in _ejemplos():
        if e["id"] == id_ejemplo:
            return FileResponse(EJEMPLOS / e["archivo"], media_type="audio/wav")
    raise HTTPException(404, "Ejemplo no encontrado")


@app.post("/transcribir")
async def transcribir(audio: UploadFile = File(...)):
    datos = await audio.read()
    try:
        y = cargar_audio(datos)
    except Exception:
        raise HTTPException(400, "No se pudo leer el audio. Use WAV, FLAC, OGG o MP3.")
    try:
        return T.transcribir(y)
    except ValueError as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
