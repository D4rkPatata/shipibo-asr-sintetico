"""Demo local: FastAPI con una página para subir o grabar un audio en shipibo-konibo y ver su transcripción.

Uso (desde 05_demo/):  .venv\\Scripts\\python demo\\app.py      → abrir http://127.0.0.1:8000
Los audios no se guardan: se transcriben en memoria y se descartan.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcriptor import Transcriptor, cargar_audio  # noqa: E402

AQUI = Path(__file__).resolve().parent
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
