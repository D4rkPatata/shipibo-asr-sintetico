"""Descarga cada audio de links.txt en su propia carpeta (WAV), listo para cortar."""

import argparse
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

BASE_DIR = Path(__file__).resolve().parent
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"   # requiere ffmpeg en el PATH
# runtimes de JS que yt-dlp puede usar para el desafio de YouTube
JS_RUNTIMES = {r: {} for r in ("deno", "node", "bun") if shutil.which(r)}


def youtube_id(url):
    """Devuelve el id del video si la url es de YouTube, si no None."""
    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host in ("youtu.be", "youtube.be"):
        return u.path.strip("/").split("/")[0] or None
    if host.endswith("youtube.com"):
        if u.path.startswith(("/shorts/", "/embed/", "/live/")):
            return u.path.strip("/").split("/")[1]
        return (parse_qs(u.query).get("v") or [None])[0]
    return None


def slugify(texto):
    """Titulo -> nombre de carpeta seguro en Windows."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^\w\s-]", "", t).strip().lower()
    return re.sub(r"[\s_-]+", "-", t)[:60].strip("-") or "audio"


def slug_from_url(url, titulo=None):
    """SoundCloud: ultimo segmento del path. YouTube: titulo + id del video."""
    vid = youtube_id(url)
    if vid:
        return f"{slugify(titulo)}-{vid}" if titulo else vid
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1] or path.replace("/", "-")
    return re.sub(r'[<>:"/\\|?*]', "-", slug)


def fetch_title(url):
    """Consulta el titulo sin descargar (para nombrar la carpeta)."""
    try:
        opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
                "js_runtimes": JS_RUNTIMES, "remote_components": ["ejs:github"]}
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False).get("title")
    except Exception:
        return None


def read_links(path):
    urls, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#") or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def download(url, dest_dir):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "ffmpeg_location": FFMPEG,
        # YouTube exige resolver un desafio de JS: hace falta un runtime
        # (node/deno) y el solver que yt-dlp baja de su repo.
        "js_runtimes": JS_RUNTIMES,
        "remote_components": ["ejs:github"],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("carpeta", nargs="?", type=Path, default=BASE_DIR,
                   help="carpeta con links.txt (default: la del script)")
    args = p.parse_args()

    base = args.carpeta.resolve()
    links_file = base if base.is_file() else base / "links.txt"
    base = links_file.parent
    if not links_file.exists():
        sys.exit(f"No existe {links_file}")

    urls = read_links(links_file)
    print(f"{len(urls)} links encontrados en {links_file}\n")

    fallidos = []
    for i, url in enumerate(urls, start=1):
        nombre = slug_from_url(url, fetch_title(url) if youtube_id(url) else None)
        dest_dir = base / nombre

        if any(dest_dir.glob("*.wav")):
            print(f"[{i}/{len(urls)}] {nombre} -> ya descargado, se omite")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(urls)}] {nombre}")
        try:
            download(url, dest_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            fallidos.append(url)

    print("\nListo.")
    if fallidos:
        print("Fallaron:")
        for url in fallidos:
            print("  -", url)


if __name__ == "__main__":
    main()
