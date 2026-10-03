"""
build_dataset_v1.py
Construye el corpus pareado Shipibo-Konibo v1.0 (Resultado R2 de la tesis).

Fuentes que integra:
  1. SK001   -> corpus de Menéndez & Gómez (2025). Ya curado: se usa tal cual.
  2. Clipped -> clips de videos de YouTube segmentados y transcritos manualmente (2-10 s).
  3. DatasetWorkshop -> grabaciones del workshop. Solo filas con
     "Paso check auditivo Daniel" == "SI" en Transcriptions_final.xlsx.
     Si existe la versión <nombre>_FIXED.wav (editada), se usa esa.

Salida (no toca los originales):
  dataset_v1.0/
    wavs/<id>.wav          16 kHz, mono, PCM 16-bit
    metadata.csv           una fila por clip
    metadata_hf.csv        formato mínimo para HF datasets / entrenamiento (file_name, transcription)
    excluidos.csv          qué se dejó fuera y por qué
    stats.json / stats.md  estadísticas para la Tabla del Capítulo 4

Uso:  python build_dataset_v1.py        (datos en la carpeta SHIPIBO_DATOS; por defecto D:\\tesis\\Code)
Requiere: pandas openpyxl soundfile scipy numpy
"""
from __future__ import annotations

import json
import re
import unicodedata
from math import gcd
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

# ----------------------------------------------------------------------------- config
ROOT = Path(os.environ.get("SHIPIBO_DATOS", r"D:\tesis\Code"))   # datos crudos y salidas
OUT = ROOT / "dataset_v1.0"
TARGET_SR = 16000
MIN_S, MAX_S = 2.0, 10.0        # indicador R2
FILTRAR_DURACION = False        # False = se incluyen todos y se marca `en_rango_2_10`
VERSION = "1.0"

SK_DIR = ROOT / "SK001"
CLIP_DIR = ROOT / "Clipped"
WS_DIR = ROOT / "DatasetWorkshop-20260902T205347Z-1-001" / "DatasetWorkshop"
WS_XLSX = WS_DIR / "Transcriptions_final.xlsx"

# Metadatos de los videos de YouTube (Anexo 2 de la tesis)
YOUTUBE = {
    "cuento1": ("https://www.youtube.com/watch?v=jGme8HsF8R4", "Relatos en lenguas originarias. Shipibo-konibo 1"),
    "cuento2": ("https://www.youtube.com/watch?v=N1dBH5lMlTE", "Relatos en lenguas originarias. Shipibo-konibo 2"),
    "cuento3": ("https://www.youtube.com/watch?v=N7wt4UVQW-Y", "Popo ainbo - La mujer búho"),
    "cuento4": ("https://www.youtube.com/watch?v=_3_H-Dw8MdI", "Jenen yoshin kené - La sirena y el kené"),
    "cuento5": ("https://www.youtube.com/watch?v=dbz2u3k-Ci4", "Wisoino - El wisoino"),
    "cuento6": ("https://www.youtube.com/watch?v=YfjQq03uvMI", "Chaxo betan ani ronin - El venado y la anaconda"),
    "cuento7": ("https://www.youtube.com/watch?v=lwDQp88r15c", "Cuento corto IKINYAMAKANA"),
    "funciones_congreso": ("https://www.youtube.com/watch?v=dnvVEJ35XIk", "Funciones del Congreso | Lengua Shipibo Konibo"),
}
# OJO: verifica que cuento5/6/7 correspondan a esos links; el orden del Anexo 2 no coincide 1 a 1 con los nombres.


# ----------------------------------------------------------------------------- utils
def read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {path}")


def normalize_text(t: str) -> str:
    """Normalización para ASR: NFC, minúsculas, sin puntuación, espacios simples.
    NO cambia la ortografía (tildes y letras se mantienen)."""
    t = unicodedata.normalize("NFC", str(t))
    t = t.lower()
    t = re.sub(r"[¡!¿?.,;:…\"“”«»()\[\]{}—–\-_/\\]", " ", t)
    t = t.replace("'", " ").replace("’", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_raw(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    return re.sub(r"\s+", " ", t).strip()


def load_audio_16k_mono(path: Path) -> tuple[np.ndarray, int, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    ch = y.shape[1]
    y = y.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        y = resample_poly(y, TARGET_SR // g, sr // g).astype("float32")
    return y, sr, ch


def clip_pct(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y) >= 0.999) * 100)


# ----------------------------------------------------------------------------- fuentes
def collect_sk001() -> list[dict]:
    # índice tolerante a nombres con espacios/puntos extra (p.ej. "SK001-0026 .wav", "SK001-1779..wav")
    wavs = {re.sub(r"[\s.]+$", "", p.stem): p for p in (SK_DIR / "wavs").glob("*.wav")}
    rows = []
    for line in read_text_any(SK_DIR / "transcription.csv").splitlines():
        if not line.strip():
            continue
        fid, text = line.split("|", 1)
        fid = fid.strip()
        path = wavs.get(fid)
        if path is None:
            raise FileNotFoundError(f"SK001: no hay audio para {fid}")
        if path.stem != fid:
            print(f"  [aviso] SK001: nombre irregular '{path.name}' -> se usa como {fid}")
        rows.append(dict(
            id=fid, src_path=path, transcription=text,
            source="menendez_gomez_2025", subsource="SK001", speaker_group="SK001",
            fixed=False, origin_url="Menéndez & Gómez (2025) - Anexo I",
        ))
    return rows


_clip_re = re.compile(r"audio_(cuento)0*(\d+)-0*(\d+)|audio_(funciones_congreso)-0*(\d+)")


def _clip_key(name: str) -> tuple[str, int] | None:
    m = _clip_re.fullmatch(name.replace(".wav", "").strip())
    if not m:
        return None
    if m.group(1):
        return f"cuento{int(m.group(2))}", int(m.group(3))
    return m.group(4), int(m.group(5))


def collect_clipped(excluded: list[dict]) -> list[dict]:
    # 1) transcripciones (csv "id|texto" o xlsx de 2 columnas sin cabecera)
    trans: dict[tuple[str, int], str] = {}
    for f in sorted((CLIP_DIR / "transcript").iterdir()):
        if f.suffix == ".csv":
            pairs = [l.split("|", 1) for l in read_text_any(f).splitlines() if l.strip()]
        elif f.suffix == ".xlsx":
            df = pd.read_excel(f, header=None).dropna(how="all")
            pairs = [(str(a), str(b)) for a, b in df.iloc[:, :2].itertuples(index=False)]
        else:
            continue
        video = f.stem.replace("audio_", "")
        ids = [p[0].strip() for p in pairs]
        renumber = len(set(ids)) != len(ids)  # p.ej. audio_cuento7: todos los IDs dicen -01
        if renumber:
            print(f"  [aviso] {f.name}: IDs repetidos -> se asignan por orden de fila (1..{len(pairs)})")
        for i, (fid, text) in enumerate(pairs, start=1):
            key = (_clip_key(fid)[0], i) if renumber else _clip_key(fid)
            if key is None:
                key = (video, i)
            trans[key] = text
    # 2) audios
    rows = []
    for wav in sorted((CLIP_DIR / "wavs").glob("audio_*.wav")):
        key = _clip_key(wav.name)
        if key is None or key not in trans:
            excluded.append(dict(source="youtube", archivo=str(wav.relative_to(ROOT)), motivo="sin transcripción"))
            continue
        video, n = key
        url, title = YOUTUBE.get(video, ("", ""))
        rows.append(dict(
            id=f"YT-{video}-{n:02d}", src_path=wav, transcription=trans.pop(key),
            source="youtube", subsource=video, speaker_group=f"YT-{video}",
            fixed=False, origin_url=url,
        ))
    for (video, n), t in trans.items():
        excluded.append(dict(source="youtube", archivo=f"audio_{video}-{n:02d}", motivo="transcripción sin audio"))
    return rows


def collect_workshop(excluded: list[dict]) -> list[dict]:
    df = pd.read_excel(WS_XLSX)
    check_col = "Paso check auditivo Daniel"
    wavs = {p.name.lower(): p for p in WS_DIR.rglob("*.wav")}
    rows = []
    for _, r in df.iterrows():
        name = str(r["nombre_archivo_audio"]).strip()
        estado = str(r[check_col]).strip().upper()
        grupo = str(r["grupo"]).strip()
        if estado != "SI":
            excluded.append(dict(source="workshop", archivo=name, motivo=f"check auditivo = {estado}; {r.get('Comentario', '')}"))
            continue
        # nombres del Excel que no coinciden con el archivo (Grupo03: audio01_grupo3 -> grupo03_oracion01)
        m = re.fullmatch(r"audio(\d+)_grupo(\d+)\.wav", name, flags=re.I)
        candidates = [name]
        if m:
            candidates.append(f"grupo{int(m.group(2)):02d}_oracion{int(m.group(1)):02d}.wav")
        path, fixed = None, False
        for c in candidates:
            fx = c[:-4] + "_FIXED.wav"
            if fx.lower() in wavs:
                path, fixed = wavs[fx.lower()], True
                break
            if c.lower() in wavs:
                path = wavs[c.lower()]
                break
        if path is None:
            excluded.append(dict(source="workshop", archivo=name, motivo="audio no encontrado"))
            continue
        corregido = str(r.get("Corregido por mi", "")).strip().upper() == "SI"
        if corregido and not fixed:
            print(f"  [aviso] {name}: marcado 'Corregido' pero no hay _FIXED.wav -> se usa el original")
        g = int(re.sub(r"\D", "", grupo) or 0)
        rows.append(dict(
            id=f"WS-g{g:02d}-{Path(path).stem.replace('_FIXED', '')}", src_path=path,
            transcription=r["texto_shipibo"], source="workshop", subsource=grupo,
            speaker_group=f"WS-{grupo}", fixed=fixed, origin_url="Workshop PLN (grabación propia)",
            texto_spa=r.get("texto_spa", ""),
        ))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    excluded: list[dict] = []
    print("Recolectando fuentes...")
    items = collect_sk001() + collect_clipped(excluded) + collect_workshop(excluded)

    meta, seen_ids = [], set()
    for it in items:
        if it["id"] in seen_ids:
            raise ValueError(f"ID duplicado: {it['id']}")
        seen_ids.add(it["id"])
        text_raw = clean_raw(it["transcription"])
        if not text_raw or text_raw.lower() == "nan":
            excluded.append(dict(source=it["source"], archivo=str(it["src_path"]), motivo="transcripción vacía"))
            continue
        y, sr0, ch0 = load_audio_16k_mono(it["src_path"])
        dur = len(y) / TARGET_SR
        in_range = MIN_S <= dur <= MAX_S
        if FILTRAR_DURACION and not in_range:
            excluded.append(dict(source=it["source"], archivo=str(it["src_path"]), motivo=f"duración {dur:.2f}s fuera de rango"))
            continue
        pct_clip = clip_pct(y)
        y = np.clip(y, -1.0, 1.0)
        out_name = f"{it['id']}.wav"
        out_path = OUT / "wavs" / out_name
        sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
        text_norm = normalize_text(text_raw)
        meta.append(dict(
            id=it["id"], file_name=f"wavs/{out_name}", transcription=text_raw, text_norm=text_norm,
            source=it["source"], subsource=it["subsource"], speaker_group=it["speaker_group"],
            duration_s=round(dur, 3), en_rango_2_10=in_range, sr_original=sr0, canales_original=ch0,
            pct_clipping=round(pct_clip, 4), n_palabras=len(text_norm.split()), n_caracteres=len(text_norm),
            fixed=it["fixed"], origin_url=it["origin_url"], texto_spa=it.get("texto_spa", ""),
            ruta_original=str(Path(it["src_path"]).relative_to(ROOT)),
        ))

    md = pd.DataFrame(meta)
    md.to_csv(OUT / "metadata.csv", index=False, encoding="utf-8")
    md[["file_name", "text_norm"]].rename(columns={"text_norm": "transcription"}).to_csv(
        OUT / "metadata_hf.csv", index=False, encoding="utf-8")
    pd.DataFrame(excluded).to_csv(OUT / "excluidos.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------ estadísticas
    def block(d: pd.DataFrame) -> dict:
        words = " ".join(d.text_norm).split()
        return dict(
            clips=int(len(d)), horas=round(d.duration_s.sum() / 3600, 3), minutos=round(d.duration_s.sum() / 60, 1),
            dur_media_s=round(d.duration_s.mean(), 2), dur_mediana_s=round(d.duration_s.median(), 2),
            dur_min_s=round(d.duration_s.min(), 2), dur_max_s=round(d.duration_s.max(), 2),
            pct_en_rango_2_10=round(d.en_rango_2_10.mean() * 100, 1),
            palabras=len(words), vocabulario=len(set(words)),
            grupos_hablante=int(d.speaker_group.nunique()),
        )
    stats = dict(version=VERSION, sample_rate=TARGET_SR, total=block(md),
                 por_fuente={s: block(g) for s, g in md.groupby("source")},
                 excluidos=len(excluded), fixed_usados=int(md.fixed.sum()),
                 clips_con_clipping=int((md.pct_clipping > 0.1).sum()))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Corpus pareado Shipibo-Konibo v{VERSION}", "",
             "| Fuente | Clips | Duración (min) | Horas | Dur. media (s) | % en 2–10 s | Palabras | Vocabulario |",
             "|---|---|---|---|---|---|---|---|"]
    for name, b in list(stats["por_fuente"].items()) + [("**TOTAL**", stats["total"])]:
        lines.append(f"| {name} | {b['clips']} | {b['minutos']} | {b['horas']} | {b['dur_media_s']} | "
                     f"{b['pct_en_rango_2_10']} | {b['palabras']} | {b['vocabulario']} |")
    lines += ["", f"Excluidos: {len(excluded)} (ver excluidos.csv). Versiones _FIXED usadas: {stats['fixed_usados']}."]
    (OUT / "stats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
