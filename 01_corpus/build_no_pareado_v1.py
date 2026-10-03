"""
build_no_pareado_v1.py
Reune el audio NO pareado de la tesis en un solo corpus (no_pareado_v1), listo para pseudoetiquetar.

Fuentes (cada una con su seguimiento.xlsx):
  audios_ministerio/RECOPILADO_FINAL            MINCUL (COVID-19) y PNUD (Peru Vota Seguro, Respira Amazonia)
  audios_cuentos_no_pareados/AUDIO_RECOPILADO   MIDIS Cuna Mas (cuentos de cuna)
  audio_jesus_contado/AUDIO_RECOPILADO          Global Recordings Network (Palabras de Vida 1)

Entra todo clip cuya columna `revision` NO sea 'eliminar' ni 'edicion' (es decir: ok, ok-otro o vacio).
Los clips del CONTROL_SET que el transcriptor entrego llevan su transcripcion humana: son la evaluacion
del pseudoetiquetado y NO deben usarse para entrenar.

Salida (no toca los originales):
  no_pareado_v1/
    wavs/<id>.wav          16 kHz, mono, PCM 16-bit
    metadata.csv           un clip por fila (todos)
    metadata_control.csv   solo los clips con transcripcion humana (file_name, transcription, text_norm, ...)
    stats.json / stats.md
  no_pareado_v1.zip        para subir a Drive

Uso:  python build_no_pareado_v1.py        (datos en la carpeta SHIPIBO_DATOS; por defecto D:\\tesis\\Code)
"""
from __future__ import annotations

import json
import re
import zipfile
from math import gcd
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from build_dataset_v1 import clean_raw, normalize_text

ROOT = Path(os.environ.get("SHIPIBO_DATOS", r"D:\tesis\Code"))   # datos crudos y salidas
OUT = ROOT / "no_pareado_v1"
TARGET_SR = 16000
VERSION = "1.0"
CONTROL_XLSX = ROOT / "CONTROL_SET" / "transcripcion.xlsx"

FUENTES = [  # carpeta, subcarpeta de clips
    ("audios_ministerio", "RECOPILADO_FINAL"),
    ("audios_cuentos_no_pareados", "AUDIO_RECOPILADO"),
    ("audio_jesus_contado", "AUDIO_RECOPILADO"),
]


def fuente_de(origen: str) -> tuple[str, str]:
    """(source, prefijo de id) segun el recurso de origen."""
    if origen.startswith("nuevas-medidas-covid-19"):
        return "mincul_covid19", "MINCUL"
    if "spot" in origen:
        return "pnud_peru_vota_seguro", "PNUDVS"
    if re.match(r"\d-", origen):
        return "pnud_respira_amazonia", "PNUDRA"
    if origen.startswith("cuentosdecuna"):
        return "midis_cuna_mas", "CUNAMAS"
    if "Words-of-Life" in origen:
        return "grn_palabras_de_vida", "GRN"
    raise ValueError(f"origen desconocido: {origen}")


def a_16k_mono(path: Path) -> tuple[np.ndarray, int, int, str]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    ch = y.shape[1]
    nota = ""
    if ch == 1:
        m = y[:, 0]
    else:
        m = y.mean(axis=1)
        # si L y R estan en contrafase, el promedio se cancela (paso con un cuento de YouTube):
        # en ese caso se usa un solo canal
        rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-12)
        if 20 * np.log10(rms(m) / max(rms(y[:, 0]), rms(y[:, 1]))) < -6:
            m, nota = y[:, 0], "contrafase: se uso el canal izquierdo"
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        m = resample_poly(m, TARGET_SR // g, sr // g).astype("float32")
    return np.clip(m, -1.0, 1.0), sr, ch, nota


def leer_control() -> pd.DataFrame:
    """Transcripciones humanas del CONTROL_SET, por nombre de archivo original."""
    tz = pd.read_excel(CONTROL_XLSX, sheet_name="Trazabilidad", header=2).dropna(subset=["archivo entregado"])
    filas = []
    for hoja in [h for h in pd.ExcelFile(CONTROL_XLSX).sheet_names if re.match(r"B\d_", h)]:
        d = pd.read_excel(CONTROL_XLSX, sheet_name=hoja, header=3)
        d = d[d["nombre del audio"].astype(str).str.endswith(".wav")]
        for r in d.itertuples(index=False):
            filas.append(dict(control_id=r[0], bloque=hoja, transcription=r[1], comentario_control=r[2]))
    c = pd.DataFrame(filas).merge(tz.rename(columns={"archivo entregado": "control_id",
                                                     "archivo original": "archivo"})[["control_id", "archivo"]],
                                  on="control_id", how="left")
    assert c.archivo.notna().all(), "hay clips del control sin trazabilidad"
    c["transcription"] = c.transcription.map(lambda t: clean_raw(t) if isinstance(t, str) and t.strip() else "")
    return c


def main():
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    clips = []
    for carpeta, sub in FUENTES:
        d = pd.read_excel(ROOT / carpeta / "seguimiento.xlsx", header=3)
        d = d[d.archivo.notna() & d.duracion_s.notna() & d.origen.astype(str).str.contains("[a-z]")].copy()
        d["revision"] = d.revision.fillna("").astype(str).str.strip().str.lower()
        d = d[~d.revision.isin(["eliminar", "edicion"])]
        d["ruta"] = [ROOT / carpeta / sub / a for a in d.archivo]
        clips.append(d)
    c = pd.concat(clips, ignore_index=True)
    c[["source", "pref"]] = [fuente_de(o) for o in c.origen]
    c = c.sort_values(["pref", "archivo"]).reset_index(drop=True)
    c["id"] = c.pref + "-" + (c.groupby("pref").cumcount() + 1).map("{:04d}".format)

    ctrl = leer_control()
    c = c.merge(ctrl, on="archivo", how="left")
    falta = set(ctrl.archivo) - set(c.archivo)
    assert not falta, f"clips del control que no estan entre los usables: {falta}"
    c["en_control"] = np.where(c.control_id.isna(), "",
                               np.where(c.transcription.fillna("") != "", "transcrito", "sin_transcribir"))

    meta = []
    for r in c.itertuples():
        y, sr0, ch0, nota = a_16k_mono(r.ruta)
        nombre = f"wavs/{r.id}.wav"
        sf.write(str(OUT / nombre), y, TARGET_SR, subtype="PCM_16")
        tr = r.transcription if isinstance(r.transcription, str) else ""
        meta.append(dict(
            id=r.id, file_name=nombre, source=r.source, subsource=r.origen, speaker_group=f"NP-{r.origen}",
            duration_s=round(len(y) / TARGET_SR, 3), sr_original=sr0, canales_original=ch0, metodo_corte=r.metodo,
            revision=r.revision, comentario_revision=r.comentario if isinstance(r.comentario, str) else "",
            nota_audio=nota, en_control=r.en_control, control_id=r.control_id if isinstance(r.control_id, str) else "",
            transcription_control=tr, text_norm_control=normalize_text(tr) if tr else "",
            comentario_control=r.comentario_control if isinstance(r.comentario_control, str) else "",
            ruta_original=str(Path(r.ruta).relative_to(ROOT)).replace("\\", "/"),
        ))
    md = pd.DataFrame(meta)
    md.to_csv(OUT / "metadata.csv", index=False, encoding="utf-8")
    ct = md[md.en_control == "transcrito"]
    ct[["id", "file_name", "control_id", "transcription_control", "text_norm_control", "source", "subsource",
        "duration_s", "comentario_control"]].rename(
        columns={"transcription_control": "transcription", "text_norm_control": "text_norm"}).to_csv(
        OUT / "metadata_control.csv", index=False, encoding="utf-8")

    def bloque(d):
        return dict(clips=int(len(d)), minutos=round(d.duration_s.sum() / 60, 2))
    stats = dict(version=VERSION, sample_rate=TARGET_SR, total=bloque(md),
                 por_fuente={s: bloque(g) for s, g in md.groupby("source")},
                 control_transcrito=bloque(ct), control_sin_transcribir=bloque(md[md.en_control == "sin_transcribir"]),
                 para_pseudoetiquetar=bloque(md[md.en_control != "transcrito"]),
                 contrafase=int((md.nota_audio != "").sum()))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = [f"# Corpus no pareado Shipibo-Konibo v{VERSION}", "", "| Fuente | Clips | Minutos |", "|---|---|---|"]
    lineas += [f"| {s} | {b['clips']} | {b['minutos']} |" for s, b in stats["por_fuente"].items()]
    lineas += [f"| **TOTAL** | {stats['total']['clips']} | {stats['total']['minutos']} |", "",
               f"- Control con transcripción humana (evaluación, NO entrenar): {ct.shape[0]} clips, {stats['control_transcrito']['minutos']} min",
               f"- Para pseudoetiquetar: {stats['para_pseudoetiquetar']['clips']} clips, {stats['para_pseudoetiquetar']['minutos']} min"]
    (OUT / "stats.md").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    with zipfile.ZipFile(ROOT / "no_pareado_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    print(f"\n{ROOT / 'no_pareado_v1.zip'} ({(ROOT / 'no_pareado_v1.zip').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
