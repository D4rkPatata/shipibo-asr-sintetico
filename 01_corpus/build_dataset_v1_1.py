"""
build_dataset_v1_1.py
Corpus pareado v1.1 = v1.0 congelado + los clips del workshop que se aprobaron despues (Transcriptions_final.xlsx).

No regenera nada de la v1.0: copia sus wavs y su metadata tal cual, y solo agrega los clips del workshop con
"Paso check auditivo Daniel" = SI que no estaban. Asi siguen validos el split, el aumento y el ASR hechos sobre v1.0.

Particion: split_v1.1.csv = split_v1.csv + los clips nuevos, cada uno en el split de su grupo de hablante
(un grupo de test_otros sigue siendo de voces no vistas). Un clip nuevo se saca ('fuera') si:
  - va a dev/test y su texto esta contenido en el texto del TTS (fuga hacia el audio sintetico), o
  - va a train y su texto es el de un clip de evaluacion.

Salida:
  dataset_v1.1/          wavs/, metadata.csv (+ columna version), metadata_hf.csv, excluidos.csv, stats.json, stats.md
  dataset_v1.1.zip + .sha256
  split_v1.1.csv

Uso:  python build_dataset_v1_1.py        (datos en la carpeta SHIPIBO_DATOS; por defecto D:\\tesis\\Code)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from build_dataset_v1 import (MAX_S, MIN_S, ROOT, TARGET_SR, clean_raw, clip_pct, collect_workshop,
                              load_audio_16k_mono, normalize_text)

V10 = ROOT / "dataset_v1.0"
OUT = ROOT / "dataset_v1.1"
SPLIT_V10 = ROOT / "split_v1.csv"
SPLIT_V11 = ROOT / "split_v1.1.csv"
TEXTO_TTS = ROOT / "Colab" / "TTS_generation" / "shp_apto_tts.txt"
SHA_V10 = "174b61329123a19a4ed9d09b4d8cb229eb7989cddc9d63d9888859c906993a5d"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    zip10 = ROOT / "dataset_v1.0.zip"
    assert sha256(zip10) == SHA_V10, "dataset_v1.0.zip cambió: la v1.1 se construye sobre la v1.0 congelada"
    m10 = pd.read_csv(V10 / "metadata.csv")
    s10 = pd.read_csv(SPLIT_V10)
    assert set(m10.id) == set(s10.id), "split_v1.csv no corresponde a dataset_v1.0"

    # --- workshop aprobado hoy vs lo que ya estaba en v1.0
    excluidos: list[dict] = []
    ws = collect_workshop(excluidos)
    nuevos = [w for w in ws if w["id"] not in set(m10.id)]
    ws_ids = {w["id"] for w in ws}
    retirados = sorted(set(m10.loc[m10.source == "workshop", "id"]) - ws_ids)
    print(f"workshop aprobado: {len(ws)} | ya en v1.0: {len(ws) - len(nuevos)} | nuevos: {len(nuevos)}")
    if retirados:
        print(f"AVISO: {len(retirados)} clips de v1.0 ya no están aprobados en el Excel; se conservan en v1.1 "
              f"(la v1.0 está congelada): {retirados}")

    # --- copiar v1.0 tal cual
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(V10 / "wavs", OUT / "wavs")
    filas = m10.assign(version="1.0").to_dict("records")

    # --- agregar los nuevos (mismo procesamiento que build_dataset_v1)
    for it in nuevos:
        texto = clean_raw(it["transcription"])
        if not texto or texto.lower() == "nan":
            excluidos.append(dict(source="workshop", archivo=str(it["src_path"]), motivo="transcripción vacía"))
            continue
        y, sr0, ch0 = load_audio_16k_mono(it["src_path"])
        dur = len(y) / TARGET_SR
        pct = clip_pct(y)
        sf.write(str(OUT / "wavs" / f"{it['id']}.wav"), np.clip(y, -1.0, 1.0), TARGET_SR, subtype="PCM_16")
        tn = normalize_text(texto)
        filas.append(dict(
            id=it["id"], file_name=f"wavs/{it['id']}.wav", transcription=texto, text_norm=tn, source=it["source"],
            subsource=it["subsource"], speaker_group=it["speaker_group"], duration_s=round(dur, 3),
            en_rango_2_10=MIN_S <= dur <= MAX_S, sr_original=sr0, canales_original=ch0,
            pct_clipping=round(pct, 4), n_palabras=len(tn.split()), n_caracteres=len(tn), fixed=it["fixed"],
            origin_url=it["origin_url"], texto_spa=it.get("texto_spa", ""),
            ruta_original=str(Path(it["src_path"]).relative_to(ROOT)), version="1.1"))
    md = pd.DataFrame(filas)
    nuevos_md = md[md.version == "1.1"]

    # --- particion: cada clip nuevo va al split de su grupo de hablante
    sp = s10.merge(m10[["id", "speaker_group"]], on="id")
    split_grupo = sp[sp.split.isin(["train", "test_otros"])].groupby("speaker_group").split.agg(
        lambda s: s.mode().iloc[0])
    blob = "\n".join(f" {normalize_text(l)} " for l in TEXTO_TTS.read_text(encoding="utf-8-sig").splitlines() if l.strip())
    sp_text = sp.merge(m10[["id", "text_norm"]], on="id")
    textos_eval = set(sp_text.loc[sp_text.split.isin(["dev", "test_sk001", "test_otros"]), "text_norm"])
    filas_split = []
    for r in nuevos_md.itertuples():
        s = split_grupo.get(r.speaker_group)
        if s is None:
            raise SystemExit(f"{r.id}: el grupo {r.speaker_group} no está en split_v1.csv; asígnalo a mano")
        motivo = ""
        if s != "train" and f" {r.text_norm} " in blob:
            s, motivo = "fuera", "texto contenido en el TTS"
        elif s == "train" and r.text_norm in textos_eval:
            s, motivo = "fuera", "texto de un clip de evaluación"
        filas_split.append(dict(id=r.id, split=s, motivo=motivo))
    nuevo_split = pd.DataFrame(filas_split)
    s11 = pd.concat([s10, nuevo_split[["id", "split"]]], ignore_index=True)
    s11.to_csv(SPLIT_V11, index=False)

    # --- salidas
    md.to_csv(OUT / "metadata.csv", index=False, encoding="utf-8")
    md[["file_name", "text_norm"]].rename(columns={"text_norm": "transcription"}).to_csv(
        OUT / "metadata_hf.csv", index=False, encoding="utf-8")
    pd.DataFrame(excluidos).to_csv(OUT / "excluidos.csv", index=False, encoding="utf-8")
    resumen = nuevos_md.merge(nuevo_split, on="id")[["id", "speaker_group", "duration_s", "split", "motivo", "text_norm"]]
    stats = dict(version="1.1", base="dataset_v1.0 (sha256 " + SHA_V10 + ")", sample_rate=TARGET_SR,
                 clips=int(len(md)), horas=round(md.duration_s.sum() / 3600, 3),
                 agregados=resumen.drop(columns="text_norm").to_dict("records"),
                 retirados_del_excel_pero_conservados=retirados)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    lineas = ["# Corpus pareado Shipibo-Konibo v1.1", "",
              f"v1.0 congelada ({len(m10)} clips) + {len(nuevos_md)} clips nuevos del workshop "
              f"({nuevos_md.duration_s.sum():.1f} s). Total: {len(md)} clips, {md.duration_s.sum() / 3600:.3f} h.", "",
              "| id | grupo | dur (s) | split | texto |", "|---|---|---|---|---|"]
    lineas += [f"| {r.id} | {r.speaker_group} | {r.duration_s} | {r.split}{' (' + r.motivo + ')' if r.motivo else ''} | {r.text_norm} |"
               for r in resumen.itertuples()]
    (OUT / "stats.md").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))

    z = ROOT / "dataset_v1.1.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(ROOT))
    h = sha256(z)
    (ROOT / "dataset_v1.1.zip.sha256").write_text(f"{h}  dataset_v1.1.zip\n")
    print(f"\n{z} ({z.stat().st_size / 1e6:.1f} MB)\nsha256 {h}\nsplit: {SPLIT_V11} "
          f"({(s11.split == 'train').sum()} train, {(s11.split == 'test_otros').sum()} test_otros)")


if __name__ == "__main__":
    main()
