"""
Corta audios largos en segmentos de voz de 2-6 s usando Silero VAD, para corpus ASR.

Pipeline (lo estandar para armar corpus ASR desde audio crudo sin transcribir):
  1. Silero VAD detecta regiones de voz con granularidad fina.
  2. Se pegan regiones vecinas separadas por pausas cortas, sin pasar de MAX_DUR.
  3. Los trozos largos se parten en su silencio interno mas ancho (recursivo).
  4. Los trozos cortos se intentan unir a un vecino; si no, van a _revisar/.
  5. Se exporta WAV mono 16 kHz PCM16 (lo que esperan wav2vec2 / Whisper / MMS)
     y un manifest.csv con tiempos y niveles para la revision manual.

Uso:
    python cut.py <carpeta_o_wav> [-o SALIDA] [--min 2.0] [--max 6.0] [--dry-run]
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad

VAD_SR = 16000          # Silero trabaja a 16 kHz
OUT_SR = 16000          # sample rate de salida (estandar ASR)
EXTENSIONES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}

# --- parametros de segmentacion -------------------------------------------
MIN_DUR = 2.0           # duracion minima aceptada (s)
MAX_DUR = 6.0           # duracion maxima aceptada (s)
MERGE_GAP = 0.40        # pausa maxima para pegar dos regiones de voz (s)
RESCUE_GAP = 0.90       # pausa maxima para rescatar un segmento corto (s)
PAD = 0.15              # margen agregado a cada lado del corte (s)

# --- filtros de calidad (mandan el segmento a _revisar/) --------------------
MAX_BASS = 0.45         # fraccion de graves sobre la cual se asume musica
MIN_SPEECH_RATIO = 0.55  # fraccion minima del segmento que debe ser voz
MIN_LR_CORR = 0.50      # bajo esta correlacion L/R, sumar a mono cancela la voz

# --- parametros del VAD ----------------------------------------------------
VAD_THRESHOLD = 0.5     # probabilidad minima para considerar voz
MIN_SPEECH_MS = 200     # ignora chispazos de voz mas cortos que esto
MIN_SILENCE_MS = 120    # silencio minimo que separa dos regiones


def load_mono(path, verbose=True):
    """Devuelve (audio_float32_mono, sample_rate_original).

    Sumar L+R cancela la voz cuando los canales vienen decorrelados (reverb
    estereo, ensanchadores). En ese caso se usa un solo canal: sumar dejaria
    la locucion "lejana" y hundiria la deteccion de voz.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        return data[:, 0], sr

    L, R = data[:, 0], data[:, 1]
    # correlacion medida en un tramo central, para no cargar todo el archivo
    n = len(L)
    a, b = n // 3, min(n, n // 3 + int(60 * sr))
    l, r = L[a:b] - L[a:b].mean(), R[a:b] - R[a:b].mean()
    denom = np.linalg.norm(l) * np.linalg.norm(r)
    corr = float(np.dot(l, r) / denom) if denom > 0 else 0.0

    if corr >= MIN_LR_CORR:
        return data.mean(axis=1), sr

    canal = 0 if np.mean(L ** 2) >= np.mean(R ** 2) else 1
    if verbose:
        print(f"    canales decorrelados (corr={corr:+.2f}): uso solo "
              f"{'L' if canal == 0 else 'R'} en vez de sumar")
    return data[:, canal], sr


def normalizar(x, objetivo_dbfs):
    """Ajusta el nivel global al RMS objetivo, sin pasarse de 0 dBFS."""
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 0:
        return x
    g = 10 ** (objetivo_dbfs / 20) / rms
    pico = float(np.max(np.abs(x)))
    if pico * g > 0.99:            # no recortar
        g = 0.99 / max(pico, 1e-9)
    return x * g


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    return torchaudio.functional.resample(t, sr_in, sr_out).squeeze(0).numpy()


def vad_regions(model, wav16, threshold=VAD_THRESHOLD, min_silence_ms=MIN_SILENCE_MS):
    """Regiones de voz [(inicio_s, fin_s), ...] detectadas por Silero."""
    model.reset_states()
    ts = get_speech_timestamps(
        torch.from_numpy(wav16),
        model,
        sampling_rate=VAD_SR,
        threshold=threshold,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=0,          # el margen lo aplicamos nosotros al final
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def merge_regions(regions, max_dur=MAX_DUR, gap=MERGE_GAP):
    """Pega regiones vecinas separadas por pausas cortas, respetando max_dur."""
    out = []
    for r in regions:
        if out and (r[0] - out[-1][1]) <= gap and (r[1] - out[-1][0]) <= max_dur:
            out[-1][1] = r[1]
        else:
            out.append([r[0], r[1]])
    return [tuple(r) for r in out]


def split_long(model, wav16, seg, max_dur=MAX_DUR, min_dur=MIN_DUR):
    """Parte un segmento largo en su silencio interno mas ancho, recursivamente."""
    start, end = seg
    if end - start <= max_dur:
        return [(start, end, "ok")]

    a, b = int(start * VAD_SR), int(end * VAD_SR)

    # cascada de pasadas cada vez mas sensibles para hallar una pausa interna
    best, best_gap = None, 0.0
    for threshold, min_sil in ((0.4, 50), (0.6, 30), (0.75, 20)):
        inner = [(s + start, e + start) for s, e in
                 vad_regions(model, wav16[a:b], threshold=threshold, min_silence_ms=min_sil)]
        # candidatos de corte = punto medio de cada pausa interna
        for prev, nxt in zip(inner, inner[1:]):
            cut = (prev[1] + nxt[0]) / 2
            # el corte debe dejar ambos lados utilizables
            if cut - start < min_dur or end - cut < min_dur:
                continue
            gap = nxt[0] - prev[1]
            if gap > best_gap + 1e-6:
                best, best_gap = cut, gap
        if best is not None:
            break

    if best is None:
        # sin pausa aprovechable: corte duro en trozos iguales (queda marcado)
        n = int(np.ceil((end - start) / max_dur))
        step = (end - start) / n
        return [(start + i * step, start + (i + 1) * step, "corte_duro") for i in range(n)]

    return (split_long(model, wav16, (start, best), max_dur, min_dur)
            + split_long(model, wav16, (best, end), max_dur, min_dur))


def rescue_short(segs, min_dur=MIN_DUR, max_dur=MAX_DUR, gap=RESCUE_GAP):
    """Intenta unir cada segmento corto con el vecino mas cercano."""
    segs = [list(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(segs):
            if s[1] - s[0] >= min_dur:
                continue
            cands = []
            if i > 0:
                cands.append((s[0] - segs[i - 1][1], i - 1))
            if i < len(segs) - 1:
                cands.append((segs[i + 1][0] - s[1], i + 1))
            cands = [
                (g, j) for g, j in cands
                if g <= gap and max(segs[i][1], segs[j][1]) - min(segs[i][0], segs[j][0]) <= max_dur
            ]
            if not cands:
                continue
            _, j = min(cands)
            lo, hi = min(i, j), max(i, j)
            tag = "ok" if segs[lo][2] == "ok" and segs[hi][2] == "ok" else "corte_duro"
            segs[lo:hi + 1] = [[segs[lo][0], segs[hi][1], tag]]
            changed = True
            break
    return [tuple(s) for s in segs]


def pad_segments(segs, total_dur, pad=PAD):
    """Agrega margen a cada lado sin invadir al vecino ni salir del archivo."""
    out = []
    for i, (s, e, tag) in enumerate(segs):
        prev_end = segs[i - 1][1] if i > 0 else 0.0
        next_start = segs[i + 1][0] if i < len(segs) - 1 else total_dur
        out.append((max(0.0, prev_end, s - pad), min(total_dur, next_start, e + pad), tag))
    return out


def dbfs(x):
    if x.size == 0:
        return -99.0
    return 20 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-9))


def bass_ratio(x, sr):
    """Fraccion de energia bajo 250 Hz. En voz ronda 0.1-0.35; la musica de
    fondo (cortinas, golpes de bajo de los spots) se dispara sobre 0.45."""
    if x.size < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spec[freqs < 250].sum() / (spec.sum() + 1e-12))


def speech_ratio(model, x16):
    """Fraccion del segmento que el VAD considera voz."""
    step = 512
    if x16.size < step:
        return 0.0
    model.reset_states()
    p = [model(torch.from_numpy(np.ascontiguousarray(x16[i:i + step])), VAD_SR).item()
         for i in range(0, len(x16) - step + 1, step)]
    model.reset_states()
    return float(np.mean(np.array(p) > 0.5))


def segment_file(model, wav_path, out_root, args):
    audio, sr = load_mono(wav_path)
    if args.norm is not None:
        audio = normalizar(audio, args.norm)
    total_dur = len(audio) / sr
    wav16 = resample(audio, sr, VAD_SR)

    segs = merge_regions(vad_regions(model, wav16), args.max, args.gap)

    expanded = []
    for s in segs:
        expanded.extend(split_long(model, wav16, s, args.max, args.min))
    segs = rescue_short(expanded, args.min, args.max, args.rescue)
    segs = pad_segments(segs, total_dur, args.pad)

    # un audio por carpeta -> nombra por la carpeta; varios juntos -> por el archivo
    hermanos = sum(1 for x in wav_path.parent.iterdir()
                   if x.is_file() and x.suffix.lower() in EXTENSIONES)
    if wav_path.parent != out_root and hermanos == 1:
        stem = wav_path.parent.name
    else:
        stem = re.sub(r'[<>:"/\\|?*\s]+', "-", wav_path.stem).strip("-")
    dest = out_root / stem
    short_dest = dest / "_revisar"

    rows, kept, short = [], 0, 0
    for i, (s, e, tag) in enumerate(segs, start=1):
        dur = e - s
        chunk = audio[int(s * sr):int(e * sr)]
        if chunk.size == 0:
            continue
        out16 = resample(chunk, sr, args.sr)
        bass = bass_ratio(out16, args.sr)
        voz = speech_ratio(model, wav16[int(s * VAD_SR):int(e * VAD_SR)])

        # motivos para mandarlo a revision manual en vez de al corpus
        if dur < args.min:
            estado = "corto"
        elif bass > args.bass:
            estado = "musica"
        elif voz < args.voz:
            estado = "poca_voz"
        else:
            estado = tag
        aparta = estado in ("corto", "musica", "poca_voz")

        folder = short_dest if aparta else dest
        name = f"{stem}_{i:04d}.wav"
        if not args.dry_run:
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(str(folder / name), out16, args.sr, subtype="PCM_16")

        rows.append({
            "archivo": name,
            "carpeta": "_revisar" if aparta else ".",
            "origen": wav_path.name,
            "inicio_s": round(s, 3),
            "fin_s": round(e, 3),
            "dur_s": round(dur, 3),
            "voz_pct": round(100 * voz),
            "grave_pct": round(100 * bass),
            "rms_dbfs": round(dbfs(out16), 1),
            "pico": round(float(np.max(np.abs(out16))) if out16.size else 0.0, 3),
            "estado": estado,
        })
        short += aparta
        kept += not aparta

    return rows, kept, short, total_dur


def main():
    # la consola de Windows es cp1252 y revienta con nombres acentuados
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", type=Path, help="carpeta con wavs (recursivo) o un wav suelto")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="carpeta de salida (default: <entrada>/_cortes)")
    p.add_argument("--min", type=float, default=MIN_DUR, help=f"duracion minima en s (default {MIN_DUR})")
    p.add_argument("--max", type=float, default=MAX_DUR, help=f"duracion maxima en s (default {MAX_DUR})")
    p.add_argument("--gap", type=float, default=MERGE_GAP,
                   help=f"pausa max. para pegar regiones (default {MERGE_GAP})")
    p.add_argument("--rescue", type=float, default=RESCUE_GAP,
                   help=f"pausa max. para rescatar cortos (default {RESCUE_GAP})")
    p.add_argument("--pad", type=float, default=PAD, help=f"margen a cada lado en s (default {PAD})")
    p.add_argument("--sr", type=int, default=OUT_SR, help=f"sample rate de salida (default {OUT_SR})")
    p.add_argument("--bass", type=float, default=MAX_BASS,
                   help=f"fraccion de graves sobre la cual se marca como musica (default {MAX_BASS})")
    p.add_argument("--norm", type=float, default=None, metavar="DBFS",
                   help="nivela el audio a este RMS antes de cortar, p.ej. -23 "
                        "(default: sin tocar el nivel)")
    p.add_argument("--voz", type=float, default=MIN_SPEECH_RATIO,
                   help=f"fraccion minima de voz en el segmento (default {MIN_SPEECH_RATIO})")
    p.add_argument("--dry-run", action="store_true", help="solo reporta, no escribe wavs")
    args = p.parse_args()

    entrada = args.entrada.resolve()
    if entrada.is_dir():
        wavs = sorted(w for w in entrada.rglob("*")
                      if w.suffix.lower() in EXTENSIONES and "_cortes" not in w.parts)
    elif entrada.is_file():
        wavs, entrada = [entrada], entrada.parent
    else:
        sys.exit(f"No existe: {entrada}")
    if not wavs:
        sys.exit(f"No se encontraron .wav en {entrada}")

    out_root = (args.out or entrada / "_cortes").resolve()
    model = load_silero_vad()

    all_rows, total_kept, total_short, total_in = [], 0, 0, 0.0
    for w in wavs:
        rows, kept, short, dur = segment_file(model, w, out_root, args)
        all_rows.extend(rows)
        total_kept += kept
        total_short += short
        total_in += dur
        extra = f" (+{short} a _revisar)" if short else ""
        print(f"{w.parent.name}/{w.name}: {dur:7.1f}s -> {kept} segmentos{extra}")

    if all_rows and not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        destino = out_root / "manifest.csv"
        try:
            f = open(destino, "w", newline="", encoding="utf-8")
        except PermissionError:
            # el csv suele quedar abierto en Excel; no vale la pena perder el trabajo
            destino = out_root / f"manifest_{int(time.time())}.csv"
            print(f"\nmanifest.csv esta bloqueado (abierto en otro programa) -> escribo {destino.name}")
            f = open(destino, "w", newline="", encoding="utf-8")
        with f:
            wr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wr.writeheader()
            wr.writerows(all_rows)

    ok = [r["dur_s"] for r in all_rows if r["estado"] != "corto"]
    print("\n" + "=" * 62)
    print(f"entrada           : {len(wavs)} archivos, {total_in / 60:.1f} min")
    print(f"segmentos {args.min:.0f}-{args.max:.0f} s   : {total_kept}   (a revisar: {total_short})")
    if ok:
        print(f"voz conservada    : {sum(ok) / 60:.1f} min  ({100 * sum(ok) / total_in:.0f}% del audio)")
        print(f"duracion media    : {np.mean(ok):.2f} s   mediana: {np.median(ok):.2f} s")
        duros = sum(1 for r in all_rows if r["estado"] == "corte_duro")
        if duros:
            print(f"cortes duros      : {duros}  (se partieron sin pausa clara)")
    if not args.dry_run:
        print(f"salida            : {out_root}")
    print("=" * 62)


if __name__ == "__main__":
    main()
