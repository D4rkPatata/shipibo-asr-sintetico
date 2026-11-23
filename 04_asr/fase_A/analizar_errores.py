"""Fase E: análisis de errores del modelo final (D26) en no_vistos, contra el mismo modelo sin LM y contra C6 + LM.

Sistemas (misma semilla, s2024):
  final_greedy  A3 solo real, greedy
  final         A3 solo real + LM texto TTS (modelo final, elegido en dev)
  c6            C6 + LM TTS+AmericasNLP (mejor configuración observada en test)

Análisis:
  1 tipos de error   sustituciones / borrados / inserciones (caracteres y palabras), y cuánto del CER es tildes o espacios
  2 confusiones      pares de caracteres más confundidos, más borrados y más insertados
  3 vocabulario      % de palabras de la referencia acertadas según dónde aparecen: train real, texto del LM, AmericasNLP o nueva
  4 por grupo        CER por fuente/hablante y por conjunto
  5 qué arregla      palabras de la referencia que un sistema acierta y el otro no (LM sobre greedy, C6 sobre final)
  6 ejemplos         clips con más error, y clips donde el LM o C6 cambian más

Uso:  python analizar_errores.py --referencias ~/tesis/evaluacion_v1/referencias.csv --lm_dir ~/tesis/fase_LM \
          --datos ~/tesis/datos --lm_texto ~/tesis/lm --salida ~/tesis/fase_E/errores
"""
import argparse
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from evaluar import normalizar, sin_tildes
from tabla_C import errores, leer

NO_VISTOS = ["test_otros", "test_control"]


def cer(refs, hyps):
    e, n = errores(list(refs), list(hyps))
    return 100 * e.sum() / n.sum()


def tipos(d):
    """Conteo de S/D/I en caracteres y palabras, y CER sin tildes / sin espacios."""
    c = jiwer.process_characters(list(d.ref), list(d.hyp))
    w = jiwer.process_words(list(d.ref), list(d.hyp))
    n_c, n_w = c.substitutions + c.deletions + c.hits, w.substitutions + w.deletions + w.hits
    junto = lambda s: s.str.replace(" ", "", regex=False)
    return dict(CER=cer(d.ref, d.hyp), sust_c=100 * c.substitutions / n_c, borr_c=100 * c.deletions / n_c,
                ins_c=100 * c.insertions / n_c, WER=100 * (w.substitutions + w.deletions + w.insertions) / n_w,
                sust_w=100 * w.substitutions / n_w, borr_w=100 * w.deletions / n_w, ins_w=100 * w.insertions / n_w,
                CER_sin_tildes=cer(d.ref.map(sin_tildes), d.hyp.map(sin_tildes)),
                CER_sin_espacios=cer(junto(d.ref), junto(d.hyp)))


def confusiones(d):
    sus, bor, ins = Counter(), Counter(), Counter()
    o = jiwer.process_characters(list(d.ref), list(d.hyp))
    for r, h, al in zip(o.references, o.hypotheses, o.alignments):
        for a in al:
            rr, hh = r[a.ref_start_idx:a.ref_end_idx], h[a.hyp_start_idx:a.hyp_end_idx]
            if a.type == "substitute":
                sus.update(zip(rr, hh))
            elif a.type == "delete":
                bor.update(rr)
            elif a.type == "insert":
                ins.update(hh)
    v = lambda c: "␣" if c == " " else c
    filas = [dict(tipo="sustitución", ref=v(r), hyp=v(h), n=n) for (r, h), n in sus.most_common(25)]
    filas += [dict(tipo="borrado", ref=v(r), hyp="", n=n) for r, n in bor.most_common(15)]
    filas += [dict(tipo="inserción", ref="", hyp=v(h), n=n) for h, n in ins.most_common(15)]
    return pd.DataFrame(filas)


def aciertos_por_palabra(d):
    """Para cada palabra de cada referencia: si el sistema la transcribió bien (alineamiento por palabras)."""
    o = jiwer.process_words(list(d.ref), list(d.hyp))
    filas = []
    for clip, r, al in zip(d.id, o.references, o.alignments):
        for a in al:
            for k in range(a.ref_start_idx, a.ref_end_idx):
                filas.append(dict(id=clip, pos=k, palabra=r[k], ok=a.type == "equal"))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencias", required=True)
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--lm_texto", required=True, help="carpeta con texto_tts_train.txt y texto_anlp.txt")
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    ref = pd.read_csv(Path(a.referencias).expanduser(), keep_default_na=False)
    ref["ref"] = ref.text_norm.map(normalizar)
    L, D, T = (Path(x).expanduser() for x in (a.lm_dir, a.datos, a.lm_texto))
    sal = Path(a.salida).expanduser()
    sal.mkdir(parents=True, exist_ok=True)

    sistemas = {"final_greedy": L / "greedy" / "mms-1b-all_real_s2024.csv", "final": L / "tts" / "mms-1b-all_real_s2024.csv",
                "c6": L / "tts_anlp" / "mms_mmstts-anlp_s2024.csv"}
    todo = {k: leer(p, ref) for k, p in sistemas.items()}
    nv = {k: d[d.split.isin(NO_VISTOS)].reset_index(drop=True) for k, d in todo.items()}
    out = []

    # 1 tipos de error
    t = pd.DataFrame([dict(sistema=k, conjunto=s, **tipos(d[d.split.isin(NO_VISTOS) if s == "no_vistos" else d.split == s]))
                      for k, d in todo.items() for s in ["dev", "test_sk001", "test_otros", "test_control", "no_vistos"]])
    t.round(2).to_csv(sal / "1_tipos.csv", index=False)
    out.append("1. Tipos de error (% sobre caracteres/palabras de la referencia)\n" + t.round(2).to_string(index=False))

    # 2 confusiones (modelo final, no_vistos)
    c = confusiones(nv["final"])
    c.to_csv(sal / "2_confusiones_final.csv", index=False)
    out.append("2. Confusiones de caracteres, modelo final en no_vistos\n" + c.to_string(index=False))

    # 3 vocabulario
    split = pd.read_csv(D / "split_v1.1.csv")
    meta = pd.read_csv(D / "dataset_v1.1" / "metadata.csv", keep_default_na=False)
    vocab = lambda textos: {w for x in textos for w in normalizar(x).split()}
    v_train = vocab(meta[meta.id.isin(split[split.split == "train"].id)].text_norm)
    v_lm = vocab((T / "texto_tts_train.txt").read_text(encoding="utf-8").splitlines())
    v_anlp = vocab((T / "texto_anlp.txt").read_text(encoding="utf-8").splitlines())
    origen = lambda w: ("1 en train real" if w in v_train else "2 solo en texto del LM (TTS)" if w in v_lm
                        else "3 solo en AmericasNLP" if w in v_anlp else "4 nueva")
    pal = {k: aciertos_por_palabra(d) for k, d in nv.items()}
    base = pal["final"][["id", "pos", "palabra"]].copy()
    base["origen"] = base.palabra.map(origen)
    for k, p in pal.items():
        base[k] = p.ok.values
    v = base.groupby("origen").agg(palabras=("palabra", "size"), **{f"acierto_{k}": (k, "mean") for k in pal})
    v.insert(1, "pct_palabras", 100 * v.palabras / v.palabras.sum())
    v[[f"acierto_{k}" for k in pal]] *= 100
    v.round(1).to_csv(sal / "3_vocabulario.csv")
    out.append("3. % de palabras de la referencia acertadas según dónde aparecen (no_vistos)\n" + v.round(1).to_string())

    # 4 por grupo
    g = pd.DataFrame([dict(split=s, fuente=f, hablante=h, clips=len(x), min=round(x.duration_s.sum() / 60, 1),
                           **{k: cer(x.ref, nv[k].loc[x.index, "hyp"]) for k in nv})
                      for (s, f, h), x in nv["final"].groupby(["split", "source", "speaker_group"])])
    g["c6_menos_final"] = g.c6 - g.final
    g.round(2).to_csv(sal / "4_por_grupo.csv", index=False)
    out.append("4. CER por fuente y hablante (no_vistos)\n" + g.round(2).to_string(index=False))

    # 5 qué arregla cada cosa (palabras de la referencia)
    filas = []
    for nuevo, viejo in [("final", "final_greedy"), ("c6", "final")]:
        arregla, rompe = base[base[nuevo] & ~base[viejo]], base[~base[nuevo] & base[viejo]]
        filas.append(dict(cambio=f"{nuevo} vs {viejo}", arregla=len(arregla), rompe=len(rompe), neto=len(arregla) - len(rompe),
                          **{f"arregla {o}": int((arregla.origen == o).sum()) for o in sorted(base.origen.unique())}))
        pd.concat([arregla.assign(efecto="arregla"), rompe.assign(efecto="rompe")]).to_csv(
            sal / f"5_{nuevo}_vs_{viejo}_palabras.csv", index=False)
    f = pd.DataFrame(filas)
    f.to_csv(sal / "5_que_arregla.csv", index=False)
    out.append("5. Palabras de la referencia que pasan de error a acierto (arregla) o al revés (rompe), no_vistos\n"
               + f.to_string(index=False))

    # 6 ejemplos
    e = nv["final"][["id", "split", "source", "duration_s", "ref"]].copy()
    for k in nv:
        e[f"hyp_{k}"] = nv[k].hyp
        e[f"cer_{k}"] = [cer([r], [h]) for r, h in zip(e.ref, nv[k].hyp)]
    e.round(1).to_csv(sal / "6_ejemplos_todos.csv", index=False)
    cols = ["id", "ref", "hyp_final_greedy", "hyp_final", "hyp_c6", "cer_final_greedy", "cer_final", "cer_c6"]
    peores = e.sort_values("cer_final", ascending=False).head(10)
    lm = e.assign(d=e.cer_final - e.cer_final_greedy).sort_values("d").head(5)
    c6 = e.assign(d=e.cer_c6 - e.cer_final).sort_values("d")
    pd.set_option("display.max_colwidth", 90)
    out.append("6a. Clips con más error del modelo final\n" + peores[cols].round(1).to_string(index=False))
    out.append("6b. Donde más ayuda el LM (final vs greedy)\n" + lm[cols].round(1).to_string(index=False))
    out.append("6c. Donde C6 más ayuda / más empeora frente al final\n"
               + pd.concat([c6.head(5), c6.tail(5)])[cols].round(1).to_string(index=False))

    pd.set_option("display.width", 250)
    texto = "\n\n".join(out)
    (sal / "resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
