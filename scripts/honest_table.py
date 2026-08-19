# -*- coding: utf-8 -*-
"""
honest_table.py — single source of truth for the manuscript's honest numbers.

Uniformly aggregates ANY tag's rows with the SAME semantics as
recompute_honest.py, so that old tags (loaded from {tag}_honest.jsonl, which
carry pure_commentary/rew_clean/honest fre/tot/parts/nli) and round-7 tags
(L1-L5, OOD_*_fix, loaded from raw {tag}.jsonl, whose rew is already the
commentary-cleaned text with honest fields) are directly comparable:

  * pure(row)  = r["pure_commentary"] if present (honest files), else
                 (empty rew) — round-7 rows store the CLEANED text in rew, so
                 an empty rew means the row collapsed to pure commentary.
  * rew(row)   = rew_clean or rew.
  * FRE stats over VALID rows only (fre>0 / non-pure); pure rows are generation
    failures and are reported as pure%, not as fre=0 outliers.
  * inband_10 / inband_12 over valid FRE vs LEVEL_TARGET_FRE.
  * tot over ALL rows (failures score 0 — failure-weighted total reward), plus
    tot_valid for context.
  * halluc = mean NLI contradiction over VALID rows only.
  * parts means over valid rows; attempts mean over all rows.

Outputs results_dir/honest_table.json + console markdown tables that map 1:1 to
manuscript tables (main chain, ablations, loop depth, OOD, seeds, baselines, ADV).

Usage: python honest_table.py [results_dir]
"""
import json
import os
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
RD = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\33378\aigc_wechat\results\legal"

LEVEL_TARGET_FRE = {"beginner": 63.0, "intermediate": 45.0, "advanced": 26.0}
LEVEL_LABEL = {"beginner": "beg", "intermediate": "int", "advanced": "adv"}

TAGS = ["G1", "G2", "G3", "G4", "G5", "G6",
        "A_full", "A_diff", "A_term", "A_copy", "A_faith", "A_fmt",
        "A_full_full_budget", "A_diff_full_budget",
        "G6_pure", "G6_pure_s11", "G6_pure_s21", "G6_pure_s42", "M2",
        "ADV", "ADV_full", "G6_full107",
        "OOD_doc", "OOD_mismatch", "OOD_size", "OOD_size50", "OOD_size10",
        "OOD_external", "OOD_ext_fix", "OOD_india_fix",
        "L1_max1", "L2_max2", "L3_max3", "L5_max5",
        "B1", "B2"]


def load_rows(tag):
    # prefer the honest (commentary-cleaned) file; raw only as fallback for
    # round-7 tags (L1-L5, OOD_*_fix) which are already clean by construction.
    for suffix in ("_honest", ""):
        p = os.path.join(RD, f"{tag}{suffix}.jsonl")
        if os.path.isfile(p):
            return [json.loads(l) for l in open(p, encoding="utf-8")], suffix
    return None, None


def pure(r):
    if "pure_commentary" in r:
        return bool(r["pure_commentary"])
    return not bool(r.get("rew", ""))


def rew(r):
    return (r.get("rew_clean") or r.get("rew") or "").strip()


def lvl(tag_rows):
    lvs = sorted(set(r["lv"] for r in tag_rows))
    return {lv: [r for r in tag_rows if r["lv"] == lv] for lv in lvs}


def aggregate(sub):
    n = len(sub)
    valid = [r for r in sub if not pure(r)]
    # FRE over ALL valid (non-pure) rewrites: negative FRE is a real score
    # (output harder than standard high-school text) and must count, otherwise
    # SFT-collapse rows with FRE in [-120,0) are silently dropped. Only rows
    # whose fre is MISSING entirely are excluded.
    fre_v = [r["fre"] for r in valid if r.get("fre") is not None]
    # Failure-weighted tot: pure-commentary rows are generation failures and
    # score 0, NEVER their stored (possibly commentary-polluted) tot. _honest
    # files already carry tot=0 for pure rows (no-op here); round-7 raw files
    # (L1-L5, OOD_*_fix) store the polluted framework tot on pure rows, so we
    # must zero it ourselves to keep both sources on the SAME semantics.
    tot = [0.0 if pure(r) else r["tot"] for r in sub]
    tgt = LEVEL_TARGET_FRE.get(valid[0]["lv"] if valid else "")
    inb10 = sum(1 for f in fre_v if abs(f - tgt) <= 10) / len(fre_v) if fre_v and tgt else None
    inb12 = sum(1 for f in fre_v if abs(f - tgt) <= 12) / len(fre_v) if fre_v and tgt else None
    contra = [r["nli"].get("contra", 0.0) for r in valid if r.get("nli")]
    parts = [r.get("parts", {}) for r in valid]
    return {
        "n": n,
        "pure": sum(1 for r in sub if pure(r)),
        "pure_rate": sum(1 for r in sub if pure(r)) / n if n else 0.0,
        "stripped": sum(1 for r in sub if r.get("stripped")),
        "valid_n": len(valid),
        "fre_mean": st.mean(fre_v) if fre_v else None,
        "fre_std": st.pstdev(fre_v) if len(fre_v) > 1 else None,
        "fre_med": st.median(fre_v) if fre_v else None,
        "inband_10": inb10,
        "inband_12": inb12,
        "tot": st.mean(tot) if tot else None,
        "tot_valid": st.mean([r["tot"] for r in valid]) if valid else None,
        "diff": st.mean([p.get("diff", 0.0) for p in parts]) if parts else None,
        "term": st.mean([p.get("term", 0.0) for p in parts]) if parts else None,
        "copy": st.mean([p.get("copy", 0.0) for p in parts]) if parts else None,
        "faith": st.mean([p.get("faith", 0.0) for p in parts]) if parts else None,
        "halluc": st.mean(contra) if contra else None,
        "attempts": st.mean([r.get("attempts", 1) for r in sub]) if sub else None,
    }


def f(v, nd=3):
    return ("%%.%df" % nd) % v if v is not None else "-"


def row(tag, d, lv=None):
    a = aggregate(d)
    lv_s = LEVEL_LABEL.get(lv, "all") if lv else "all"
    out = {"tag": tag, "lv": lv_s,
           "n": a["n"], "pure": a["pure"], "pure_rate": a["pure_rate"],
           "stripped": a["stripped"], "valid_n": a["valid_n"],
           "fre_mean": a["fre_mean"], "fre_med": a["fre_med"],
           "inband_10": a["inband_10"], "inband_12": a["inband_12"],
           "tot": a["tot"], "tot_valid": a["tot_valid"],
           "diff": a["diff"], "term": a["term"], "copy": a["copy"],
           "faith": a["faith"], "halluc": a["halluc"], "attempts": a["attempts"]}
    return out, a


def table(tags, data, title, cols=None):
    print("\n## %s" % title)
    rows_out = []
    for t in tags:
        if t not in data:
            continue
        rows = data[t]["rows"]
        by = lvl(rows)
        for lv in by:
            o, _ = row(t, by[lv], lv)
            rows_out.append(o)
            print("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
                  % (t, LEVEL_LABEL.get(lv, lv), o["n"], f(o["fre_mean"], 1),
                     f(o["fre_med"], 1), f(o["inband_10"], 3), f(o["tot"], 3),
                     f(o["pure_rate"], 3), f(o["stripped"]), f(o["halluc"], 3),
                     f(o["attempts"], 2)))
        o, a = row(t, rows)
        rows_out.append(o)
        print("| **%s** | all | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
              % (t, o["n"], f(o["fre_mean"], 1), f(o["fre_med"], 1),
                 f(o["inband_10"], 3), f(o["tot"], 3), f(o["pure_rate"], 3),
                 f(o["stripped"]), f(o["halluc"], 3), f(o["attempts"], 2)))
    return rows_out


def main():
    data = {}
    missing = []
    for t in TAGS:
        rows, suffix = load_rows(t)
        if rows is None:
            missing.append(t)
            continue
        data[t] = {"rows": rows, "suffix": suffix}
    print("loaded %d/%d tags (missing: %s)" % (len(data), len(TAGS), missing), flush=True)

    out = {"tags": {}}
    for t in data:
        rows = data[t]["rows"]
        by = lvl(rows)
        per = {}
        for lv in by:
            o, _ = row(t, by[lv], lv)
            per[lv] = o
        o, a = row(t, rows)
        per["all"] = o
        out["tags"][t] = {"levels": per, "all": o,
                          "fre_valid_n": a["valid_n"]}

    print("\n=== TABLE: main chain G1-G6 ===")
    out["main_chain"] = table(["G1", "G2", "G3", "G4", "G5", "G6"], data, "Main chain (per level + all)")

    print("\n=== TABLE: ablations ===")
    out["ablations"] = table(["A_full", "A_diff", "A_term", "A_copy", "A_faith", "A_fmt",
                              "A_full_full_budget", "A_diff_full_budget"], data, "Ablations")

    print("\n=== TABLE: loop depth L1-L5 ===")
    out["loop_depth"] = table(["L1_max1", "L2_max2", "L3_max3", "L5_max5"], data, "Loop depth")

    print("\n=== TABLE: OOD ===")
    out["ood"] = table(["G6", "OOD_doc", "OOD_mismatch", "OOD_size", "OOD_size50",
                        "OOD_size10", "OOD_external", "OOD_ext_fix", "OOD_india_fix"],
                       data, "OOD (G6 reference first)")

    print("\n=== TABLE: seeds ===")
    out["seeds"] = table(["G6_pure", "G6_pure_s11", "G6_pure_s21", "G6_pure_s42",
                          "M2", "ADV", "ADV_full", "G6_full107"], data, "Seeds / purity / ADV / full-test")

    print("\n=== TABLE: baselines ===")
    out["baselines"] = table(["B1", "B2"], data, "Baselines")

    # ---- pollution quantification: raw-scored vs honest re-scored ----
    # For tags that have BOTH a raw file (stored fre/tot computed on the
    # possibly-commentary-polluted rewrite) and a _honest file, report the mean
    # shift the decontamination + full reward stack caused on the aligned rows.
    print("\n=== TABLE: commentary pollution effect (raw-stored vs honest) ===")
    print("| tag | n_aligned | Δfre (raw−honest) | Δtot (raw−honest) | Δfaith | pure% |")
    print("|---|---|---|---|---|---|")
    out["pollution"] = {}
    for t in TAGS:
        rp = os.path.join(RD, t + ".jsonl")
        hp = os.path.join(RD, t + "_honest.jsonl")
        if not (os.path.isfile(rp) and os.path.isfile(hp)):
            continue
        raw = {r["id"] + "\x00" + r["lv"]: r for r in (json.loads(l) for l in open(rp, encoding="utf-8"))}
        hon = [json.loads(l) for l in open(hp, encoding="utf-8")]
        pairs = []
        for r in hon:
            k = r["id"] + "\x00" + r["lv"]
            if k in raw:
                pairs.append((raw[k], r))
        if not pairs:
            continue
        dfre = [ra.get("fre", 0.0) - h["fre"] for ra, h in pairs]
        dtot = [ra.get("tot", 0.0) - h["tot"] for ra, h in pairs]
        dfaith = [ra.get("parts", {}).get("faith", 0.0) - h["parts"]["faith"] for ra, h in pairs]
        pure_n = sum(1 for _, h in pairs if h.get("pure_commentary"))
        out["pollution"][t] = {
            "n_aligned": len(pairs),
            "delta_fre": st.mean(dfre), "delta_tot": st.mean(dtot),
            "delta_faith": st.mean(dfaith), "pure_n": pure_n}
        print("| %s | %d | %+.2f | %+.3f | %+.3f | %d |"
              % (t, len(pairs), st.mean(dfre), st.mean(dtot), st.mean(dfaith), pure_n))

    p = os.path.join(RD, "honest_table.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nhonest_table -> %s" % p, flush=True)


if __name__ == "__main__":
    main()
