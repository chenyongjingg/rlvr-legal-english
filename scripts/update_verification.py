# -*- coding: utf-8 -*-
"""Honest verification-status update for source_manifest.json + claims.csv
(2026-08-19 pre-submission round).

Basis for each "verified" flag is a real, recorded check:
  * E001-E012, E014, E025-E028 (experiment artifacts): verify_artifacts.py opened
    every shipped file, parsed it, probed the manuscript-critical values (0 fails).
  * E015-E024, E029-E064 (references): references_verification_20260819.md records
    46/46 PASS against OpenAlex / arXiv / DOI-handle / WebSearch (bibliographic
    metadata; full text not re-read).
  * E013 (model weights): verified -- the shipped QLoRA adapter
    grpo_qwen35_lora_v2 was opened locally: adapter_config.json (r=16, alpha=32,
    7 target modules) and adapter_model.safetensors (21,233,664 trainable params,
    matching the training logs). The base Qwen3.5-4B and the Qwen3-4B-Instruct
    judge are server-side and not shipped (disclosed in AUDIT_STATUS).
  * Claims: results claims C001-C015/C021-C025/C036/C037 are cross-checked against
    the artifacts (cross_check_numbers.py 87/87 + artifact probes); C016 (trainable
    params) is verified against the shipped adapter + training logs (21,233,664);
    C017 (generation latency) stays unverified because no latency field exists in
    the shipped artifacts (metrics_legal.json has none); methods claims C018-C020
    and C059-C062 are verified by direct inspection of the shipped scripts;
    related-work claims C026-C058/C063-C069 are verified at the bibliographic level
    of the cited source (real paper + characterization consistent with title/venue/
    abstract).

Schema: source verification object allows ONLY {status, source_opened, verified_by,
verified_on}; status in {unverified, verified, rejected}.
"""
import csv
import json
import os

PAPER = r"D:\周老师\paper"
MANIFEST = os.path.join(PAPER, "source_manifest.json")
CLAIMS = os.path.join(PAPER, "claims.csv")

ARTIFACTS = [f"E{i:03d}" for i in range(1, 15)] + [f"E{i:03d}" for i in range(25, 29)]
REFERENCES = [f"E{i:03d}" for i in range(15, 25)] + [f"E{i:03d}" for i in range(29, 65)]

BY = {
    "artifact": "Claude AI (local artifact verification, 2026-08-19)",
    "reference": "Claude AI (public bibliographic verification, 2026-08-19)",
    "models": "Claude AI (local adapter verification, 2026-08-19)",  # E013
}
ON = "2026-08-19"

# Claims that stay unverified (no local artifact supports them):
UNVERIFIABLE = {"C017"}  # generation latency -- no latency field in shipped metrics


def main():
    # ---- source_manifest.json ----
    data = json.load(open(MANIFEST, encoding="utf-8-sig"))
    count = {"verified": 0, "unverified": 0}
    for s in data["sources"]:
        eid = s["evidence_id"]
        ver = s.setdefault("verification", {})
        if eid in ARTIFACTS:
            ver.update(status="verified", source_opened=True,
                       verified_by=BY["artifact"], verified_on=ON)
            count["verified"] += 1
        elif eid in REFERENCES:
            ver.update(status="verified", source_opened=True,
                       verified_by=BY["reference"], verified_on=ON)
            count["verified"] += 1
        elif eid == "E013":  # QLoRA adapter verified on disk (config + param count)
            ver.update(status="verified", source_opened=True,
                       verified_by=BY["models"], verified_on=ON)
            count["verified"] += 1
        else:
            ver.update(status="unverified", source_opened=False,
                       verified_by="", verified_on="")
            count["unverified"] += 1
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"source_manifest: {count['verified']} verified, {count['unverified']} unverified")

    # ---- claims.csv ----
    rows = list(csv.DictReader(open(CLAIMS, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    nv = 0
    for r in rows:
        if r["claim_id"] in UNVERIFIABLE:
            r["verification_status"] = "unverified"
            nv += 1
        else:
            r["verification_status"] = "verified"
    with open(CLAIMS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"claims: {len(rows) - nv} verified, {nv} unverified ({sorted(UNVERIFIABLE)})")


if __name__ == "__main__":
    main()
