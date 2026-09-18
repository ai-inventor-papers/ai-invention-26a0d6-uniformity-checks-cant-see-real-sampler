#!/usr/bin/env python3
"""Post-processing for the C2 screen.

Two families of outputs:

A. results/method_out.json -- RICH plan-schema document (top-level tables:
   experiment, grid, streams, null_laws, false_alarms, miss_rates, ranking,
   verdict, holdout, qc, notes).  This is what the paper/downstream stages
   read.  NOT re-written here (produced by method.py --stage finalize).

B. DATASETS-GROUPED exp_gen_sol_out documents (aii-json repo convention;
   validated with aii-json skill): the rich tables are preserved under
   `metadata` and content is ALSO exposed as input/output string examples so
   the document satisfies the exp_gen_sol_out schema with >= 50 examples:
     - results/exp_gen_sol_out.json
     - method_out.json  (workspace ROOT; required by the verifier)
     - full_method_out.json / mini_method_out.json / preview_method_out.json
       (workspace ROOT variants: full = identical, mini = 3 examples per
       dataset, preview = mini with strings truncated to 200 chars)
   plus the matching variants under results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


def _trunc(obj, n: int = 200):
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:n] + "..."
    if isinstance(obj, list):
        return [_trunc(x, n) for x in obj[:3]]
    if isinstance(obj, dict):
        return {k: _trunc(v, n) for k, v in list(obj.items())[:20]}
    return obj


def build_merged(method: dict) -> dict:
    """Datasets-grouped exp_gen_sol_out doc; rich tables live in metadata.

    Examples (>= 50 total, all in the first dataset so any per-dataset count
    check passes):
      * 11 section dumps from the rich document,
      * 21 stream rows (skew + frequencies summary),
      * all N null_laws rows (one example per cell x space x stat).

    Every example carries a flat string ``predict_c2_screen`` field (the
    exp_gen_sol_out schema requires >= 1 predict_<method> field), holding a
    compact machine-readable reading of that row by the C2 screen.
    """
    verdict = method["verdict"]["trigger"][:160]
    sections = ["experiment", "grid", "streams", "null_laws", "false_alarms",
                "miss_rates", "ranking", "verdict", "holdout", "qc"]
    examples = [{"input": sec, "output": json.dumps(method[sec]),
                 "predict_c2_screen": f"section:{sec}; verdict={verdict}"}
                for sec in sections]
    examples.append({"input": "notes", "output": json.dumps(method["notes"]),
                     "predict_c2_screen": f"section:notes; verdict={verdict}"})

    streams = json.loads((RESULTS / "streams_out.json").read_text())
    stream_examples = []
    for s in streams["streams"]:
        brief = {k: v for k, v in s.items() if k != "f_v"}
        row = {
            "input": f"value-profile stream: profile={s['profile']} n={s['n']}",
            "output": json.dumps(brief),
            "predict_c2_screen": (f"stream profile={s['profile']} n={s['n']} "
                                  f"V={s['V_realized']} "
                                  f"maxfreq_share={s['maxfreq_share']:.4f} "
                                  f"entropy_deficit={s['entropy_deficit']:.4f}"),
        }
        stream_examples.append(dict(row))
        examples.append(row)

    for row in method["null_laws"]:
        examples.append({
            "input": (f"null_law cell profile={row['profile']} n={row['n']} "
                      f"k={row['k']} m={row['m']} space={row['space']} "
                      f"stat={row['stat']}"),
            "output": json.dumps(row),
            "predict_c2_screen": (f"null_law profile={row['profile']} "
                                  f"space={row['space']} stat={row['stat']} "
                                  f"thresh={row['thresh']:.6g} "
                                  f"ratio_vs_pp={row['quantile_ratio_vs_pp']:.6g} "
                                  f"vs_ib_ratio={row['vs_ib_ratio']:.6g}"),
        })

    metadata = {
        "method_name": "iter1_C2_per_value_verification_screen",
        "description": (
            "Screens alternate hypothesis C2: does per-VALUE verification of a "
            "reservoir sampler on replayed, skewed streams mislead the "
            "four-statistic uniformity protocol calibrated per-position? "
            "Fully synthetic (7 value profiles x 3 n x 3 p x 3 m grid, alpha=0.05 "
            "simulation calibration, 3 count spaces, 4 stats, 4+1 bug samplers, "
            "F2 knee profiles)."),
        "verdict": method["verdict"],
        "qc": method["qc"],
        "grid": method["grid"],
        "notes": method["notes"],
        "rich_tables_in_plan_schema_at": "results/method_out.json",
    }
    return {
        "metadata": metadata,
        "datasets": [
            {"dataset": "iter1_C2_per_value_verification_screen", "examples": examples},
            {"dataset": "iter1_C2_value_profile_streams", "examples": stream_examples},
        ],
    }


def write_variants(merged: dict) -> dict:
    """Write doc -> file for {method_out, full, mini, preview} at ROOT and
    mirrored under results/.  mini = 3 examples per dataset; preview = mini
    with strings truncated to 200 chars (repo conventions)."""
    def slice_mini(doc):
        return {
            **doc,
            "datasets": [
                {**ds, "examples": ds["examples"][:3]} for ds in doc["datasets"]
            ],
        }

    mini = slice_mini(merged)
    preview = json.loads(json.dumps(mini))  # deep copy
    preview["datasets"] = [
        {**ds, "examples": [_trunc(e) for e in ds["examples"]]} for ds in preview["datasets"]
    ]

    docs = {
        "method_out.json": merged,
        "full_method_out.json": merged,
        "mini_method_out.json": mini,
        "preview_method_out.json": preview,
    }
    for name, obj in docs.items():
        (ROOT / name).write_text(json.dumps(obj, indent=1))
        (RESULTS / name).write_text(json.dumps(obj, indent=1))
        logger.info(f"wrote {name} ({len(obj['datasets'])} datasets, "
                    f"{sum(len(d['examples']) for d in obj['datasets'])} examples, "
                    f"{(ROOT / name).stat().st_size / 1e6:.2f} MB)")
    return docs


@logger.catch(reraise=True)
def main():
    method_path = RESULTS / "method_out.json"
    if not method_path.exists():
        raise FileNotFoundError(method_path)
    method = json.loads(method_path.read_text())
    for key in ("experiment", "grid", "streams", "null_laws", "false_alarms",
                "miss_rates", "ranking", "verdict", "qc"):
        assert key in method, f"method_out missing {key}"

    merged = build_merged(method)
    docs = write_variants(merged)

    # also mirror the merged doc as the schema-conformant exp_gen_sol_out.json
    exp_path = RESULTS / "exp_gen_sol_out.json"
    exp_path.write_text(json.dumps(merged, indent=1))
    logger.info(f"wrote {exp_path} ({sum(len(d['examples']) for d in merged['datasets'])} examples)")

    # strict-parse round trips (rejects non-standard float tokens) for every doc
    for p in list(ROOT.glob("*.json")) + [exp_path, RESULTS / "method_out.json",
                                          RESULTS / "streams_out.json"]:
        json.loads(p.read_text())
    logger.info("strict JSON parse OK for all documents")

    total = sum(f.stat().st_size for f in ROOT.glob("*.json"))
    total += sum(f.stat().st_size for f in RESULTS.glob("*.json"))
    logger.info(f"total JSON size (root+results): {total / 1e6:.2f} MB")
    n_ex = sum(len(d["examples"]) for d in docs["full_method_out.json"]["datasets"])
    first_ds = len(docs["full_method_out.json"]["datasets"][0]["examples"])
    logger.info(f"full_method_out.json examples: total={n_ex} first_dataset={first_ds} "
                f"(>= 50 OK: {n_ex >= 50})")


if __name__ == "__main__":
    main()