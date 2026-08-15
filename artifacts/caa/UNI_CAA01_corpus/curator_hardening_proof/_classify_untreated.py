"""Classify UNI_CAA01 pairs by panic / brief / calm-tail readiness."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "research"))

from unicaa01_demo_oracles import ORACLES  # noqa: E402

CORPUS = ROOT / "artifacts/caa/UNI_CAA01_corpus"
BRIEFS = CORPUS / "curator_briefs"
TAILS = CORPUS / "calm_tails"
EXP = ROOT / "experiments/unicaa01_corpus/out"
PACKS = CORPUS / "curation_packs"


def main() -> None:
    all_pairs = sorted(ORACLES)

    briefs_by_pair: dict[str, list[dict]] = defaultdict(list)
    for path in BRIEFS.glob("unicaa01_*.json"):
        brief = json.loads(path.read_text(encoding="utf-8"))
        briefs_by_pair[str(brief["pair_id"])].append(brief)

    status_by_pair: dict[str, list[dict]] = defaultdict(list)
    for txt in TAILS.glob("unicaa01_*.txt"):
        meta = json.loads(txt.with_suffix(".meta.json").read_text(encoding="utf-8"))
        brief_id = str(meta.get("brief_id") or txt.stem)
        brief_path = BRIEFS / f"{brief_id}.json"
        pair = (
            json.loads(brief_path.read_text(encoding="utf-8"))["pair_id"]
            if brief_path.is_file()
            else "?"
        )
        status_by_pair[pair].append(
            {"brief_id": brief_id, "status": meta.get("status"), "file": txt.name}
        )

    packs_by_pair: dict[str, list[str]] = defaultdict(list)
    for path in PACKS.glob("*panic*.json"):
        if path.name == "index.json":
            continue
        try:
            pack = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pair = str(pack.get("pair_id") or "")
        if pair:
            packs_by_pair[pair].append(path.name)

    rows = []
    for pair in all_pairs:
        # Prefer exact pair_id in filename.
        cands = sorted(EXP.glob(f"*{pair}*panic*latest.json"))
        inv_n = None
        confs: list[str] = []
        exp_name = None
        if cands:
            exp_name = cands[0].name
            data = json.loads(cands[0].read_text(encoding="utf-8"))
            inv = data.get("invocations") or []
            inv_n = len(inv) if isinstance(inv, list) else int(data.get("n_invocations") or 0)
            for item in inv if isinstance(inv, list) else []:
                tracer = item.get("tracer") or {}
                conf = tracer.get("confidence") or item.get("tracer_confidence")
                if conf:
                    confs.append(str(conf))

        briefs = briefs_by_pair.get(pair, [])
        statuses = status_by_pair.get(pair, [])
        admitted = [s for s in statuses if s["status"] == "admitted"]
        quar = [s for s in statuses if s["status"] == "quarantined"]
        used_ids = {s["brief_id"] for s in statuses}
        unused = [b["brief_id"] for b in briefs if b["brief_id"] not in used_ids]
        packs = packs_by_pair.get(pair, [])

        if admitted:
            cat = "READY_ADMITTED"
        elif quar and not admitted:
            cat = "QUARANTINED_ONLY"
        elif briefs and not statuses:
            cat = "HAS_BRIEF_NO_TAIL"
        elif packs and not briefs:
            cat = "HAS_PACK_NO_BRIEF"
        elif (inv_n or 0) > 0:
            cat = "HAS_PANIC_INVOCATIONS_NO_PACK"
        else:
            cat = "NO_USABLE_PANIC_INVOCATIONS"

        rows.append(
            {
                "cat": cat,
                "pair": pair,
                "inv_n": inv_n,
                "confs": sorted(set(confs)),
                "n_briefs": len(briefs),
                "n_admitted": len(admitted),
                "n_quarantined": len(quar),
                "n_unused_briefs": len(unused),
                "unused_briefs": unused,
                "n_packs": len(packs),
                "exp": exp_name,
            }
        )

    print("CATS", dict(Counter(r["cat"] for r in rows)))
    print()
    for cat in (
        "HAS_BRIEF_NO_TAIL",
        "HAS_PACK_NO_BRIEF",
        "HAS_PANIC_INVOCATIONS_NO_PACK",
        "QUARANTINED_ONLY",
        "NO_USABLE_PANIC_INVOCATIONS",
        "READY_ADMITTED",
    ):
        subset = [r for r in rows if r["cat"] == cat]
        if not subset:
            continue
        print(f"=== {cat} ({len(subset)}) ===")
        for r in subset:
            print(
                f"  {r['pair']:22} inv={r['inv_n']} packs={r['n_packs']} "
                f"briefs={r['n_briefs']} admitted={r['n_admitted']} "
                f"quar={r['n_quarantined']} unused={r['n_unused_briefs']} "
                f"conf={r['confs']} exp={r['exp']}"
            )
            for ub in r["unused_briefs"]:
                bp = BRIEFS / f"{ub}.json"
                pref = json.loads(bp.read_text(encoding="utf-8"))["prefix_lock"]["prefix_text"]
                print(f"    unused {ub}")
                print(f"      pref={pref[:110]!r}")
        print()

    out = CORPUS / "curator_hardening_proof" / "G26_untreated_classification.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
