import json
from pathlib import Path

briefs = Path("artifacts/caa/UNI_CAA01_corpus/curator_briefs")
fixes = {
    "p02_lru_cache": ("[1, None]", "demo() returns [1, None] after LRU eviction."),
    "p16_event_snapshot": ("25", "demo() returns final balance 25 after events."),
    "p21_dijkstra": ("4", "demo() returns shortest path length 4 to node 2."),
    "p63_heap_kth": ("5", "demo() returns 2nd smallest = 5 for [7,3,5]."),
}
for pid, (expect, success) in fixes.items():
    short = pid.split("_")[0]
    for p in briefs.glob(f"unicaa01_{short}_panic__*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("pair_id") != pid:
            continue
        d["demo_oracle"] = {"pair_id": pid, "check_mode": "eq", "expect_repr": expect}
        d["success_criteria"] = success
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print("updated", p.name)
