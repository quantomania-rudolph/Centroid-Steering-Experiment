from pathlib import Path
ROOT = Path(r"c:\Users\Rudol\Desktop\ThetaData_greeks\J-Do It")
for rel in [
    "src/jlens_qwen/caa/tension_audit.py",
    "research/prove_cast_tension.py",
    "tests/unit/test_tension_audit.py",
]:
    p = ROOT / rel
    if not p.is_file():
        continue
    raw = p.read_bytes()
    text = raw.decode("utf-16") if len(raw) > 1 and raw[1] == 0 else raw.decode("utf-8", "replace").replace("\x00", "")
    p.write_text(text, encoding="utf-8", newline="\n")
    print("fixed", p)