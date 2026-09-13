from pathlib import Path

p = Path("scripts/apply_feature_norm_phase2.py")
text = p.read_text()
marker = "# CI gate for the first-class packaged integration.\n"
if text.count(marker) != 1:
    raise SystemExit(f"expected one CI patch marker, found {text.count(marker)}")
p.write_text(text.split(marker, 1)[0])
