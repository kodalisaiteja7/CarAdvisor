"""One-off: replace dashes in ui/app.py user-facing strings (blog HTML, docstrings)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "ui" / "app.py"
t = p.read_text(encoding="utf-8")
t = t.replace("\u2014", ", ").replace("\u2013", " to ")
t = re.sub(r"(\d{4})-(\d{4})", r"\1 to \2", t)
# compound words letter-letter (13th-gen -> 13th gen); limit iterations
for _ in range(20):
    n = t
    t = re.sub(r"([A-Za-z0-9])-([A-Za-z])", r"\1 \2", t)
    if n == t:
        break
p.write_text(t, encoding="utf-8")
print("Updated", p)
