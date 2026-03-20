"""Replace em/en dashes and 4 digit year ranges in ui/app.py (blog + docstrings)."""
from __future__ import annotations

import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "ui" / "app.py"
t = p.read_text(encoding="utf-8")
t = t.replace("\u2014", ", ").replace("\u2013", " to ")
t = re.sub(r"(\d{4})-(\d{4})", r"\1 to \2", t)
t = re.sub(r",\s{2,}", ", ", t)
p.write_text(t, encoding="utf-8")
print("Updated", p)
