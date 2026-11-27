"""Build the course version of the site: one flat app/ folder for the Probability for AI submit form.

    python site/scripts/build_course_app.py OUT_DIR

The course uploads only app/, flattens folders (files must have unique names), and allows imports from react,
recharts, styled-components, react-icons/fa, sweetalert2, katex and relative paths. The GitHub Pages site and this
folder share the same source; only site/src/main.tsx and fonts-github.css (the Archivo font) stay out.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
SKIP = {"main.tsx", "fonts-github.css", "vite-env.d.ts"}


def rewrite(text: str) -> str:
    text = re.sub(r'from "\./components/([^"]+)"', r'from "./\1"', text)
    text = re.sub(r'from "\.\./([^"]+)"', r'from "./\1"', text)
    text = re.sub(r'from "\./data/([^"]+)"', r'from "./\1"', text)
    text = re.sub(r'from "\./assets/frames/([^"]+)"', r'from "./\1"', text)
    return text


def main(out: Path) -> None:
    app = out / "app"
    if app.exists():
        shutil.rmtree(app)
    app.mkdir(parents=True)
    seen: set[str] = set()
    for f in sorted(SRC.rglob("*")):
        if not f.is_file() or f.name in SKIP:
            continue
        if f.name in seen:
            raise SystemExit(f"two files named {f.name}: the course needs unique names")
        seen.add(f.name)
        if f.suffix in {".ts", ".tsx"}:
            (app / f.name).write_text(rewrite(f.read_text()))
        else:
            shutil.copy2(f, app / f.name)
    print(f"wrote {len(seen)} files to {app}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
