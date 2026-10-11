"""The dashboard pages are single HTML files with inline JavaScript; one stray
quote stops the whole page. Syntax-check the scripts when Node.js is available."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "kalshi_bot" / "web" / "static"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("page", ["index.html", "login.html"])
def test_page_script_parses(tmp_path, page):
    js = re.search(r"<script>(.*)</script>", (STATIC / page).read_text(encoding="utf-8"), re.S).group(1)
    f = tmp_path / "page.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run([shutil.which("node"), "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:600]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
@pytest.mark.parametrize("dpr", [1, 1.25, 1.5, 2, 3])
def test_chart_canvas_keeps_its_size_on_high_dpi_screens(tmp_path, dpr):
    """Sizing a canvas for a sharp screen rewrites its height attribute. The chart
    must keep its 240 px CSS height however often it is redrawn (it used to grow
    on every redraw whenever the pixel ratio was not exactly 1)."""
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    fn = re.search(r"function setupCanvas\(c\)\{.*?return \[g,w,h\];\}", page, re.S).group(0)
    f = tmp_path / "canvas.js"
    f.write_text(f"""
        globalThis.window = {{devicePixelRatio: {dpr}}};
        const attrs = {{height: '240'}};
        const c = {{clientWidth: 701, dataset: {{}}, style: {{}}, _w: 300,
                   getAttribute: k => attrs[k],
                   getContext: () => ({{setTransform() {{}}, clearRect() {{}}}}),
                   get height() {{ return +attrs.height; }}, set height(v) {{ attrs.height = String(Math.trunc(v)); }},
                   get width() {{ return this._w; }}, set width(v) {{ this._w = Math.trunc(v); }}}};
        {fn}
        let out;
        for (let i = 0; i < 6; i++) out = setupCanvas(c);
        console.log(JSON.stringify([c.width, c.height, c.style.height, out[1], out[2]]));
        """, encoding="utf-8")
    r = subprocess.run([shutil.which("node"), str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:600]
    assert json.loads(r.stdout) == [int(701 * dpr + 0.5), int(240 * dpr + 0.5), "240px", 701, 240]
