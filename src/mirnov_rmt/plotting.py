"""Plotting and reporting helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt


def savefig(path: Path) -> None:
    """Save current matplotlib figure with consistent options."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_report(
    output: Path,
    title: str,
    sections: list[tuple[str, str, list[str]]],
) -> None:
    """Write a compact local HTML report.

    Each section is ``(heading, html_body, image_paths)``. Image paths should be
    relative to the report file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.5;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#172033}",
        "h1,h2{line-height:1.2}",
        "code{background:#eef2f7;padding:.1rem .25rem;border-radius:4px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}",
        "img{max-width:100%;border:1px solid #d7dee8;border-radius:6px}",
        "table{border-collapse:collapse;width:100%;margin:1rem 0}",
        "td,th{border:1px solid #d7dee8;padding:.45rem;text-align:left}",
        "</style>",
        "</head><body>",
        f"<h1>{escape(title)}</h1>",
    ]
    for heading, body, images in sections:
        parts.append(f"<h2>{escape(heading)}</h2>")
        parts.append(body)
        if images:
            parts.append('<div class="grid">')
            for image in images:
                parts.append(f'<a href="{escape(image)}"><img src="{escape(image)}" alt="{escape(image)}"></a>')
            parts.append("</div>")
    parts.append("</body></html>")
    output.write_text("\n".join(parts), encoding="utf-8")
