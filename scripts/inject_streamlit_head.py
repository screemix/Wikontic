"""Inject early head CSS/JS into Streamlit's index.html for consistent branding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit_ui import ACCENT_COLOR  # noqa: E402

MARKER = "<!-- wikontic-head-inject -->"


def build_injection(color: str) -> str:
    return f"""{MARKER}
<style id="wikontic-decoration-style">
[data-testid="stDecoration"],
#stDecoration,
.stDecoration,
[data-testid="stHeader"] [data-testid="stDecoration"] {{
  background: {color} !important;
  background-color: {color} !important;
  background-image: none !important;
}}
</style>
<script id="wikontic-decoration-script">
(function () {{
  var color = "{color}";
  var selector = '[data-testid="stDecoration"], #stDecoration, .stDecoration';
  var watched = new WeakSet();

  function applyTo(el) {{
    el.style.setProperty("background", color, "important");
    el.style.setProperty("background-color", color, "important");
    el.style.setProperty("background-image", "none", "important");
  }}

  function watchElement(el) {{
    if (watched.has(el)) {{
      return;
    }}
    watched.add(el);
    applyTo(el);
    new MutationObserver(function () {{
      applyTo(el);
    }}).observe(el, {{
      attributes: true,
      attributeFilter: ["style", "class"],
    }});
  }}

  function applyDecorationStyle() {{
    document.querySelectorAll(selector).forEach(watchElement);
  }}

  applyDecorationStyle();

  new MutationObserver(applyDecorationStyle).observe(document.documentElement, {{
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["style", "class"],
  }});

  function loop() {{
    applyDecorationStyle();
    requestAnimationFrame(loop);
  }}
  requestAnimationFrame(loop);
}})();
</script>
"""


def find_streamlit_index_html() -> Path:
    try:
        import streamlit
    except ImportError as exc:
        raise SystemExit("streamlit is not installed") from exc

    return Path(streamlit.__file__).resolve().parent / "static" / "index.html"


def inject(index_path: Path, color: str = ACCENT_COLOR) -> None:
    html = index_path.read_text(encoding="utf-8")
    injection = build_injection(color)

    if MARKER in html:
        start = html.index(MARKER)
        end = html.find("</head>", start)
        if end == -1:
            raise SystemExit(f"Could not find </head> after marker in {index_path}")
        html = html[:start] + injection + html[end:]
    elif "</head>" in html:
        html = html.replace("</head>", injection + "</head>", 1)
    else:
        raise SystemExit(f"Could not find </head> in {index_path}")

    index_path.write_text(html, encoding="utf-8")
    print(f"Injected Wikontic head styles into {index_path}")


def main() -> None:
    inject(find_streamlit_index_html())


if __name__ == "__main__":
    main()
