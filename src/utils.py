"""
utils.py
--------
Shared plotting helpers and style settings used across eda.py and modeling.py.
All charts use a dark theme that renders well in Jupyter and when saved to PNG.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Global dark theme ─────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"
ACCENT     = "#58a6ff"

# Sentiment palette — aligned to emotional heat (blue → red)
SENTIMENT_PALETTE = {
    "Extreme Fear": "#ef5675",
    "Fear":         "#ff764a",
    "Neutral":      "#ffa600",
    "Greed":        "#7adb78",
    "Extreme Greed":"#2dc9a7",
}

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def apply_dark_style() -> None:
    """Apply a consistent dark-mode style to all subsequent matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "figure.dpi":        130,
    })


def save_fig(fig: plt.Figure, filename: str, output_dir: str = "outputs/figures") -> None:
    """
    Save a figure to *output_dir/filename* with tight layout.
    Creates the directory if it does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Saved → {path}")
    plt.close(fig)


def sentiment_colors(labels: list[str]) -> list[str]:
    """Return a list of hex color strings corresponding to sentiment labels."""
    return [SENTIMENT_PALETTE.get(lbl, ACCENT) for lbl in labels]
