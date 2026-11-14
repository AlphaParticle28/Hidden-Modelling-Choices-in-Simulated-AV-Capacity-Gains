"""Shared figure style, so every figure in the project looks the same."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical colours, always used in this order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def save(fig, path, dpi=150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
