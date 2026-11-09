"""Generate the README banner (dark/light), the animated protocol figure and the stat badges.

Everything is drawn from scratch with matplotlib/PIL — no radiograph pixels are used anywhere, so
the outputs can be published with the repository.  Run from the repository root:

    python scripts/make_readme_assets.py
"""
import html
import io
import os
import time
import urllib.request
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Ellipse, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
BADGES = os.path.join(ROOT, "docs", "badges")
os.makedirs(FIG, exist_ok=True)
os.makedirs(BADGES, exist_ok=True)


@dataclass
class Theme:
    bg_top: str
    bg_bot: str
    title: str
    sub: str
    accent: str
    film: str
    film_edge: str
    lung: str
    lung_edge: str
    bone: str
    text: str
    fill: str = "#8d939c"  # "mean fill" grey used for the occlusion
    ev: str = "#e74c3c"  # evidence box
    rd: str = "#9aa5b8"  # size-matched control box
    lesion: str = "#f3d8b6"


DARK = Theme("#0a1020", "#182c52", "#ffffff", "#a9bbe0", "#7fd1b9", "#1b2640", "#44608f", "#2d3f66", "#4b6395", "#5a74a8", "#e3eaf7")
LIGHT = Theme("#f8fafd", "#e2e9f4", "#0d1b33", "#3e4f70", "#0f8a6c", "#dde5f0", "#9fb0cc", "#c6d2e6", "#9fb0cc", "#a9b8d2", "#1c2a44")


def draw_film(ax, x, y, w, h, T, mask_ev=False, mask_rd=False, show_ev=True, show_rd=True, labels=None):
    """Abstract chest radiograph: film, spine, ribs, lungs, heart, a small lesion, the two boxes."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={0.05 * w}", fc=T.film, ec=T.film_edge, lw=1.6, zorder=1))
    ax.add_patch(Rectangle((x + 0.46 * w, y + 0.08 * h), 0.08 * w, 0.86 * h, fc=T.bone, ec="none", alpha=0.55, zorder=2))
    for i in range(6):  # posterior ribs as faint arches
        yy = y + (0.80 - 0.11 * i) * h
        for cx in (x + 0.29 * w, x + 0.71 * w):
            ax.add_patch(Arc((cx, yy), 0.40 * w, 0.14 * h, theta1=0, theta2=180, color=T.bone, alpha=0.45, lw=1.0, zorder=3))
    ax.add_patch(Ellipse((x + 0.30 * w, y + 0.53 * h), 0.33 * w, 0.66 * h, angle=6, fc=T.lung, ec=T.lung_edge, lw=1, alpha=0.92, zorder=4))
    ax.add_patch(Ellipse((x + 0.70 * w, y + 0.53 * h), 0.33 * w, 0.66 * h, angle=-6, fc=T.lung, ec=T.lung_edge, lw=1, alpha=0.92, zorder=4))
    ax.add_patch(Ellipse((x + 0.57 * w, y + 0.36 * h), 0.30 * w, 0.30 * h, fc=T.film, ec=T.film_edge, lw=1, alpha=0.95, zorder=5))
    bw, bh = 0.13 * w, 0.12 * h
    ex, ey = x + 0.64 * w, y + 0.60 * h  # evidence box (right lung, upper zone)
    rx, ry = x + 0.17 * w, y + 0.24 * h  # size-matched control (left lung, lower zone)
    ax.add_patch(Ellipse((ex + bw / 2, ey + bh / 2), 0.055 * w, 0.05 * h, fc=T.lesion, ec="none", alpha=0.95, zorder=5.5))  # the finding
    if show_ev:
        ax.add_patch(Rectangle((ex, ey), bw, bh, fc=T.fill if mask_ev else "none", ec=T.ev, lw=2.2, zorder=6))
    if show_rd:
        ax.add_patch(Rectangle((rx, ry), bw, bh, fc=T.fill if mask_rd else "none", ec=T.rd, lw=1.8, ls=(0, (3, 2)), zorder=6))
    if labels:
        lx, dx = labels  # x position of the label column and arrow start offset
        ax.annotate("evidence box\n(radiologist consensus)", xy=(ex + bw, ey + bh / 2), xytext=(lx, ey + bh / 2 + 0.05 * h), color=T.text, fontsize=10.5, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=T.text, lw=0.8, alpha=0.8))
        ax.annotate("size-matched control\n(avoids every finding)", xy=(rx + bw, ry + bh / 2), xytext=(lx, ry + bh / 2 - 0.02 * h), color=T.text, fontsize=10.5, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=T.text, lw=0.8, alpha=0.8))
    return (ex, ey, rx, ry, bw, bh)


def banner(T, path):
    W, H = 16.0, 4.4
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("g", [T.bg_top, T.bg_bot])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), extent=[0, W, 0, H], aspect="auto", cmap=cmap, zorder=0, origin="upper")
    grid_c = "#ffffff" if T is DARK else "#0d1b33"
    for gx in np.arange(0.5, W, 0.5):
        ax.plot([gx, gx], [0, H], color=grid_c, alpha=0.035, lw=0.8, zorder=0.5)
    for gy in np.arange(0.5, H, 0.5):
        ax.plot([0, W], [gy, gy], color=grid_c, alpha=0.035, lw=0.8, zorder=0.5)
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, W, 90), rng.uniform(0, H, 90), s=rng.uniform(2, 9, 90), color=grid_c, alpha=0.08, zorder=0.6, lw=0)
    ax.text(0.75, 2.92, "VinDr-Ground-VQA", fontsize=47, fontweight="bold", color=T.title, va="center", zorder=5)
    ax.plot([0.78, 4.9], [2.36, 2.36], color=T.accent, lw=3.5, solid_capstyle="round", zorder=5)
    ax.text(0.78, 1.92, "Grounded chest X-ray VQA with an area-matched occlusion control", fontsize=19.5, color=T.sub, va="center", zorder=5)
    ax.text(0.78, 1.30, "102,072 questions   •   8,434 radiographs   •   22 findings   •   lesion-level evidence boxes   •   9 models",
            fontsize=12.8, color=T.accent, va="center", zorder=5)
    ax.text(0.78, 0.78, "ML4H 2026   ·   labels  +  evaluation harness  +  reference results   ·   does the model use the marked evidence?",
            fontsize=12, color=T.sub, va="center", alpha=0.9, zorder=5)
    draw_film(ax, 11.35, 0.45, 2.6, 3.5, T, labels=(14.1, 0))
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------
# animated protocol figure
STEPS = [
    ("①  Ask the yes/no question on the full image", "m_full"),
    ("②  Mask the radiologist's evidence box (mean fill)  →  drop  M_roi", "m_roi"),
    ("③  Mask a size-matched random region that avoids every finding  →  drop  M_rand", "m_rand"),
    ("④  excess  =  M_roi − M_rand   —   the evidence-specific part of the reliance", "excess"),
    ("⑤  grounding fraction  =  excess / (m_full − m_blind)   —   0 for a blind reader, 1 at the ceiling", "gf"),
]
VALS = {"full image": 2.4, "evidence masked": 0.6, "control masked": 2.1, "text only (blind)": 0.4}
COL = {"full image": "#2c3e50", "evidence masked": "#c0392b", "control masked": "#7f8c8d", "text only (blind)": "#b8c1cc"}


def frame(step):
    T = LIGHT
    fig = plt.figure(figsize=(10, 4.7), dpi=110, facecolor="white")
    fig.text(0.03, 0.945, STEPS[step][0], fontsize=12.6, fontweight="bold", color="#1c2a44", va="center")
    fig.text(0.03, 0.885, "Q: “Is there a nodule in the right upper lung?”      model’s first-token log-odds for “Yes”, in nats  (illustrative values)",
             fontsize=9.6, color="#5b6b85", va="center")
    axL = fig.add_axes([0.03, 0.04, 0.33, 0.80])
    axL.set_xlim(0, 3.3)
    axL.set_ylim(0, 4.0)
    axL.axis("off")
    draw_film(axL, 0.15, 0.1, 2.9, 3.8, T, mask_ev=(step == 1), mask_rd=(step == 2), show_ev=step >= 1 and step != 2, show_rd=step >= 2)
    if step == 1:
        axL.text(1.6, 0.0, "evidence box masked", color=T.ev, fontsize=9.5, ha="center", va="top", fontweight="bold")
    if step == 2:
        axL.text(1.6, 0.0, "size-matched control masked", color="#5d6b80", fontsize=9.5, ha="center", va="top", fontweight="bold")
    axR = fig.add_axes([0.47, 0.17, 0.50, 0.63])
    names = ["full image", "evidence masked", "control masked", "text only (blind)"]
    shown = {0: 1, 1: 2, 2: 3, 3: 3, 4: 4}[step]
    ypos = {n: 3 - i for i, n in enumerate(names)}
    for n in names[:shown]:
        axR.barh(ypos[n], VALS[n], color=COL[n], height=0.58, zorder=3)
        axR.text(VALS[n] + 0.05, ypos[n], f"{VALS[n]:.1f}", va="center", fontsize=10, color="#1c2a44", zorder=4)
    axR.set_yticks([ypos[n] for n in names])
    axR.set_yticklabels([n if n in names[:shown] else "" for n in names], fontsize=10)
    axR.set_xlim(0, 3.3)
    axR.set_ylim(-0.7, 3.7)
    axR.set_xlabel("log-odds margin for “Yes”  (nats)", fontsize=10, color="#1c2a44")
    axR.grid(axis="x", color="#e6eaf0", zorder=0)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.tick_params(colors="#1c2a44", labelsize=9.5)

    def drop(n, label, color):
        y = ypos[n]
        axR.annotate("", xy=(VALS[n], y + 0.38), xytext=(VALS["full image"], y + 0.38), arrowprops=dict(arrowstyle="<->", color=color, lw=1.4), zorder=5)
        axR.text((VALS[n] + VALS["full image"]) / 2, y + 0.47, label, ha="center", va="bottom", fontsize=9.5, color=color, fontweight="bold", zorder=5)

    if step >= 1:
        drop("evidence masked", "M_roi = 1.8", "#c0392b")
    if step >= 2:
        drop("control masked", "M_rand = 0.3", "#5d6b80")
    if step == 3:
        axR.text(1.95, 0.0, "excess = 1.8 − 0.3 = 1.5 nats", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0f8a6c",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#e8f6f1", ec="#0f8a6c", lw=1.2), zorder=6)
    if step == 4:
        axR.text(1.95, 0.0, "GF = 1.5 / (2.4 − 0.4) = 0.75", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0f8a6c",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#e8f6f1", ec="#0f8a6c", lw=1.2), zorder=6)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return im


def gif(path, sheet_path):
    frames = [frame(s) for s in range(len(STEPS))]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=[1700, 1900, 1900, 2400, 2600], loop=0, disposal=2)
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * len(frames)), "white")
    for i, f in enumerate(frames):
        sheet.paste(f, (0, i * h))
    sheet.save(sheet_path)


# ----------------------------------------------------------------------------------------------------
# badges: shields.io when reachable, otherwise an equivalent SVG drawn locally
def local_badge(label, value, colour):
    tw = lambda s: int(round(6.6 * len(s)))
    lw, rw = tw(label) + 10, tw(value) + 10
    w, L, R = lw + rw, html.escape(label), html.escape(value)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{L}: {R}"><title>{L}: {R}</title>'
            f'<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
            f'<clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="{lw}" height="20" fill="#555"/>'
            f'<rect x="{lw}" width="{rw}" height="20" fill="{colour}"/><rect width="{w}" height="20" fill="url(#s)"/></g>'
            f'<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="110" text-rendering="geometricPrecision">'
            f'<text x="{lw * 5}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(lw - 10) * 10}">{L}</text>'
            f'<text x="{lw * 5}" y="140" transform="scale(.1)" textLength="{(lw - 10) * 10}">{L}</text>'
            f'<text x="{(lw + rw / 2) * 10:.0f}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(rw - 10) * 10}">{R}</text>'
            f'<text x="{(lw + rw / 2) * 10:.0f}" y="140" transform="scale(.1)" textLength="{(rw - 10) * 10}">{R}</text></g></svg>')


def badge(name, label, value, colour):
    out = os.path.join(BADGES, f"{name}.svg")
    url = "https://img.shields.io/badge/" + "-".join(s.replace("-", "--").replace("_", "__").replace(" ", "%20").replace(",", "%2C") for s in (label, value)) + "-" + colour.lstrip("#")
    for _ in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=8).read().decode()
            if b.lstrip().startswith("<svg"):
                open(out, "w").write(b)
                return "shields.io"
        except Exception:
            time.sleep(1.5)
    open(out, "w").write(local_badge(label, value, colour))
    return "local"


if __name__ == "__main__":
    banner(DARK, os.path.join(FIG, "banner_dark.png"))
    banner(LIGHT, os.path.join(FIG, "banner_light.png"))
    gif(os.path.join(FIG, "protocol.gif"), "/tmp/protocol_sheet.png")
    for name, label, value, colour in [("questions", "questions", "102,072", "#2f81f7"), ("radiographs", "radiographs", "8,434", "#2f81f7"),
                                       ("findings", "findings", "22", "#2f81f7"), ("boxes", "evidence boxes", "lesion-level", "#e05d44")]:
        print(name, badge(name, label, value, colour))
    for f in sorted(os.listdir(FIG)):
        print(f"{os.path.getsize(os.path.join(FIG, f)) / 1024:8.0f} KB  {f}")
