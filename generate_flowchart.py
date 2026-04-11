"""
Generate an updated F1 Race Prediction pipeline flowchart as a PNG image.
Dark premium theme — tight layout, no wasted space.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── colours ──────────────────────────────────────────────────────────────
BG       = "#0f1923"
BOX_DARK = "#1a2736"
BORDER   = "#2a3a4d"
RED      = "#c0392b"
RED_B    = "#e74c3c"
PURPLE   = "#2d2d7a"
PURP_B   = "#4a4aaa"
GREEN    = "#1a7a42"
GREEN_B  = "#27ae60"
ORANGE   = "#b7950b"
ORANGE_B = "#d4ac0d"
BROWN    = "#9c4a10"
BROWN_B  = "#c0601a"
WHITE    = "#f0f0f0"
GREY_TXT = "#90a4ae"
ARROW_C  = "#4a7aaa"

fig, ax = plt.subplots(figsize=(9, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(4.5, 17.5)
ax.axis("off")

# ── helper: draw a rounded box ──────────────────────────────────────────
def draw_box(cx, cy, w, h, title, subtitle, fill=BOX_DARK, border=BORDER,
             title_size=12, sub_size=8.5):
    x = cx - w / 2
    y = cy - h / 2
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fill, edgecolor=border, linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(box)
    ty = cy + 0.12 if subtitle else cy
    ax.text(cx, ty, title, color=WHITE, fontsize=title_size,
            fontweight="bold", ha="center", va="center", zorder=3,
            fontfamily="Segoe UI")
    if subtitle:
        ax.text(cx, cy - 0.18, subtitle, color=GREY_TXT, fontsize=sub_size,
                ha="center", va="center", zorder=3, fontfamily="Segoe UI",
                style="italic")

def draw_arrow(x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=ARROW_C, lw=2.2, mutation_scale=18,
        ),
        zorder=1,
    )

# ── node positions ───────────────────────────────────────────────────────
W, H   = 5.4, 0.78
WS, HS = 2.8, 0.7

SP  = 1.15   # vertical spacing between boxes
CX  = 5.0
top = 16.3   # y of first box

Y_api    = top
Y_raw    = Y_api    - SP
Y_clean  = Y_raw    - SP
Y_feat   = Y_clean  - SP - 0.15
Y_split  = Y_feat   - SP - 0.15
Y_target = Y_split  - SP
Y_weight = Y_target - SP
Y_model  = Y_weight - SP - 0.05
Y_rank   = Y_model  - SP - 0.05
Y_eval   = Y_rank   - SP

TRAIN_X, TEST_X = 3.3, 6.7

# ── draw nodes ───────────────────────────────────────────────────────────
draw_box(CX, Y_api, 2.8, 0.65, "FastF1 API", None, RED, RED_B, title_size=13)

draw_box(CX, Y_raw, W, H,
         "Raw Data Collection", "4 seasons: 2022-2025, 1829 rows")

draw_box(CX, Y_clean, W, H,
         "Data Cleaning & Filtering", "Drop ALL DNFs, status_eval, snake_case")

draw_box(CX, Y_feat, W + 1.0, 1.05,
         "Feature Engineering",
         "gap_to_pole_norm, target_encodings, intra_season_form,\n"
         "grid_position, weighted_finish_form, team_perf, dnf_rate, standings",
         PURPLE, PURP_B, sub_size=8)

draw_box(TRAIN_X, Y_split, WS, HS,
         "Train Set", "Seasons 2022-2024", GREEN, GREEN_B)
draw_box(TEST_X,  Y_split, WS, HS,
         "Test Set",  "Season 2025", BROWN, BROWN_B)

draw_box(CX, Y_target, W, H,
         "Target Transformation",
         "Predict delta = grid_position - finish_position",
         ORANGE, ORANGE_B)

draw_box(CX, Y_weight, W, H,
         "Season Weighting",
         "2022 = 0.2,  2023 = 0.5,  2024 = 1.0",
         GREEN, GREEN_B)

draw_box(CX, Y_model, W + 0.4, 0.9,
         "Model Ensemble",
         "LR + RF + XGB + LightGBM + CatBoost  ->  Ridge Meta-Learner",
         PURPLE, PURP_B)

draw_box(CX, Y_rank, W, H,
         "Rank Post-Processing",
         "Force 1-20 integer ranks per race",
         ORANGE, ORANGE_B)

draw_box(CX, Y_eval, W, H,
         "Evaluation",
         "MAE on 2025 test set: 2.225",
         GREEN, GREEN_B)

# ── arrows ───────────────────────────────────────────────────────────────
g  = 0.39
gb = 0.52   # clearance for larger boxes

draw_arrow(CX, Y_api   - 0.32, CX, Y_raw   + g)
draw_arrow(CX, Y_raw   - g,    CX, Y_clean  + g)
draw_arrow(CX, Y_clean - g,    CX, Y_feat   + gb)

# Feature eng -> train / test
draw_arrow(4.0, Y_feat  - gb,  TRAIN_X, Y_split + 0.35)
draw_arrow(6.0, Y_feat  - gb,  TEST_X,  Y_split + 0.35)

# Train -> target transform
draw_arrow(TRAIN_X, Y_split - 0.35, CX, Y_target + g)

# Target -> weighting -> model
draw_arrow(CX, Y_target - g, CX, Y_weight + g)
draw_arrow(CX, Y_weight - g, CX, Y_model  + 0.45)

# Test -> model (vertical down then horizontal)
# Vertical line from Test Set
test_bottom = Y_split - 0.35
model_entry_y = Y_model + 0.10
ax.plot([TEST_X, TEST_X], [test_bottom, model_entry_y], color=ARROW_C,
        lw=2.2, zorder=1)
# Horizontal arrow into model
draw_arrow(TEST_X, model_entry_y, CX + W/2 + 0.2, model_entry_y)

# Model -> rank -> eval
draw_arrow(CX, Y_model - 0.45, CX, Y_rank + g)
draw_arrow(CX, Y_rank  - g,    CX, Y_eval + g)

# ── checkmark badge on Evaluation ────────────────────────────────────────
badge_x = CX + W/2 - 0.1
badge_y = Y_eval + 0.15
circle = plt.Circle((badge_x, badge_y), 0.2,
                     facecolor=GREEN_B, edgecolor=WHITE, linewidth=1.5, zorder=4)
ax.add_patch(circle)
# Draw a checkmark using lines
check_pts_x = [badge_x - 0.08, badge_x - 0.02, badge_x + 0.10]
check_pts_y = [badge_y,        badge_y - 0.07,  badge_y + 0.10]
ax.plot(check_pts_x, check_pts_y, color=WHITE, lw=2, zorder=5,
        solid_capstyle="round", solid_joinstyle="round")

# ── title ────────────────────────────────────────────────────────────────
ax.text(CX, 17.2, "F1 Race Prediction Pipeline", color=WHITE, fontsize=19,
        fontweight="bold", ha="center", va="center", fontfamily="Segoe UI")
ax.text(CX, 16.85, "Drop-DNF  |  Delta Target  |  Rank Post-Processing",
        color=GREY_TXT, fontsize=10, ha="center", va="center",
        fontfamily="Segoe UI", style="italic")

# ── save ─────────────────────────────────────────────────────────────────
out = r"d:\Python Programming\F1 Race Prediction\outputs\pipeline_flowchart.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.4)
plt.close()
print("Saved to outputs/pipeline_flowchart.png")
