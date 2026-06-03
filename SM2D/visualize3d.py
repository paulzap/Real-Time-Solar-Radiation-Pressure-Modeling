"""
visualize3d.py
==============
3-D visualization of satellite shadow / illumination results.

Primary renderer: Plotly (WebGL)
  - Per-face flat colors are angle-independent (lighting.ambient=1.0, diffuse=0.0)
  - Proper depth buffering — no painter's-algorithm artifacts
  - Interactive HTML saved next to the source CSV and opened in the default browser

Fallback renderer: Matplotlib (Poly3DCollection with shade=False)
  - Used only when Plotly is not installed
  - shade=False prevents the automatic normal-based darkening

Color scheme:
  Yellow      (#FFD700) — illuminated  (Label = 1, direct sun)
  Dark orange (#FF8C00) — 1st-bounce reflection
  Orange-red  (#FF4500) — 2nd-bounce reflection
  Orange      (#FF8C00) — 3rd+ bounce reflection
  Gray        (#606060) — shadowed     (Label = 0)
  Red         (#E53935) — misclassified (Correct = 0)
"""

import os
import sys
import webbrowser
import numpy as np
import pandas as pd

# ── Plotly (preferred) ───────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ── Matplotlib (fallback) ────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Color palette ────────────────────────────────────────────────────────────
_C_LIT    = '#FFD700'   # gold        — illuminated (direct sun)
_C_REFL1  = '#d860e0'   # dark orange — hit by 1st-bounce reflection
_C_REFL2  = '#4ade2c'   # orange-red  — hit by 2nd-bounce reflection
_C_REFL3  = '#FF8C00'   # orange      — hit by 3rd+ bounce reflection
_C_SHADOW = '#606060'   # gray        — shadowed / not reached by any light
_C_WRONG  = '#E53935'   # red         — misclassified

# Keep legacy alias so any external code using _C_REFL still works
_C_REFL = _C_REFL1


def _face_color(label: int, correct: int, refl_bounce: int = -1) -> str:
    if correct == 0:
        return _C_WRONG
    if label == 1:
        return _C_LIT
    if refl_bounce == 1:
        return _C_REFL1
    if refl_bounce == 2:
        return _C_REFL2
    if refl_bounce >= 3:
        return _C_REFL3
    return _C_SHADOW


# ── Clone-triangle handling ──────────────────────────────────────────────────

def _resolve_clones(triangles, normals, labels, correct, refl_bounce,
                    tolerance: float = 1e-6, epsilon: float = 0.01):
    """
    Detect triangles that share identical vertices but have opposite normals
    (double-sided panels stored as two coplanar faces). Each member of such a
    pair is offset by *epsilon* along its own normal so both faces are visible
    and z-fighting is eliminated.
    """
    vertex_hash: dict = {}
    for i in range(len(triangles)):
        key = tuple(np.round(triangles[i].flatten() / tolerance).astype(int))
        vertex_hash.setdefault(key, []).append(i)

    out_tris, out_labs, out_norms, out_corr, out_refl = [], [], [], [], []

    for indices in vertex_hash.values():
        if len(indices) == 1:
            i = indices[0]
            out_tris.append(triangles[i])
            out_labs.append(labels[i])
            out_norms.append(normals[i])
            out_corr.append(correct[i])
            out_refl.append(refl_bounce[i])
            continue

        processed: set = set()
        for i in indices:
            if i in processed:
                continue
            paired = False
            for j in indices:
                if j == i or j in processed:
                    continue
                if np.all(np.abs(normals[i] + normals[j]) < tolerance):
                    base = triangles[i]
                    out_tris.append(base + epsilon * normals[i])
                    out_labs.append(labels[i])
                    out_norms.append(normals[i])
                    out_corr.append(correct[i])
                    out_refl.append(refl_bounce[i])
                    out_tris.append(base + epsilon * normals[j])
                    out_labs.append(labels[j])
                    out_norms.append(normals[j])
                    out_corr.append(correct[j])
                    out_refl.append(refl_bounce[j])
                    processed.add(i)
                    processed.add(j)
                    paired = True
                    break
            if not paired and i not in processed:
                out_tris.append(triangles[i])
                out_labs.append(labels[i])
                out_norms.append(normals[i])
                out_corr.append(correct[i])
                out_refl.append(refl_bounce[i])
                processed.add(i)

    return (np.array(out_tris), np.array(out_labs),
            np.array(out_norms), np.array(out_corr),
            np.array(out_refl, dtype=int))


# ── Arrow helpers ────────────────────────────────────────────────────────────

def _plotly_arrow(vec, color: str, name: str, length: float):
    """Line + cone representing a 3-D arrow from the origin."""
    v = np.asarray(vec, dtype=float)
    mag = np.linalg.norm(v)
    if mag < 1e-10:
        return []
    d = v / mag
    tip   = d * length
    shaft = d * (length * 0.95)
    label = f'{name} = [{v[0]:.3e}, {v[1]:.3e}, {v[2]:.3e}]'
    line = go.Scatter3d(
        x=[0, shaft[0]], y=[0, shaft[1]], z=[0, shaft[2]],
        mode='lines',
        line=dict(color=color, width=5),
        name=label,
        showlegend=True,
        legendgroup=name,
    )
    cone = go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[d[0]],   v=[d[1]],   w=[d[2]],
        sizemode='absolute',
        sizeref=length * 0.15,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        showlegend=False,
        anchor='tip',
        legendgroup=name,
    )
    return [line, cone]


# ── Plotly renderer ──────────────────────────────────────────────────────────

def _render_plotly(tris, labs, norms, corr, refl,
                   show_normals: bool, force, moment,
                   out_html: str) -> None:
    N = len(tris)

    # Flatten vertex arrays for Mesh3d
    x = tris[:, :, 0].ravel()
    y = tris[:, :, 1].ravel()
    z = tris[:, :, 2].ravel()
    base = np.arange(N, dtype=np.int32) * 3

    face_colors = [_face_color(int(labs[t]), int(corr[t]), int(refl[t])) for t in range(N)]

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=base, j=base + 1, k=base + 2,
        facecolor=face_colors,
        flatshading=True,
        # ambient=1 → color is purely the face color regardless of viewing angle
        lighting=dict(ambient=1.0, diffuse=0.0,
                      specular=0.0, roughness=1.0, fresnel=0.0),
        name='Satellite',
        showlegend=False,
        hovertemplate=(
            'Face %{customdata[0]}<br>'
            'Label=%{customdata[1]}<br>'
            'Correct=%{customdata[2]}'
            '<extra></extra>'
        ),
        customdata=np.column_stack([np.arange(N), labs.astype(int), corr.astype(int)]),
    )
    traces = [mesh]

    # Normal vectors (thin lines from centroid)
    if show_normals:
        centers = tris.mean(axis=1)
        norm_len = (tris.reshape(-1, 3).max(axis=0) -
                    tris.reshape(-1, 3).min(axis=0)).max() * 0.04
        norm_len = max(norm_len, 0.5)
        for t in range(N):
            c = centers[t]
            tip = c + norms[t] * norm_len
            col = _C_LIT if labs[t] == 1 else _C_SHADOW
            traces.append(go.Scatter3d(
                x=[c[0], tip[0]], y=[c[1], tip[1]], z=[c[2], tip[2]],
                mode='lines',
                line=dict(color=col, width=1),
                showlegend=False,
                hoverinfo='skip',
            ))

    # Force / moment arrows
    all_pts = tris.reshape(-1, 3)
    model_size = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max())
    arrow_len = max(model_size * 0.5, 1.0)
    if force is not None:
        traces.extend(_plotly_arrow(force,  'orangered',   'F', arrow_len))
    if moment is not None:
        traces.extend(_plotly_arrow(moment, 'dodgerblue',  'M', arrow_len))

    # Legend proxies (colored squares)
    for col, label in [(_C_LIT,    'Illuminated (direct)'),
                       (_C_REFL1,  'Reflected light (bounce 1)'),
                       (_C_REFL2,  'Reflected light (bounce 2)'),
                       (_C_REFL3,  'Reflected light (bounce 3+)'),
                       (_C_SHADOW, 'Shadowed'),
                       (_C_WRONG,  'Misclassified')]:
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(color=col, size=10, symbol='square'),
            name=label,
            showlegend=True,
        ))

    # Equal-aspect scene bounds — minimum extent ±20 so arrows always fit
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    ctr  = (mins + maxs) / 2.0
    half = max(float((maxs - mins).max()) / 2.0, arrow_len, 20.0) * 1.1

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Satellite Shadow Visualization',
            scene=dict(
        xaxis=dict(range=[ctr[0] - half, ctr[0] + half], title='X',
                   showbackground=False, gridcolor='lightgray', zerolinecolor='lightgray'),
        yaxis=dict(range=[ctr[1] - half, ctr[1] + half], title='Y',
                   showbackground=False, gridcolor='lightgray', zerolinecolor='lightgray'),
        zaxis=dict(range=[ctr[2] - half, ctr[2] + half], title='Z',
                   showbackground=False, gridcolor='lightgray', zerolinecolor='lightgray'),
        aspectmode='cube',
        bgcolor='white',          # фон самого 3D-контейнера
    ),
    paper_bgcolor='white',        # фон вокруг графика
        font=dict(color='white'),
        legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(out_html, include_plotlyjs='cdn')
    # os.startfile on Windows always uses the registered default app for .html
    # (e.g. Opera), unlike webbrowser.open() which can resolve to Edge.
    # os.startfile requires an absolute path.
    abs_html = os.path.abspath(out_html)
    if sys.platform == 'win32':
        os.startfile(abs_html)
    else:
        webbrowser.open('file:///' + abs_html.replace('\\', '/'))
    print(f'[visualize3d] Plotly HTML: {abs_html}')


# ── Matplotlib fallback renderer ─────────────────────────────────────────────

def _render_matplotlib(tris, labs, norms, corr, refl,
                        show_normals: bool, force, moment) -> None:
    N = len(tris)
    colors = [_face_color(int(labs[i]), int(corr[i]), int(refl[i])) for i in range(N)]

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # shade=False disables the normal-based darkening (available in mpl >= 3.7;
    # falls back gracefully via try/except for older versions)
    try:
        col3d = Poly3DCollection(tris, shade=False,
                                 alpha=1.0, edgecolor='none')
    except TypeError:
        col3d = Poly3DCollection(tris, alpha=1.0, edgecolor='none')

    col3d.set_facecolor(colors)
    ax.add_collection3d(col3d)

    if show_normals:
        centers = tris.mean(axis=1)
        for i in range(N):
            c  = centers[i]
            col = _C_LIT if labs[i] == 1 else _C_SHADOW
            ax.quiver(c[0], c[1], c[2],
                      norms[i, 0], norms[i, 1], norms[i, 2],
                      color=col, length=1.0, normalize=True)

    all_pts   = tris.reshape(-1, 3)
    mins      = all_pts.min(axis=0)
    maxs      = all_pts.max(axis=0)
    arrow_len = max(20.0, float((maxs - mins).max()) * 0.5)
    mins      = np.minimum(mins, -20.0)
    maxs      = np.maximum(maxs,  20.0)
    max_range = float((maxs - mins).max())
    mid       = (maxs + mins) / 2.0
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    legend_handles = []

    def draw_arrow(vec, color, name):
        v    = np.asarray(vec, dtype=float)
        mag  = np.linalg.norm(v)
        if mag < 1e-10:
            return
        d = v / mag * arrow_len
        ax.quiver(0, 0, 0, d[0], d[1], d[2],
                  color=color, linewidth=2.5, arrow_length_ratio=0.18)
        ax.text(d[0] * 1.08, d[1] * 1.08, d[2] * 1.08,
                name, color=color, fontsize=9, fontweight='bold')
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=2.5,
                       label=f'{name} = [{v[0]:.3e}, {v[1]:.3e}, {v[2]:.3e}]'))

    if force  is not None:
        draw_arrow(force,  'orangered',  'F')
    if moment is not None:
        draw_arrow(moment, 'dodgerblue', 'M')

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    ax.set_title('3D Satellite Shadow Visualization (matplotlib fallback)')
    plt.show(block=True)


# ── Public entry point ───────────────────────────────────────────────────────

def visualize_triangles(csv_file: str,
                        step: int = 1,
                        show_normals: bool = False,
                        force=None,
                        moment=None) -> None:
    """
    Visualize satellite illumination / shadow results from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the results CSV (columns: V1_X … V3_Z, Normal_X/Y/Z,
        Label, and optionally Correct).
    step : int
        Sampling stride: 1 = every triangle, N = every N-th triangle.
        0 = auto (limits render to ~5 000 faces).
    show_normals : bool
        Draw surface normal vectors at each centroid.
    force : sequence[float] | None
        SRP force  vector [Fx, Fy, Fz] — drawn as an orange arrow.
    moment : sequence[float] | None
        SRP moment vector [Mx, My, Mz] — drawn as a blue arrow.
    """
    # ── Load CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f'[visualize3d] File not found: {csv_file}')
        return
    except Exception as e:
        print(f'[visualize3d] Error reading CSV: {e}')
        return

    triangles   = (df[['V1_X', 'V1_Y', 'V1_Z',
                        'V2_X', 'V2_Y', 'V2_Z',
                        'V3_X', 'V3_Y', 'V3_Z']]
                   .values.reshape(-1, 3, 3))
    normals     = df[['Normal_X', 'Normal_Y', 'Normal_Z']].values
    labels      = df['Label'].values
    correct     = (df['Correct'].values
                   if 'Correct' in df.columns
                   else np.ones(len(labels), dtype=int))
    # ReflectionBounce: -1=never hit, 0=direct, 1=1st reflection, 2+=later
    # Present only in CSVs written by save_results_with_bounces().
    refl_bounce = (df['ReflectionBounce'].values.astype(int)
                   if 'ReflectionBounce' in df.columns
                   else np.full(len(labels), -1, dtype=int))

    if len(triangles) == 0:
        print('[visualize3d] No triangles in CSV.')
        return

    # ── Auto step ────────────────────────────────────────────────────────────
    MAX_FACES = 5000
    if step == 0:
        step = max(1, len(triangles) // MAX_FACES + 1)
        if step > 1:
            print(f'[visualize3d] Auto step={step} '
                  f'(showing ~{len(triangles) // step} of {len(triangles)} faces)')

    # ── Clone handling ────────────────────────────────────────────────────────
    vis_tris, vis_labs, vis_norms, vis_corr, vis_refl = _resolve_clones(
        triangles, normals, labels, correct, refl_bounce)

    # ── Apply sampling stride ─────────────────────────────────────────────────
    idx       = np.arange(0, len(vis_tris), max(1, step))
    vis_tris  = vis_tris[idx]
    vis_labs  = vis_labs[idx]
    vis_norms = vis_norms[idx]
    vis_corr  = vis_corr[idx]
    vis_refl  = vis_refl[idx]

    # ── Dispatch to renderer ──────────────────────────────────────────────────
    if _PLOTLY:
        out_html = os.path.splitext(csv_file)[0] + '_visualization.html'
        _render_plotly(vis_tris, vis_labs, vis_norms, vis_corr, vis_refl,
                       show_normals, force, moment, out_html)
    else:
        print('[visualize3d] Plotly not found — using matplotlib fallback. '
              'Install with: pip install plotly')
        _render_matplotlib(vis_tris, vis_labs, vis_norms, vis_corr, vis_refl,
                           show_normals, force, moment)
