"""
animate_srp.py
==============
Plays a recording written by SolaRT FrameRecorder (demo section 17, or any
program that uses FrameRecorder) as a self-contained HTML page: the satellite
with its moving parts, illuminated and shadowed triangles, the direction to
the sun and the force and moment vectors, played in real time; play / pause,
a time slider, the camera turns with the mouse while the animation runs.

    python animate_srp.py <recording folder> [--style tech|space]
                          [--frame body|view] [--every N] [--fps F]
                          [--out page.html] [--no-open]

Two looks, switchable on the page:
  tech   - white background, gold = illuminated, gray = shadowed, body axes,
           the sun's path; for visual checking;
  space  - star sky with the Milky Way, the sun with its glow far away,
           materials guessed from the component types (solar cells, gold
           foil, white paint, metal); the shadows are still the library's.
Two frames: body (the satellite stands still, the sun moves round it) and
view (the frame of the recording's body_to_view rotation, e.g. inertial: the
satellite turns, the sun and the stars stay). The stars are fixed in the view
frame, so pass body -> inertial to FrameRecorder for a true sky.

The page draws with WebGL 2 and needs nothing else (no internet, no plotting
package): the geometry is stored once, each frame adds only the motion of
every mesh and one bit per triangle, all gzip-compressed. Only numpy is
needed here. For display and visual checking: the numbers come from the
library, the page only draws them. Recording format: docs/USAGE.md.
"""

import argparse
import base64
import gzip
import json
import os
import sys
import webbrowser

import numpy as np

# Space look: base color, specular strength, shininess, pattern
# (0 none, 1 solar cells, 2 crinkled foil). Chosen by material().
MATERIALS = [
    ([0.72, 0.72, 0.72], 0.25, 30, 0),    # 0 other
    ([0.05, 0.08, 0.24], 0.40, 220, 1),   # 1 solar cells
    ([0.86, 0.62, 0.20], 0.75, 22, 2),    # 2 gold foil (multi-layer insulation)
    ([0.88, 0.88, 0.86], 0.15, 20, 0),    # 3 white paint (radiators, antennas)
    ([0.20, 0.20, 0.22], 0.30, 40, 0),    # 4 dark structure (booms, struts)
    ([0.58, 0.59, 0.61], 0.85, 70, 0),    # 5 bare metal (engines, thrusters, tanks)
    ([0.10, 0.10, 0.11], 0.35, 50, 0),    # 6 black (instruments)
]


def material(ctype, name):
    """Material of a mesh from its HDF5 component_type and name (a guess for display)."""
    t, n = ctype.lower(), name.lower()
    if 'solar' in t or 'solar' in n or t == 'panel' or 'array' in n:
        return 1
    if 'radiator' in t or 'antenna' in t or 'dish' in n:
        return 3
    if 'boom' in t or 'strut' in t or 'mast' in t:
        return 4
    if 'engine' in t or 'thruster' in t or 'tank' in t or 'nozzle' in t:
        return 5
    if 'instrument' in t or 'experiment' in t or 'sensor' in t or 'camera' in t:
        return 6
    if 'bus' in t or 'body' in t or 'module' in t:
        return 2
    return 0


def read_recording(folder):
    with open(os.path.join(folder, 'scene.json'), encoding='utf-8') as f:
        scene = json.load(f)
    if scene.get('format') != 'SolaRT frames':
        sys.exit(f"{folder}: not a SolaRT recording")
    n, m = scene['triangles'], len(scene['meshes'])
    geometry = np.fromfile(os.path.join(folder, 'geometry.bin'), dtype='<f4').reshape(n, 9)
    record = np.dtype([('head', '<f8', (19,)), ('motion', '<f4', (m, 12)), ('label', 'u1', (n,))])
    if record.itemsize != scene['record_bytes']:
        sys.exit(f"{folder}: record size {record.itemsize} != {scene['record_bytes']} in scene.json")
    frames = np.fromfile(os.path.join(folder, 'frames.bin'), dtype=record, count=scene['frames'])
    return scene, geometry, frames


def b64(a):
    """gzip, then base64: the page unpacks it with the browser's own DecompressionStream."""
    raw = np.ascontiguousarray(a).tobytes()
    return base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode('ascii')


def scene_bounds(geometry, meshes, frames, views):
    """Centre and radius of the model over all frames, drawn with the given rotations."""
    corners = []
    for k, mesh in enumerate(meshes):
        if mesh['count'] == 0:
            continue
        p = geometry[mesh['first']:mesh['first'] + mesh['count']].reshape(-1, 3).astype(np.float64)
        lo, hi = p.min(axis=0), p.max(axis=0)
        box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        mot = frames['motion'][:, k].astype(np.float64).reshape(-1, 3, 4)
        q = np.einsum('fij,bj->fbi', mot[:, :, :3], box) + mot[:, None, :, 3]
        corners.append(np.einsum('fij,fbj->fbi', views, q).reshape(-1, 3))
    allc = np.vstack(corners)
    lo, hi = allc.min(axis=0), allc.max(axis=0)
    return (0.5 * (lo + hi)).tolist(), 0.5 * float(np.linalg.norm(hi - lo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--style', choices=['tech', 'space'], default='tech', help='look at start')
    ap.add_argument('--frame', choices=['body', 'view'], default=None,
                    help='frame at start (default: body for tech, view for space)')
    ap.add_argument('--every', type=int, default=1, help='use every N-th frame')
    ap.add_argument('--fps', type=int, default=15, help='frames per second at start (changeable on the page)')
    ap.add_argument('--out', default=None, help='page to write (default: <folder>/animation.html)')
    ap.add_argument('--no-open', action='store_true', help='do not open the page in the browser')
    args = ap.parse_args()

    scene, geometry, frames = read_recording(args.folder)
    step = max(1, args.every)
    frames = frames[::step]
    notes = scene.get('notes', [])[::step]
    n, meshes = scene['triangles'], scene['meshes']
    if len(frames) == 0:
        sys.exit(f"{args.folder}: no frames")

    head = frames['head']
    views = head[:, 10:19].reshape(-1, 3, 3)
    has_view = not np.allclose(views, np.eye(3))
    centre_body, radius_body = scene_bounds(geometry, meshes, frames, np.broadcast_to(np.eye(3), views.shape))
    centre_view, radius_view = scene_bounds(geometry, meshes, frames, views)
    frame = args.frame or ('view' if args.style == 'space' else 'body')
    if not has_view:
        frame = 'body'

    # Vertices quantised to 16 bits over the bounding box of the file pose
    # (0.3 mm on a 20 m model): half the size, invisible on the screen.
    p = geometry.reshape(-1, 3).astype(np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    quant = np.round((p - lo) / span * 65535.0).astype('<u2')

    scale = float(scene.get('pressure_scale', 1.0))
    labels = frames['label'] > 0
    force, moment = head[:, 4:7], head[:, 7:10]
    meta = dict(
        model=os.path.basename(scene.get('model', '').replace('\\', '/')), N=n, F=len(frames),
        meshes=[[m['first'], m['count']] for m in meshes],
        material=[material(m.get('type', ''), m.get('name', '')) for m in meshes],
        materials=MATERIALS,
        lo=lo.tolist(), span=span.tolist(),
        centre={'body': centre_body, 'view': centre_view}, radius={'body': radius_body, 'view': radius_view},
        time=head[:, 0].tolist(), sun=head[:, 1:4].ravel().tolist(), view=head[:, 10:19].ravel().tolist(),
        force=force.ravel().tolist(), moment=moment.ravel().tolist(),
        fmax=float(np.linalg.norm(force, axis=1).max()), mmax=float(np.linalg.norm(moment, axis=1).max()),
        mref=scene.get('moment_reference', [0.0, 0.0, 0.0]),
        units=['m²', 'm³', 'per unit pressure'] if scale == 1.0 else ['N', 'N·m', ''],
        lit=labels.sum(axis=1).tolist(), notes=notes, fps=args.fps,
        style=args.style, frame=frame, hasView=bool(has_view))
    bits = np.packbits(labels, axis=1, bitorder='little')

    page = (PAGE.replace('__TITLE__', meta['model'] or 'SolaRT animation')
                .replace('__META__', json.dumps(meta))
                .replace('__GEO__', b64(quant))
                .replace('__MOT__', b64(frames['motion'].astype('<f4')))
                .replace('__LAB__', b64(bits)))
    out = args.out or os.path.join(args.folder, 'animation.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f"  {len(frames)} frames, {n} triangles -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open('file://' + os.path.abspath(out))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__ - SolaRT animation</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #fff;
               font-family: "Segoe UI", Arial, sans-serif; }
  #gl { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
  #info { position: fixed; top: 8px; left: 12px; right: 280px; font-size: 14px; color: #222; }
  #help { position: fixed; top: 8px; right: 12px; font-size: 12px; color: #666; text-align: right; }
  #bar { position: fixed; left: 12px; right: 12px; bottom: 10px; display: flex; gap: 12px; flex-wrap: wrap;
         align-items: center; font-size: 13px; background: rgba(255,255,255,0.85); color: #222;
         padding: 6px 10px; border-radius: 6px; }
  #slider { flex: 1; min-width: 160px; }
  button { min-width: 64px; }
  .lab { position: fixed; font-size: 13px; font-weight: 600; pointer-events: none; }
  body.space #info { color: #e8e8e8; }
  body.space #help { color: #999; }
  body.space #bar { background: rgba(20,20,28,0.75); color: #ddd; }
</style></head>
<body>
<canvas id="gl"></canvas>
<div id="info"></div>
<div id="help">drag: turn &nbsp; wheel: zoom &nbsp; right drag: move<br>space: play / pause &nbsp; &larr; &rarr;: frame<br>
  arrows: scaled to the largest value of the recording</div>
<div id="bar"><button id="play">Play</button><input id="slider" type="range" min="0" value="0" step="1">
  <span id="tval"></span>
  <label>speed <select id="fps"></select> frames/s</label>
  <label><input type="checkbox" id="showF" checked> force</label>
  <label><input type="checkbox" id="showM" checked> moment</label>
  <label>look <select id="style"><option value="tech">technical</option><option value="space">space</option></select></label>
  <label id="frameBox">frame <select id="frame"><option value="body">body</option><option value="view">view</option></select></label>
</div>
<div class="lab" id="lx" style="color:#c62828">x</div>
<div class="lab" id="ly" style="color:#2e7d32">y</div>
<div class="lab" id="lz" style="color:#1565c0">z</div>
<div class="lab" id="lF" style="color:#d81b78">F</div>
<div class="lab" id="lM" style="color:#00a3b4">M</div>
<script type="text/plain" id="geo">__GEO__</script>
<script type="text/plain" id="mot">__MOT__</script>
<script type="text/plain" id="lab">__LAB__</script>
<script>const META = __META__;</script>
<script>
'use strict';
(async function () {
const info = document.getElementById('info');
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: true });
if (!gl) { info.textContent = 'WebGL 2 is not available in this browser.'; return; }
if (typeof DecompressionStream === 'undefined') {
  info.textContent = 'This browser cannot unpack the data: use a current Chrome, Edge or Firefox.'; return;
}
info.textContent = 'Loading ' + META.N + ' triangles, ' + META.F + ' frames...';

// base64 + gzip -> ArrayBuffer, decoded by the browser itself
async function bytes(id) {
  const node = document.getElementById(id);
  const res = await fetch('data:application/octet-stream;base64,' + node.textContent.trim());
  node.textContent = '';
  return new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
const N = META.N, F = META.F, M = META.meshes.length, B = Math.ceil(N / 8);
const geo = new Uint16Array(await bytes('geo'));
const mot = new Float32Array(await bytes('mot'));
const lab = new Uint8Array(await bytes('lab'));

// ---- shaders ---------------------------------------------------------------
function program(vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
  const u = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(p, i).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}
const meshProg = program(`#version 300 es
layout(location = 0) in vec3 q;          // vertex at the file pose, 16-bit
layout(location = 1) in float lit;
uniform vec3 uLo, uSpan;
uniform mat4 uModel, uViewProj;
out vec3 vPos, vQ; flat out float vLit;
void main() {
  vQ = uLo + q * uSpan;
  vec4 w = uModel * vec4(vQ, 1.0);
  vPos = w.xyz; vLit = lit;
  gl_Position = uViewProj * w;
}`, `#version 300 es
precision highp float;
in vec3 vPos, vQ; flat in float vLit;
uniform vec3 uSun, uEye, uBase;
uniform int uStyle, uPattern;            // style: 0 technical, 1 space
uniform float uKs, uShine;
out vec4 color;
float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
void main() {
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  vec3 v = normalize(uEye - vPos);
  if (dot(n, v) < 0.0) n = -n;            // the side the camera sees
  float d = max(dot(n, uSun), 0.0);
  if (uStyle == 0) {
    vec3 base = vLit > 0.5 ? vec3(1.0, 0.843, 0.0) : vec3(0.42, 0.42, 0.44);
    color = vec4(base * (0.55 + 0.5 * d), 1.0);
    return;
  }
  vec3 base = uBase;
  if (uPattern == 1) {                    // solar cells: a grid of silver lines in the panel plane
    vec3 nq = abs(normalize(cross(dFdx(vQ), dFdy(vQ))));
    vec2 uv = (nq.x > nq.y && nq.x > nq.z ? vQ.yz : (nq.y > nq.z ? vQ.xz : vQ.xy)) / 0.16;
    vec2 fw = fwidth(uv);
    vec2 g = abs(fract(uv + 0.5) - 0.5) / max(fw, 1e-4);
    float line = (1.0 - clamp(min(g.x, g.y) - 0.5, 0.0, 1.0)) * clamp(1.5 - 4.0 * max(fw.x, fw.y), 0.0, 1.0);
    base = mix(base, vec3(0.62, 0.64, 0.68), 0.55 * line);
  } else if (uPattern == 2) {             // crinkled foil
    base *= 0.78 + 0.44 * hash(floor(vQ * 12.0));
  }
  vec3 h = normalize(uSun + v);
  float lit = vLit > 0.5 ? 1.0 : 0.0;     // the library's shadow test
  float spec = lit * step(0.0, dot(n, uSun)) * uKs * pow(max(dot(n, h), 0.0), uShine);
  // faint fill from the camera side (light reflected by the Earth), so the shadowed side keeps its shape
  vec3 c = base * (0.03 + 1.2 * lit * d) + vec3(0.55, 0.65, 0.9) * base * 0.12 * max(dot(n, v), 0.0)
          + vec3(1.0, 0.97, 0.9) * spec;
  color = vec4(1.0 - exp(-1.7 * c), 1.0);
}`);
const solidProg = program(`#version 300 es
layout(location = 0) in vec3 p;
uniform mat4 uModel, uViewProj;
out vec3 vPos;
void main() { vec4 w = uModel * vec4(p, 1.0); vPos = w.xyz; gl_Position = uViewProj * w; }`,
`#version 300 es
precision highp float;
in vec3 vPos; uniform vec3 uEye, uColor; out vec4 color;
void main() {
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  color = vec4(uColor * (0.45 + 0.55 * abs(dot(n, normalize(uEye - vPos)))), 1.0);
}`);
const lineProg = program(`#version 300 es
layout(location = 0) in vec3 p;
uniform mat4 uViewProj; uniform float uSize;
void main() { gl_Position = uViewProj * vec4(p, 1.0); gl_PointSize = uSize; }`,
`#version 300 es
precision mediump float;
uniform vec4 uColor; uniform bool uRound;
out vec4 color;
void main() {
  if (uRound && length(gl_PointCoord - 0.5) > 0.5) discard;
  color = uColor;
}`);
const starProg = program(`#version 300 es
layout(location = 0) in vec4 star;       // direction in the sky frame, brightness
layout(location = 1) in vec3 tint;
uniform mat4 uViewProj; uniform mat3 uSkyToRender; uniform float uDpr;
out float vB; out vec3 vC;
void main() {
  vec4 c = uViewProj * vec4(uSkyToRender * star.xyz, 0.0);
  gl_Position = vec4(c.xy, c.w, c.w);    // at infinity
  vB = star.w; vC = tint;
  gl_PointSize = (1.0 + 2.6 * star.w) * uDpr;
}`, `#version 300 es
precision mediump float;
in float vB; in vec3 vC; out vec4 color;
void main() {
  float r = length(gl_PointCoord - 0.5);
  float a = (0.25 + vB) * (1.0 - smoothstep(0.15, 0.5, r));
  color = vec4(vC * a, 1.0);
}`);
const skyProg = program(`#version 300 es
out vec2 vXY;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  vXY = p; gl_Position = vec4(p, 0.0, 1.0);
}`, `#version 300 es
precision highp float;
in vec2 vXY;
uniform vec3 uCamX, uCamY, uCamZ, uSun;  // camera basis, sun: render frame
uniform float uTan, uAspect;
uniform mat3 uSky;                        // render frame -> sky frame
out vec4 color;
float h(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float noise(vec3 p) {
  vec3 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(h(i), h(i + vec3(1, 0, 0)), f.x), mix(h(i + vec3(0, 1, 0)), h(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(h(i + vec3(0, 0, 1)), h(i + vec3(1, 0, 1)), f.x), mix(h(i + vec3(0, 1, 1)), h(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++) { s += a * noise(p); p *= 2.03; a *= 0.5; } return s; }
void main() {
  vec3 d = normalize(uCamX * vXY.x * uTan * uAspect + uCamY * vXY.y * uTan - uCamZ);
  vec3 ds = uSky * d;
  // Milky Way: a band with dust lanes
  vec3 g = normalize(vec3(0.35, -0.6, 0.72));
  float lat = dot(ds, g);
  float band = exp(-lat * lat / 0.03) * (0.3 + 0.7 * fbm(ds * 3.5));
  float dust = smoothstep(0.45, 0.75, fbm(ds * 7.0 + 3.0)) * exp(-lat * lat / 0.006);
  vec3 c = vec3(0.003, 0.004, 0.009) + band * (1.0 - 0.7 * dust) * vec3(0.09, 0.082, 0.075);
  // the sun: limb-darkened disc, glow, diffraction spikes
  float ca = dot(d, uSun);
  float ang = acos(clamp(ca, -1.0, 1.0));
  const float r0 = 0.011;
  if (ang < r0) { float mu = sqrt(1.0 - (ang / r0) * (ang / r0)); c += vec3(1.0, 0.96, 0.86) * 40.0 * (0.35 + 0.65 * mu); }
  c += vec3(1.0, 0.86, 0.62) * (3.0 * exp(-ang / 0.01) + 0.7 * exp(-ang / 0.045) + 0.12 * exp(-ang / 0.25));
  vec3 t1 = normalize(cross(uSun, abs(uSun.z) < 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0))), t2 = cross(uSun, t1);
  float phi = atan(dot(d, t2), dot(d, t1));
  c += vec3(1.0, 0.9, 0.75) * (pow(abs(cos(3.0 * phi)), 120.0) + 0.5 * pow(abs(cos(3.0 * phi + 0.52)), 300.0))
       * 0.9 * exp(-ang / 0.09) * step(0.0, ca);
  color = vec4(1.0 - exp(-c), 1.0);
}`);

// ---- buffers ---------------------------------------------------------------
const meshVao = gl.createVertexArray();
gl.bindVertexArray(meshVao);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, geo, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.UNSIGNED_SHORT, true, 0, 0);
const litData = new Uint8Array(3 * N);
const litBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, litBuf);
gl.bufferData(gl.ARRAY_BUFFER, litData, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 1, gl.UNSIGNED_BYTE, true, 0, 0);

const lineVao = gl.createVertexArray();
gl.bindVertexArray(lineVao);
const lineBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

// arrow along +z: shaft of radius 1 up to z = 0.82, head of radius 2.6 to z = 1
const arrowPts = [];
const S = 20;
for (let i = 0; i < S; i++) {
  const a0 = 2 * Math.PI * i / S, a1 = 2 * Math.PI * (i + 1) / S;
  const c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
  arrowPts.push(c0, s0, 0, c1, s1, 0, c1, s1, 0.82, c0, s0, 0, c1, s1, 0.82, c0, s0, 0.82);
  arrowPts.push(2.6 * c0, 2.6 * s0, 0.82, 2.6 * c1, 2.6 * s1, 0.82, 0, 0, 1);
  arrowPts.push(2.6 * c1, 2.6 * s1, 0.82, 2.6 * c0, 2.6 * s0, 0.82, 0, 0, 0.82);
}
const arrowVao = gl.createVertexArray();
gl.bindVertexArray(arrowVao);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arrowPts), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

// stars: uniform over the sky plus a denser band along the Milky Way (fixed seed)
let seed = 12345;
const rnd = () => { seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const gal = (() => { const v = [0.35, -0.6, 0.72], l = Math.hypot(...v); return v.map(x => x / l); })();
const tints = [[0.75, 0.85, 1.0], [1.0, 1.0, 1.0], [1.0, 0.95, 0.82], [1.0, 0.8, 0.62]];
const starData = [], tintData = [];
for (let i = 0; i < 16000; i++) {
  let x, y, z;
  if (i < 7000) { z = 2 * rnd() - 1; const a = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z); x = r * Math.cos(a); y = r * Math.sin(a); }
  else {
    // a random direction pulled towards the galactic plane
    do { x = 2 * rnd() - 1; y = 2 * rnd() - 1; z = 2 * rnd() - 1; } while (x * x + y * y + z * z > 1 || x * x + y * y + z * z < 1e-3);
    const l = Math.hypot(x, y, z); x /= l; y /= l; z /= l;
    const dg = x * gal[0] + y * gal[1] + z * gal[2], keep = 0.12 * (rnd() + rnd() + rnd() - 1.5) / 0.5;
    x += (keep - dg) * gal[0]; y += (keep - dg) * gal[1]; z += (keep - dg) * gal[2];
    const m = Math.hypot(x, y, z); x /= m; y /= m; z /= m;
  }
  const b = i < 7000 ? Math.pow(rnd(), 7) : 0.35 * Math.pow(rnd(), 4);
  starData.push(x, y, z, b);
  tintData.push(...tints[Math.min(3, Math.floor(rnd() * rnd() * 5))]);
}
const starVao = gl.createVertexArray();
gl.bindVertexArray(starVao);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(starData), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tintData), gl.STATIC_DRAW);
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
const emptyVao = gl.createVertexArray();
gl.bindVertexArray(null);

// ---- matrices (column-major, as WebGL wants) -------------------------------
function mul(a, b) {
  const r = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let i = 0; i < 4; i++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[c * 4 + k]; r[c * 4 + i] = s;
  }
  return r;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), r = new Float32Array(16);
  r[0] = f / aspect; r[5] = f; r[10] = (far + near) / (near - far); r[11] = -1;
  r[14] = 2 * far * near / (near - far);
  return r;
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function lookAt(e, t, up) {
  const z = norm(sub(e, t)), x = norm(cross(up, z)), y = cross(z, x);
  return { m: new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
                                -dot(x, e), -dot(y, e), -dot(z, e), 1]), x, y, z };
}

// ---- state -------------------------------------------------------------------
let frame = 0, playing = false, last = 0;
let style = META.style, frameMode = META.frame;
const useView = () => frameMode === 'view';
// rotation drawn for frame f: body_to_view in the view frame, identity in the body frame
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const rot = f => useView() ? META.view.slice(9 * f, 9 * f + 9) : I3;
const apply = (R, x) => [0, 1, 2].map(i => R[3 * i] * x[0] + R[3 * i + 1] * x[1] + R[3 * i + 2] * x[2]);
const vec = (arr, f) => arr.slice(3 * f, 3 * f + 3);
function modelMatrix(f, k) {          // rot(f) (3x3) times the motion [R|t] (3x4) -> column-major 4x4
  const V = rot(f), o = (f * M + k) * 12, r = new Float32Array(16);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let c = 0; c < 3; c++) s += V[3 * i + c] * mot[o + 4 * c + j];
    r[j * 4 + i] = s;
  }
  r[15] = 1;
  return r;
}

// ---- camera --------------------------------------------------------------------
let az = 0.8, el = 0.45, zoom = 4.2, target = null;
const C = () => META.centre[frameMode], R = () => META.radius[frameMode];
function resetTarget() { target = C().slice(); }
resetTarget();
// space look: look past the satellite with the sun 17 degrees off the line of sight, near the edge
function sunCamera() {
  const s = apply(rot(frame), vec(META.sun, frame));
  const ax = norm(cross(s, Math.abs(s[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0]));
  const a = 17 * Math.PI / 180, d = s.map((x, i) => Math.cos(a) * x + Math.sin(a) * cross(ax, s)[i]);
  const e = d.map(x => -x);                         // from the satellite to the camera
  az = Math.atan2(e[1], e[0]); el = Math.max(-1.4, Math.min(1.4, Math.asin(e[2]))); zoom = 3.4;
}
function techCamera() { az = 0.8; el = 0.45; zoom = 4.2; }
let drag = null;
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, b: e.button }; });
window.addEventListener('mouseup', () => { drag = null; });
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.b === 0) { az -= dx * 0.008; el = Math.max(-1.55, Math.min(1.55, el + dy * 0.008)); }
  else {
    const s = zoom * R() * 0.0015, ca = Math.cos(az), sa = Math.sin(az), se = Math.sin(el), ce = Math.cos(el);
    const right = [-sa, ca, 0], up = [-se * ca, -se * sa, ce];
    for (let i = 0; i < 3; i++) target[i] += (-dx * right[i] + dy * up[i]) * s;
  }
  draw();
});
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(e.deltaY * 0.001); draw(); },
                        { passive: false });

// ---- controls ------------------------------------------------------------------
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
const fpsSel = document.getElementById('fps'), styleSel = document.getElementById('style');
const frameSel = document.getElementById('frame'), showF = document.getElementById('showF');
const showM = document.getElementById('showM');
slider.max = F - 1;
for (const v of [2, 5, 10, 15, 30, 60]) {
  const o = document.createElement('option');
  o.textContent = v; o.selected = v === META.fps;
  fpsSel.appendChild(o);
}
if (Number(fpsSel.value) !== META.fps) fpsSel.value = '15';
styleSel.value = style; frameSel.value = frameMode;
if (!META.hasView) document.getElementById('frameBox').style.display = 'none';
function applyStyle() { document.body.classList.toggle('space', style === 'space'); }
applyStyle();
styleSel.addEventListener('change', () => {
  style = styleSel.value; applyStyle();
  if (META.hasView) { frameMode = style === 'space' ? 'view' : 'body'; frameSel.value = frameMode; }
  resetTarget();
  if (style === 'space') sunCamera(); else techCamera();
  draw();
});
frameSel.addEventListener('change', () => { frameMode = frameSel.value; resetTarget(); draw(); });
showF.addEventListener('change', draw);
showM.addEventListener('change', draw);

function setFrame(f) {
  frame = (f + F) % F;
  const base = frame * B;
  for (let t = 0; t < N; t++) {
    const v = (lab[base + (t >> 3)] >> (t & 7)) & 1 ? 255 : 0;
    litData[3 * t] = litData[3 * t + 1] = litData[3 * t + 2] = v;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, litBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, litData);
  slider.value = frame;
  const note = META.notes[frame] || '', u = META.units;
  const fm = Math.hypot(...vec(META.force, frame)), mm = Math.hypot(...vec(META.moment, frame));
  info.textContent = `${META.model}   |   t = ${META.time[frame].toFixed(0)} s   |   ${note}   |   ` +
    `lit ${META.lit[frame]} of ${N} triangles   |   |F| = ${fm.toPrecision(4)} ${u[0]},  ` +
    `|M| = ${mm.toPrecision(4)} ${u[1]}${u[2] ? '  (' + u[2] + ')' : ''}`;
  document.getElementById('tval').textContent = `frame ${frame + 1} / ${F}`;
  draw();
}

// ---- drawing -------------------------------------------------------------------
function lines(pts, mode, color, size, round, vp) {
  gl.useProgram(lineProg.p);
  gl.bindVertexArray(lineVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STREAM_DRAW);
  gl.uniformMatrix4fv(lineProg.u.uViewProj, false, vp);
  gl.uniform4fv(lineProg.u.uColor, color);
  gl.uniform1f(lineProg.u.uSize, size);
  gl.uniform1i(lineProg.u.uRound, round ? 1 : 0);
  gl.drawArrays(mode, 0, pts.length / 3);
}
function project(vp, p) {
  const c = [0, 1, 2, 3].map(r => vp[r] * p[0] + vp[4 + r] * p[1] + vp[8 + r] * p[2] + vp[12 + r]);
  if (c[3] <= 0) return null;
  return [(c[0] / c[3] + 1) / 2 * canvas.clientWidth, (1 - c[1] / c[3]) / 2 * canvas.clientHeight];
}
function placeLabel(id, xy) {
  const el2 = document.getElementById(id);
  if (!xy) { el2.style.display = 'none'; return; }
  el2.style.display = ''; el2.style.left = xy[0] + 'px'; el2.style.top = (xy[1] - 8) + 'px';
}
function arrow(o, v, len, thick, color, eye, vp) {
  const L = Math.hypot(...v);
  if (!(L > 0) || !(len > 0)) return null;
  const u = v.map(x => x / L), p1 = norm(cross(u, Math.abs(u[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0])), p2 = cross(u, p1);
  const m = new Float32Array([p1[0] * thick, p1[1] * thick, p1[2] * thick, 0, p2[0] * thick, p2[1] * thick, p2[2] * thick, 0,
                              u[0] * len, u[1] * len, u[2] * len, 0, o[0], o[1], o[2], 1]);
  gl.useProgram(solidProg.p);
  gl.bindVertexArray(arrowVao);
  gl.uniformMatrix4fv(solidProg.u.uModel, false, m);
  gl.uniformMatrix4fv(solidProg.u.uViewProj, false, vp);
  gl.uniform3fv(solidProg.u.uEye, eye);
  gl.uniform3fv(solidProg.u.uColor, color);
  gl.drawArrays(gl.TRIANGLES, 0, arrowPts.length / 3);
  return [o[0] + 1.06 * len * u[0], o[1] + 1.06 * len * u[1], o[2] + 1.06 * len * u[2]];
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  const space = style === 'space';
  gl.clearColor(space ? 0 : 1, space ? 0 : 1, space ? 0 : 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const Rr = R(), dist = zoom * Rr, fovy = space ? 0.8 : 0.6;
  const eye = [target[0] + dist * Math.cos(el) * Math.cos(az), target[1] + dist * Math.cos(el) * Math.sin(az),
               target[2] + dist * Math.sin(el)];
  const cam = lookAt(eye, target, [0, 0, 1]);
  const vp = mul(perspective(fovy, w / h, dist * 0.01, dist * 20), cam.m);
  const V = rot(frame);
  const sun = apply(V, vec(META.sun, frame));

  if (space) {
    // sky frame = view frame: render -> sky is V (body) or identity (view)
    const Vb = META.view.slice(9 * frame, 9 * frame + 9);
    const toSky = useView() ? I3 : Vb;
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(skyProg.p);
    gl.bindVertexArray(emptyVao);
    gl.uniform3fv(skyProg.u.uCamX, cam.x); gl.uniform3fv(skyProg.u.uCamY, cam.y); gl.uniform3fv(skyProg.u.uCamZ, cam.z);
    gl.uniform3fv(skyProg.u.uSun, sun);
    gl.uniform1f(skyProg.u.uTan, Math.tan(fovy / 2)); gl.uniform1f(skyProg.u.uAspect, w / h);
    gl.uniformMatrix3fv(skyProg.u.uSky, true, toSky);           // row-major
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(starProg.p);
    gl.bindVertexArray(starVao);
    gl.uniformMatrix4fv(starProg.u.uViewProj, false, vp);
    gl.uniformMatrix3fv(starProg.u.uSkyToRender, false, toSky);  // transpose of toSky
    gl.uniform1f(starProg.u.uDpr, dpr);
    gl.drawArrays(gl.POINTS, 0, starData.length / 4);
    gl.disable(gl.BLEND);
  }

  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(meshProg.p);
  gl.bindVertexArray(meshVao);
  const u = meshProg.u;
  gl.uniform3fv(u.uLo, META.lo); gl.uniform3fv(u.uSpan, META.span);
  gl.uniform3fv(u.uSun, sun); gl.uniform3fv(u.uEye, eye);
  gl.uniformMatrix4fv(u.uViewProj, false, vp);
  gl.uniform1i(u.uStyle, space ? 1 : 0);
  for (let k = 0; k < M; k++) {
    const [first, count] = META.meshes[k];
    if (!count) continue;
    const mat = META.materials[META.material[k]];
    gl.uniform3fv(u.uBase, mat[0]); gl.uniform1f(u.uKs, mat[1]); gl.uniform1f(u.uShine, mat[2]);
    gl.uniform1i(u.uPattern, mat[3]);
    gl.uniformMatrix4fv(u.uModel, false, modelMatrix(frame, k));
    gl.drawArrays(gl.TRIANGLES, 3 * first, 3 * count);
  }

  const c = C(), L = 0.5 * Rr, o = apply(V, [0, 0, 0]);
  const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(a => apply(V, a));
  if (!space) {   // body axes, path of the sun, ray to the sun, the sun
    const cols = [[0.78, 0.16, 0.16, 1], [0.18, 0.49, 0.2, 1], [0.08, 0.4, 0.75, 1]];
    ax.forEach((a, i) => lines([...o, o[0] + L * a[0], o[1] + L * a[1], o[2] + L * a[2]], gl.LINES, cols[i], 1, false, vp));
    const path = [];
    for (let f = 0; f < F; f++) { const s = apply(rot(f), vec(META.sun, f)); for (let i = 0; i < 3; i++) path.push(c[i] + 1.05 * Rr * s[i]); }
    const sp = [c[0] + 1.05 * Rr * sun[0], c[1] + 1.05 * Rr * sun[1], c[2] + 1.05 * Rr * sun[2]];
    lines(path, gl.LINE_STRIP, [1, 0.55, 0, 0.4], 1, false, vp);
    lines([...c, ...sp], gl.LINES, [1, 0.55, 0, 1], 1, false, vp);
    lines(sp, gl.POINTS, [1, 0.55, 0, 1], 16 * dpr, true, vp);
  }
  ['lx', 'ly', 'lz'].forEach((id, i) =>
    placeLabel(id, space ? null : project(vp, [o[0] + 1.08 * L * ax[i][0], o[1] + 1.08 * L * ax[i][1], o[2] + 1.08 * L * ax[i][2]])));

  // force and moment, from the moment reference point, drawn over the model
  gl.disable(gl.DEPTH_TEST);
  const ref = apply(V, META.mref), thick = 0.012 * Rr;
  const fv = apply(V, vec(META.force, frame)), mv = apply(V, vec(META.moment, frame));
  const tipF = showF.checked && META.fmax > 0 ?
    arrow(ref, fv, 0.9 * Rr * Math.hypot(...fv) / META.fmax, thick, [0.85, 0.11, 0.47], eye, vp) : null;
  const tipM = showM.checked && META.mmax > 0 ?
    arrow(ref, mv, 0.9 * Rr * Math.hypot(...mv) / META.mmax, thick, [0.0, 0.64, 0.71], eye, vp) : null;
  placeLabel('lF', tipF && project(vp, tipF));
  placeLabel('lM', tipM && project(vp, tipM));
}

function tick(now) {
  const dpr = window.devicePixelRatio || 1;
  if (playing && now - last >= 1000 / Number(fpsSel.value)) { last = now; setFrame(frame + 1); }
  else if (canvas.width !== Math.round(canvas.clientWidth * dpr) ||
           canvas.height !== Math.round(canvas.clientHeight * dpr)) draw();   // resized without an event
  requestAnimationFrame(tick);
}
function toggle() { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; }
playBtn.addEventListener('click', toggle);
slider.addEventListener('input', () => setFrame(Number(slider.value)));
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  if (e.code === 'ArrowRight') setFrame(frame + 1);
  if (e.code === 'ArrowLeft') setFrame(frame - 1);
});
window.addEventListener('resize', draw);
if (style === 'space') sunCamera();
setFrame(0);
requestAnimationFrame(tick);
window.player = { setFrame, draw, gl, N, F, setStyle: s => { styleSel.value = s; styleSel.dispatchEvent(new Event('change')); },
                  setFrameMode: m => { frameSel.value = m; frameSel.dispatchEvent(new Event('change')); } };
})().catch(e => { document.getElementById('info').textContent = 'Cannot show the animation: ' + e; });
</script>
</body></html>
"""

if __name__ == '__main__':
    main()
