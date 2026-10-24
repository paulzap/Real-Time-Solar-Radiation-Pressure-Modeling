"""
animate_srp.py
==============
Plays a recording written by SolaRT FrameRecorder (demo section 17, or any
program that uses FrameRecorder) as a self-contained HTML page: the satellite
with its moving parts, illuminated and shadowed triangles and the direction to
the sun, played in real time; play / pause, a time slider, the camera turns
with the mouse while the animation runs.

    python animate_srp.py <recording folder> [--every N] [--fps F]
                          [--out page.html] [--no-open]

The page draws with WebGL 2 in the browser and needs nothing else (no
internet, no plotting package): the geometry is stored once, each frame adds
only the motion of every mesh and one bit per triangle, so models of a
million triangles play smoothly. Only numpy is needed here.

For display and visual checking: the numbers come from the library, the page
only draws them. Recording format: docs/USAGE.md ("Animation").

Colors as in visualize3d.py: gold - illuminated, gray - shadowed. The scene
is lit from the sun, so a face whose visible side is turned away from it is
darker.
"""

import argparse
import base64
import gzip
import json
import os
import sys
import webbrowser

import numpy as np


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


def scene_bounds(geometry, meshes, frames):
    """Centre and radius of everything drawn, over all frames."""
    corners = []
    view = frames['head'][:, 10:19].reshape(-1, 3, 3)
    for k, mesh in enumerate(meshes):
        if mesh['count'] == 0:
            continue
        p = geometry[mesh['first']:mesh['first'] + mesh['count']].reshape(-1, 3).astype(np.float64)
        lo, hi = p.min(axis=0), p.max(axis=0)
        box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        mot = frames['motion'][:, k].astype(np.float64).reshape(-1, 3, 4)
        q = np.einsum('fij,bj->fbi', mot[:, :, :3], box) + mot[:, None, :, 3]
        corners.append(np.einsum('fij,fbj->fbi', view, q).reshape(-1, 3))
    allc = np.vstack(corners)
    lo, hi = allc.min(axis=0), allc.max(axis=0)
    return 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
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

    centre, radius = scene_bounds(geometry, meshes, frames)
    # Vertices quantised to 16 bits over the bounding box of the file pose
    # (0.3 mm on a 20 m model): half the size, invisible on the screen.
    p = geometry.reshape(-1, 3).astype(np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    quant = np.round((p - lo) / span * 65535.0).astype('<u2')

    labels = frames['label'] > 0
    meta = dict(
        model=os.path.basename(scene.get('model', '').replace('\\', '/')), N=n, F=len(frames),
        meshes=[[m['first'], m['count']] for m in meshes],
        lo=lo.tolist(), span=span.tolist(), centre=centre.tolist(), radius=radius,
        time=frames['head'][:, 0].tolist(), sun=frames['head'][:, 1:4].ravel().tolist(),
        force=np.linalg.norm(frames['head'][:, 4:7], axis=1).tolist(),
        view=frames['head'][:, 10:19].ravel().tolist(),
        lit=labels.sum(axis=1).tolist(), notes=notes, fps=args.fps)
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
  #info { position: fixed; top: 8px; left: 12px; right: 270px; font-size: 14px; color: #222; }
  #help { position: fixed; top: 8px; right: 12px; font-size: 12px; color: #666; text-align: right; }
  #bar { position: fixed; left: 12px; right: 12px; bottom: 10px; display: flex; gap: 12px;
         align-items: center; font-size: 13px; background: rgba(255,255,255,0.85);
         padding: 6px 10px; border-radius: 6px; }
  #slider { flex: 1; }
  button { min-width: 64px; }
  .lab { position: fixed; font-size: 13px; font-weight: 600; pointer-events: none; }
</style></head>
<body>
<canvas id="gl"></canvas>
<div id="info"></div>
<div id="help">drag: turn &nbsp; wheel: zoom &nbsp; right drag: move<br>space: play / pause &nbsp; &larr; &rarr;: frame</div>
<div id="bar"><button id="play">Play</button><input id="slider" type="range" min="0" value="0" step="1">
  <span id="tval"></span><label>speed <select id="fps"></select> frames/s</label></div>
<div class="lab" id="lx" style="color:#c62828">x</div>
<div class="lab" id="ly" style="color:#2e7d32">y</div>
<div class="lab" id="lz" style="color:#1565c0">z</div>
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

// ---- shaders -------------------------------------------------------------
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
  return p;
}
const meshProg = program(`#version 300 es
layout(location = 0) in vec3 q;          // vertex at the file pose, 16-bit
layout(location = 1) in float lit;
uniform vec3 uLo, uSpan;
uniform mat4 uModel, uViewProj;
out vec3 vPos; flat out float vLit;
void main() {
  vec4 w = uModel * vec4(uLo + q * uSpan, 1.0);
  vPos = w.xyz; vLit = lit;
  gl_Position = uViewProj * w;
}`, `#version 300 es
precision highp float;
in vec3 vPos; flat in float vLit;
uniform vec3 uSun, uEye;
out vec4 color;
void main() {
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  if (dot(n, uEye - vPos) < 0.0) n = -n;          // the side the camera sees
  float d = max(dot(n, uSun), 0.0);
  vec3 base = vLit > 0.5 ? vec3(1.0, 0.843, 0.0) : vec3(0.42, 0.42, 0.44);
  color = vec4(base * (0.55 + 0.5 * d), 1.0);
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

// ---- buffers -------------------------------------------------------------
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
gl.bindVertexArray(null);

// ---- matrices (column-major, as WebGL wants) -----------------------------
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
function lookAt(e, t, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(e, t)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
                           -dot(x, e), -dot(y, e), -dot(z, e), 1]);
}
// row-major view V (3x3) times motion [R|t] (3x4) -> column-major 4x4
function modelMatrix(f, k) {
  const v = 9 * f, o = (f * M + k) * 12, r = new Float32Array(16);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let c = 0; c < 3; c++) s += META.view[v + 3 * i + c] * mot[o + 4 * c + j];
    r[j * 4 + i] = s;
  }
  r[15] = 1;
  return r;
}
function toView(f, x) {
  const V = META.view, v = 9 * f;
  return [0, 1, 2].map(i => V[v + 3 * i] * x[0] + V[v + 3 * i + 1] * x[1] + V[v + 3 * i + 2] * x[2]);
}

// ---- camera --------------------------------------------------------------
const C = META.centre, R = META.radius;
let az = 0.8, el = 0.45, dist = 4.2 * R;
const target = C.slice();
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
    const s = dist * 0.0015, ca = Math.cos(az), sa = Math.sin(az), se = Math.sin(el), ce = Math.cos(el);
    const right = [-sa, ca, 0], up = [-se * ca, -se * sa, ce];
    for (let i = 0; i < 3; i++) target[i] += (-dx * right[i] + dy * up[i]) * s;
  }
  draw();
});
canvas.addEventListener('wheel', e => { e.preventDefault(); dist *= Math.exp(e.deltaY * 0.001); draw(); },
                        { passive: false });

// ---- frames --------------------------------------------------------------
let frame = 0, playing = false, last = 0;
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
const fpsSel = document.getElementById('fps');
slider.max = F - 1;
for (const v of [2, 5, 10, 15, 30, 60]) {
  const o = document.createElement('option');
  o.textContent = v; o.selected = v === META.fps;
  fpsSel.appendChild(o);
}
if (fpsSel.selectedIndex < 0 || Number(fpsSel.value) !== META.fps) fpsSel.value = '15';

const sunPath = [];
for (let f = 0; f < F; f++) {
  const s = toView(f, META.sun.slice(3 * f, 3 * f + 3));
  for (let i = 0; i < 3; i++) sunPath.push(C[i] + 1.05 * R * s[i]);
}

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
  const note = META.notes[frame] || '';
  document.getElementById('info').textContent =
    `${META.model}   |   t = ${META.time[frame].toFixed(0)} s   |   ${note}   |   ` +
    `lit ${META.lit[frame]} of ${N} triangles,  |F| = ${META.force[frame].toPrecision(4)}`;
  document.getElementById('tval').textContent = `frame ${frame + 1} / ${F}`;
  draw();
}

function lines(pts, mode, color, size, round, vp) {
  gl.useProgram(lineProg);
  gl.bindVertexArray(lineVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STREAM_DRAW);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, 'uViewProj'), false, vp);
  gl.uniform4fv(gl.getUniformLocation(lineProg, 'uColor'), color);
  gl.uniform1f(gl.getUniformLocation(lineProg, 'uSize'), size);
  gl.uniform1i(gl.getUniformLocation(lineProg, 'uRound'), round ? 1 : 0);
  gl.drawArrays(mode, 0, pts.length / 3);
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  const eye = [target[0] + dist * Math.cos(el) * Math.cos(az), target[1] + dist * Math.cos(el) * Math.sin(az),
               target[2] + dist * Math.sin(el)];
  const vp = mul(perspective(0.6, w / h, dist * 0.01, dist * 20), lookAt(eye, target, [0, 0, 1]));
  const sun = toView(frame, META.sun.slice(3 * frame, 3 * frame + 3));

  gl.useProgram(meshProg);
  gl.bindVertexArray(meshVao);
  gl.uniform3fv(gl.getUniformLocation(meshProg, 'uLo'), META.lo);
  gl.uniform3fv(gl.getUniformLocation(meshProg, 'uSpan'), META.span);
  gl.uniform3fv(gl.getUniformLocation(meshProg, 'uSun'), sun);
  gl.uniform3fv(gl.getUniformLocation(meshProg, 'uEye'), eye);
  gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, 'uViewProj'), false, vp);
  const uModel = gl.getUniformLocation(meshProg, 'uModel');
  for (let k = 0; k < M; k++) {
    const [first, count] = META.meshes[k];
    if (!count) continue;
    gl.uniformMatrix4fv(uModel, false, modelMatrix(frame, k));
    gl.drawArrays(gl.TRIANGLES, 3 * first, 3 * count);
  }

  // body axes, path of the sun, ray to the sun, the sun
  const L = 0.5 * R, o = toView(frame, [0, 0, 0]);
  const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(a => toView(frame, a));
  const cols = [[0.78, 0.16, 0.16, 1], [0.18, 0.49, 0.2, 1], [0.08, 0.4, 0.75, 1]];
  ax.forEach((a, i) => lines([...o, o[0] + L * a[0], o[1] + L * a[1], o[2] + L * a[2]], gl.LINES, cols[i], 1, false, vp));
  const sp = [C[0] + 1.05 * R * sun[0], C[1] + 1.05 * R * sun[1], C[2] + 1.05 * R * sun[2]];
  lines(sunPath, gl.LINE_STRIP, [1, 0.55, 0, 0.4], 1, false, vp);
  lines([...C, ...sp], gl.LINES, [1, 0.55, 0, 1], 1, false, vp);
  lines(sp, gl.POINTS, [1, 0.55, 0, 1], 16 * dpr, true, vp);

  ['lx', 'ly', 'lz'].forEach((id, i) => {
    const p = [o[0] + 1.08 * L * ax[i][0], o[1] + 1.08 * L * ax[i][1], o[2] + 1.08 * L * ax[i][2]];
    const c = [0, 1, 2, 3].map(r => vp[r] * p[0] + vp[4 + r] * p[1] + vp[8 + r] * p[2] + vp[12 + r]);
    const label = document.getElementById(id);
    if (c[3] <= 0) { label.style.display = 'none'; return; }
    label.style.display = '';
    label.style.left = ((c[0] / c[3] + 1) / 2 * canvas.clientWidth) + 'px';
    label.style.top = ((1 - c[1] / c[3]) / 2 * canvas.clientHeight - 8) + 'px';
  });
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
  if (e.code === 'Space') { e.preventDefault(); toggle(); }
  if (e.code === 'ArrowRight') setFrame(frame + 1);
  if (e.code === 'ArrowLeft') setFrame(frame - 1);
});
window.addEventListener('resize', draw);
setFrame(0);
requestAnimationFrame(tick);
window.player = { setFrame, draw, gl, N, F };   // for checks from the browser console
})().catch(e => { document.getElementById('info').textContent = 'Cannot show the animation: ' + e; });
</script>
</body></html>
"""

if __name__ == '__main__':
    main()
