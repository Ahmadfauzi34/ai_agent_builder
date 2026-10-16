// 3D geometry builders for burn-math (used by bin/geom_build.mjs).
//
// Canonical representation: homogeneous 4x4 matrices, ROW-MAJOR internally
// (m[r*4+c]), because affine chains compose by plain matrix multiplication:
//   T2(T1(x)) = (T2·T1) x   — one matrix per chain, exactly one linear layer.
//
// Conventions (documented, tested, fixed):
// - angles in DEGREES everywhere (agents think in degrees, not radians);
// - right-handed coords, ACTIVE rotations (vectors rotate, frame fixed);
// - euler order 'ZYX' (default): R = Rz(yaw) · Ry(pitch) · Rx(roll), intrinsic;
// - quaternion (w,x,y,z), normalized on input, same handedness as matrices;
// - engine weights are COLUMN-MAJOR: use mat4ToColumnMajor() when emitting.
// - engine has no perspective divide: projection matrices can be BUILT here,
//   but the x/w divide stays host-side (documented limit, not a silent gap).
//
// Every builder is pure (no shared state) — each op's behavior is local and
// independently testable, per the atomic-geometry principle.

export const deg2rad = (d) => (d * Math.PI) / 180;

export function mat4Identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

// C = A·B (row-major). Applies B first, then A: x -> A(B(x)).
export function mat4Mul(A, B) {
  const C = new Array(16).fill(0);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[r * 4 + k] * B[k * 4 + c];
      C[r * 4 + c] = s;
    }
  return C;
}

function rotXRad(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
}
function rotYRad(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
}
function rotZRad(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export const rotX4 = (deg) => rotXRad(deg2rad(deg));
export const rotY4 = (deg) => rotYRad(deg2rad(deg));
export const rotZ4 = (deg) => rotZRad(deg2rad(deg));

// Rodrigues rotation about an arbitrary axis (normalized internally).
export function rotAxisAngle4(ax, ay, az, deg) {
  const n = Math.hypot(ax, ay, az);
  if (!(n > 0)) throw new Error('rotAxisAngle: axis must be non-zero');
  const [x, y, z] = [ax / n, ay / n, az / n];
  const a = deg2rad(deg), c = Math.cos(a), s = Math.sin(a), t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
    0, 0, 0, 1,
  ];
}

// Euler -> matrix. order 'ZYX': R = Rz(yaw)·Ry(pitch)·Rx(roll).
export function euler4(roll, pitch, yaw, order = 'ZYX') {
  const rx = rotXRad(deg2rad(roll)), ry = rotYRad(deg2rad(pitch)), rz = rotZRad(deg2rad(yaw));
  const M = { X: rx, Y: ry, Z: rz };
  const seq = order.toUpperCase().split('');
  if (seq.length !== 3 || new Set(seq).size !== 3 || seq.some((k) => !M[k]))
    throw new Error(`euler: order must be a permutation of XYZ, got '${order}'`);
  // 'ZYX' means apply X first: R = M[Z]·M[Y]·M[X]
  return mat4Mul(M[seq[0]], mat4Mul(M[seq[1]], M[seq[2]]));
}

function quatMul(a, b) {
  // Hamilton product a*b (apply b first, then a). [w,x,y,z]
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}
function axisQuat(x, y, z, deg) {
  const h = deg2rad(deg) / 2, s = Math.sin(h);
  return [Math.cos(h), x * s, y * s, z * s];
}
// Euler (same convention as euler4) -> quaternion [w,x,y,z].
export function quatFromEuler(roll, pitch, yaw, order = 'ZYX') {
  const q = { X: axisQuat(1, 0, 0, roll), Y: axisQuat(0, 1, 0, pitch), Z: axisQuat(0, 0, 1, yaw) };
  const seq = order.toUpperCase().split('');
  if (seq.length !== 3 || new Set(seq).size !== 3 || seq.some((k) => !q[k]))
    throw new Error(`quatFromEuler: order must be a permutation of XYZ, got '${order}'`);
  return quatMul(q[seq[0]], quatMul(q[seq[1]], q[seq[2]]));
}

// Quaternion -> 4x4 rotation matrix (normalizes input).
export function quatToMat4(w, x, y, z) {
  const n = Math.hypot(w, x, y, z);
  if (!(n > 0)) throw new Error('quatToMat4: zero quaternion');
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    1 - (yy + zz), xy - wz, xz + wy, 0,
    xy + wz, 1 - (xx + zz), yz - wx, 0,
    xz - wy, yz + wx, 1 - (xx + yy), 0,
    0, 0, 0, 1,
  ];
}

export function translate4(tx, ty, tz) {
  return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1];
}

export function scale4(sx, sy, sz) {
  return [sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1];
}

// Row-major 4x4 -> column-major 16-array for linear(4,4) weights:
// engine computes y_i = Σ_j w[i + j*4]·x_j, so w[i + j*4] = m[i*4+j].
export function mat4ToColumnMajor(m) {
  const w = new Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) w[i + j * 4] = m[i * 4 + j];
  return w;
}

export const homogenize = ([x, y, z]) => [x, y, z, 1];
export const dehomogenize = ([x, y, z, w]) => [x / w, y / w, z / w];

// Apply a row-major 4x4 to a 3-vector in f64 (host-side reference).
export function applyMat4(m, v3) {
  const [x, y, z, w] = homogenize(v3);
  const o = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) o[i] = m[i * 4] * x + m[i * 4 + 1] * y + m[i * 4 + 2] * z + m[i * 4 + 3] * w;
  return dehomogenize(o);
}

// Compose a chain of 4x4s into one matrix (first op applied first).
export function composeChain(mats) {
  return mats.reduce((acc, m) => mat4Mul(m, acc), mat4Identity());
}

// ---- normals (for lighting) ----
//
// Normals are direction vectors: they must NOT go through the projection
// matrix, and under non-uniform scale the model matrix itself gives the
// WRONG answer. The correct transform is the inverse-transpose of the
// model's upper 3x3: N = inverse(transpose(M3)). Still linear — but it is
// derived from the model matrix, not equal to it.
//
// normalMat3(model4) -> row-major 3x3. Apply to [nx,ny,nz], then renormalize
// host-side. Defining invariant (tested): perpendicularity is preserved —
// for tangent t and normal n with n·t = 0, (N·n)·(M·t) = 0.
function mat3Inverse(m) {
  // m: row-major 3x3 [a b c; d e f; g h i]
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!(Math.abs(det) > 0)) throw new Error('mat3Inverse: singular matrix');
  const inv = [
    A / det, (c * h - b * i) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, (c * d - a * f) / det,
    C / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
  return inv;
}

export function normalMat3(model4) {
  const m3 = [
    model4[0], model4[1], model4[2],
    model4[4], model4[5], model4[6],
    model4[8], model4[9], model4[10],
  ];
  const inv = mat3Inverse(m3);
  // transpose of inverse
  return [
    inv[0], inv[3], inv[6],
    inv[1], inv[4], inv[7],
    inv[2], inv[5], inv[8],
  ];
}

// Apply a row-major 3x3 to a 3-vector (f64, host-side).
export function applyMat3(m3, [x, y, z]) {
  return [
    m3[0] * x + m3[1] * y + m3[2] * z,
    m3[3] * x + m3[4] * y + m3[5] * z,
    m3[6] * x + m3[7] * y + m3[8] * z,
  ];
}

// ---- near-plane triangle clipping (clip space, host-side) ----
//
// Sutherland–Hodgman against the half-space z >= -w (the near plane).
// Operates on homogeneous clip coords, so interpolation is exact and the
// perspective divide afterwards is safe. Vertices exactly on the plane
// count as inside.
//
// Returns {triangles: [[c0,c1,c2], ...], spansCamera}: each triangle is 3
// clip-space vertices. spansCamera=true when any input vertex has w <= 0 —
// the edge crosses the camera plane singularity and is NOT clipped here;
// the caller must flag it (like projectClip flags behind vertices), never
// silently project it.
export function clipTriangleNear(c0, c1, c2) {
  const verts = [c0, c1, c2];
  if (verts.some((v) => !(v[3] > 0))) return { triangles: [], spansCamera: true };
  const inside = (p) => p[2] + p[3] >= 0;
  const intersect = (A, B) => {
    const dz = B[2] - A[2], dw = B[3] - A[3];
    const denom = dz + dw;
    // crossing edge guarantees denom != 0 (one side in, one out)
    const t = -(A[3] + A[2]) / denom;
    return A.map((a, k) => a + t * (B[k] - a));
  };
  let poly = verts;
  // single half-space pass
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const curr = poly[i], next = poly[(i + 1) % poly.length];
    const cIn = inside(curr), nIn = inside(next);
    if (cIn && nIn) out.push(next);
    else if (cIn && !nIn) out.push(intersect(curr, next));
    else if (!cIn && nIn) { out.push(intersect(curr, next)); out.push(next); }
  }
  poly = out;
  // fan triangulation
  const triangles = [];
  for (let i = 1; i + 1 < poly.length; i++) triangles.push([poly[0], poly[i], poly[i + 1]]);
  return { triangles, spansCamera: false };
}

// Apply a row-major 4x4 to a 3-vector in f64, WITHOUT the perspective
// divide: returns clip coordinates [x, y, z, w] (host-side reference for
// the engine path, where the engine's linear layer produces these 4 values
// and the divide happens host-side).
export function applyMat4Clip(m, v3) {
  const [x, y, z, w] = homogenize(v3);
  const o = [0, 0, 0, 0];
  for (let i = 0; i < 4; i++) o[i] = m[i * 4] * x + m[i * 4 + 1] * y + m[i * 4 + 2] * z + m[i * 4 + 3] * w;
  return o;
}

// Camera view matrix (world -> camera), row-major, ACTIVE convention.
// Camera looks down its -z; eye/target/up in world coords.
export function lookAt4(eye, target, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (v) => { const n = Math.hypot(v[0], v[1], v[2]); if (!(n > 0)) throw new Error('lookAt4: zero-length vector'); return [v[0] / n, v[1] / n, v[2] / n]; };
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return [
    x[0], x[1], x[2], -dot(x, eye),
    y[0], y[1], y[2], -dot(y, eye),
    z[0], z[1], z[2], -dot(z, eye),
    0, 0, 0, 1,
  ];
}

// OpenGL-style perspective projection (row-major, ACTIVE). Maps camera
// space to clip coords with w_clip = -z_cam; points in FRONT of the camera
// (z_cam < 0) get w > 0. NDC z lands in [-1, 1].
export function perspective4(fovDeg, aspect, near, far) {
  if (!(near > 0 && far > near)) throw new Error('perspective4: need 0 < near < far');
  const f = 1 / Math.tan(deg2rad(fovDeg) / 2);
  const a = aspect > 0 ? aspect : 1;
  return [
    f / a, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
    0, 0, -1, 0,
  ];
}
