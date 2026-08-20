# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Negative controls for `soft_depth`, written from the defect they now catch.

WHAT WAS WRONG. `soft_depth` evaluates every pixel against every face. Outside a triangle the
barycentric weights are negative, so the perspective-correct denominator `inv` crosses zero and
`1.0 / inv.clamp(min=1e-9)` returned 1e9. Those pixels entered the softmin's weighted mean, and
`cov = sigmoid(d / tau)` cannot remove them: the sigmoid is 0.5 ON the triangle edge whatever
`tau` is, so a pixel just outside contributes half of 1e9.

WHY IT SURVIVED. Nothing raised, the coverage mask stayed correct, and the depth map still had
structure -- it was merely off by eight orders of magnitude. On ANNY at 256x256 the returned
depth was 2.0e5 .. 4.5e8 for a body 1.7 m tall (about a doorway) viewed from 5 m (about a car
length). Inside the triangles the same interpolation was correct to 3 decimal places, so every
component tested alone would have passed.

THE CONTROLS. Each breaks one thing and asserts `soft_depth` says so. Control 1 runs the
RETRACTED expression and asserts it FAILS, because a fix with no failing negative control has
certified the defect rather than removed it.

Run:  python test_depth_term.py
"""

import math
import sys

import torch

from depth_term import soft_depth
from silhouette import Camera, _inside_distance

W = H = 64
FOV = 45.0
F = (W / 2) / math.tan(math.radians(FOV) / 2)


def cam(view=None):
    return Camera(width=W, height=H, fx=F, fy=F, cx=W / 2, cy=H / 2,
                  view=torch.eye(4, dtype=torch.float64) if view is None else view)


def quad(z0, z1):
    """A wall across the view, near corners at depth z0 and far corners at z1.

    z0 == z1 is parallel to the image plane, where screen-linear and perspective-correct
    interpolation agree. z0 != z1 is the case that separates them.
    """
    v = torch.tensor([[-1.0, -1.0, z0], [1.0, -1.0, z1], [1.0, 1.0, z1], [-1.0, 1.0, z0]],
                     dtype=torch.float64)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def covered(depth, weight, frac=0.5):
    m = weight > weight.max() * frac
    return depth[m], int(m.sum())


CONTROLS = []


def control(name):
    def deco(fn):
        CONTROLS.append((name, fn))
        return fn
    return deco


# --- 1. the retracted expression, asserted to fail -------------------------------------

@control("the retracted unbounded 1/inv returns 1e9 where the fix returns the geometry")
def _retracted_fails():
    # THE GEOMETRY MATTERS, and two earlier versions of this control got it wrong.
    #
    # `inv` is the interpolated 1/z of the triangle's PLANE, so it reaches zero on that plane's
    # vanishing line. A wall parallel to the image plane has a constant `inv` and no zero at
    # all -- that version measured 3.0000000000000013. A gently slanted wall puts the vanishing
    # line off screen -- that version measured 20.26. Both would have certified the defect.
    #
    # A STEEP face is what triggers it, and on a real body that is not an exotic case: every
    # near-silhouette face grazes the camera, so ANNY's own mesh at 2.7 .. 3.3 m reached 1e9.
    v, f = quad(0.6, 60.0)
    c = cam()
    tri2d = c.project(v)[f]
    v_cam = (c.view.to(v) @ torch.cat([v, torch.ones_like(v[:, :1])], -1).T).T[:, :3]
    z_vert = v_cam[:, 2].clamp(min=1e-4)[f]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    px = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1)
    a, b, cc = tri2d[:, 0], tri2d[:, 1], tri2d[:, 2]

    def cr(u, w):
        return u[..., 0] * w[..., 1] - u[..., 1] * w[..., 0]

    area = cr(b - a, cc - a)[:, None]
    wa = cr(cc[:, None] - b[:, None], px[None] - b[:, None]) / area
    wb = cr(a[:, None] - cc[:, None], px[None] - cc[:, None]) / area
    wc = 1.0 - wa - wb
    inv = wa / z_vert[:, 0, None] + wb / z_vert[:, 1, None] + wc / z_vert[:, 2, None]
    z_bad = 1.0 / inv.clamp(min=1e-9)                       # <- the retracted line
    assert float(inv.min()) < 0, "inv never crossed zero; this control has stopped testing"
    assert float(z_bad.max()) > 1e6, (
        f"the old formula did not blow up ({float(z_bad.max()):.3e}); "
        "this control has stopped testing")

    # The same geometry through the shipped path.
    d, wt = soft_depth(v, f, c)
    vals, n = covered(d, wt)
    hi = float(vals.max())
    assert hi <= 60.0 + 1e-6, f"the fix still returns {hi:.3e} on a face grazing the camera"
    return (f"unbounded {float(z_bad.max()):.3e} against bounded {hi:.3f} "
            f"on identical geometry ({n} px)")


# --- 2. the properties the fix claims ---------------------------------------------------

@control("a wall at z=3 renders depth 3, not 1e9")
def _flat_wall():
    v, f = quad(3.0, 3.0)
    d, wt = soft_depth(v, f, cam())
    vals, n = covered(d, wt)
    assert n > 100, f"only {n} covered pixels; the camera or projection is wrong"
    err = float((vals - 3.0).abs().max())
    assert err < 1e-6, f"depth off by {err} on a wall parallel to the image plane"
    return f"{n} px, max error {err:.2e}"


@control("depth never leaves the range the geometry occupies")
def _within_geometry():
    v, f = quad(2.0, 6.0)
    d, wt = soft_depth(v, f, cam())
    # Over COVERED pixels. Where nothing is drawn `den` underflows the 1e-9 clamp and the
    # ratio is arbitrary, which is exactly why `weight` is returned alongside and why the
    # affine alignment is solved over the mask rather than the frame.
    vals, n = covered(d, wt)
    lo, hi = float(vals.min()), float(vals.max())
    assert 2.0 - 1e-6 <= lo and hi <= 6.0 + 1e-6, f"depth {lo:.3f}..{hi:.3f} outside 2..6"
    return f"{n} px, {lo:.3f} .. {hi:.3f} inside 2 .. 6"


@control("the interpolation is perspective-correct, not screen-linear")
def _perspective():
    # The property IS the definition: 1/z is linear across the screen, z is not. Measuring it
    # as a straightness test avoids the first version's mistake, which compared the centre
    # pixel against a harmonic mean -- the screen centre is not the geometric midpoint of a
    # slanted span, so that number was never going to be the one to check.
    v, f = quad(2.0, 6.0)
    d, wt = soft_depth(v, f, cam())
    row, wrow = d[H // 2], wt[H // 2]
    m = wrow > wrow.max() * 0.9
    x = torch.arange(W, dtype=torch.float64)[m]
    z = row[m]
    assert int(m.sum()) > 20, f"only {int(m.sum())} covered pixels in the row"

    def r2(y):
        A = torch.stack([x, torch.ones_like(x)], -1)
        res = y - A @ torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
        return 1.0 - float(res.var() / y.var())

    r2_inv, r2_z = r2(1.0 / z), r2(z)
    assert r2_inv > 0.9999, f"1/z is not linear across the screen (R2 {r2_inv:.6f})"
    assert r2_z < r2_inv, f"z is as straight as 1/z ({r2_z:.6f}); this is screen-linear"
    return f"R2 of 1/z {r2_inv:.6f} against R2 of z {r2_z:.6f}"


@control("depth over the body does not track tau across a decade")
def _tau_invariance():
    # The symptom that named this bug: with the unbounded formula, sharpening the edge changed
    # how much 1e9 leaked in, so the depth moved with tau. It is an edge softness. It must move
    # the boundary and leave the interior alone.
    v, f = quad(3.0, 4.0)
    got = []
    for tau in (0.3, 1.0, 3.0):
        d, wt = soft_depth(v, f, cam(), tau=tau)
        vals, _ = covered(d, wt, frac=0.9)
        got.append(float(vals.mean()))
    spread = max(got) - min(got)
    assert spread < 0.05, f"interior depth moved {spread:.3f} across tau {got}"
    return f"tau 0.3/1.0/3.0 -> {', '.join(f'{g:.4f}' for g in got)}"


@control("gradients still reach the vertices")
def _differentiable():
    v, f = quad(3.0, 4.0)
    v = v.clone().requires_grad_(True)
    d, wt = soft_depth(v, f, cam())
    d.sum().backward()
    g = float(v.grad.abs().sum())
    assert g > 0 and math.isfinite(g), f"gradient is {g}; the clamp has cut the graph"
    return f"|grad| {g:.4f}"


@control("a body at ANNY's scale renders at ANNY's depth")
def _anny_scale():
    # 1.7 m tall, about a doorway, at 5 m, about a car length. The corpus run reported
    # 2.0e5 .. 4.5e8 here.
    torch.manual_seed(0)
    v, f = quad(4.5, 5.5)
    v = torch.cat([v, v + torch.tensor([0.05, 0.85, 0.1], dtype=torch.float64)])
    f = torch.cat([f, f + 4])
    d, wt = soft_depth(v, f, cam())
    vals, n = covered(d, wt)
    lo, hi = float(vals.min()), float(vals.max())
    assert 4.0 < lo and hi < 6.5, f"depth {lo:.3e} .. {hi:.3e} for a body at 4.5 .. 5.6 m"
    return f"{n} px, {lo:.3f} .. {hi:.3f} m (about a car length away)"


def main():
    fails = []
    for name, fn in CONTROLS:
        try:
            detail = fn()
            print(f"  ok   {name}\n         {detail}")
        except AssertionError as e:
            fails.append(name)
            print(f"  FAIL {name}\n         {e}")
    print()
    if fails:
        print(f"{len(fails)} of {len(CONTROLS)} controls failed.")
        return 1
    print(f"{len(CONTROLS)} controls, each failing for its own reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
