"""Differentiable soft depth: the forward that pairs with Marigold, and the third opinion.

## Where this sits

Every stage of the avatar pipeline has a forward and an inverse, and most have both:

    pose -> mesh          FK + LBS (sinew-solve)      <-  AnnyInverter / LBFGS
    skeleton -> skinning  LBS apply                   <-  SkinTokens (MIT)
    phenotype -> build    624 blendshapes             <-  soft silhouette
    scene -> depth        THIS FILE                   <-  Marigold

Marigold is a learned inverse with no forward to check it against. This is that forward.

## Why depth is the right third opinion

The silhouette measured its own blind spot exactly: a depth-plus-scale change scored 0.849,
which is precisely its self-IoU floor -- perfectly blind. Depth is the axis it cannot see, so
Marigold is not merely another estimator, it is the complement.

    LBFGS vertex   sees 3D with correspondence   blind to whether correspondence is right
    silhouette     sees the outline              blind to depth and to the interior
    depth          sees the interior             blind to ABSOLUTE SCALE

Three ambiguities, each covered by another. A limb rotating about its own axis moves neither
the outline nor the vertex residual, and does move the depth field.

## The affine ambiguity, which decides whether this works at all

Marigold predicts AFFINE-INVARIANT depth: it recovers structure up to an unknown scale and
shift, because a single image cannot determine metric depth. Comparing its output to a
rendered depth buffer directly therefore measures the ambiguity rather than the fit, and
would report a large error on a perfect reconstruction.

So the alignment is solved INSIDE the objective: least squares for the (a, b) minimising
|| a*pred + b - rendered ||^2 over the shared mask, then the residual after alignment is the
loss. Two parameters, closed form, differentiable.

It has to be inside rather than a preprocessing step. Solved once against an initial guess,
the alignment would bake in that guess's error and every later step would be scored against a
stale reference -- the fit would then be optimising toward its own first mistake.

The control that matters is the one asserting this worked: scaling the target depth must
produce ZERO change in the loss. If it does not, the affine solve is absorbing real error
instead of the ambiguity, and the number is not a measurement.
"""

from __future__ import annotations

import torch

from silhouette import (Camera, _block_pixels, _morton_order, _work_items,
                        ELEM_BUDGET, influence_pad, soft_silhouette, tau_for_bleed)


def soft_depth(verts: torch.Tensor, faces: torch.Tensor, cam: Camera,
               tau: float | None = None, chunk: int | None = None,
               cull: bool = True, max_faces: int = 4096,
               budget: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable depth and coverage. Returns (depth, weight), each (H, W).

    A silhouette is the union of every face regardless of depth order, which is what let it
    skip the z-buffer. Depth cannot: a pixel's depth is the NEAREST surface, not all of them.

    A hard nearest-surface test is a discrete argmin and has no gradient, so this uses a
    softmin instead -- faces are weighted by exp(-z/beta) and the depth is their weighted
    mean. Rendering the mean rather than a hard front surface is what keeps the value
    differentiable, and it is also honest about self-occlusion: where two surfaces nearly
    coincide the value sits between them rather than snapping.

    Depth is INTERPOLATED ACROSS each face, not taken as the face mean. A per-face constant
    renders a faceted map: the first version of this did that and a wedge spanning z 4.5 to
    5.5 came back spanning 4.833 to 4.899, because two triangles can only produce two values.
    On ANNY's 36,108 faces the error would have been subtler and no less wrong.

    The interpolation is perspective-correct -- linear in 1/z, not in z. Screen-space linear
    interpolation of z is a standard and silently wrong shortcut: it is exact only for
    surfaces parallel to the image plane, which is the one case where the depth term has
    nothing to say anyway.

    `tau` defaults to half a pixel of bleed at this mesh's face count, not to a constant.

    `weight` is returned rather than folded away because it is the coverage mask, and the
    alignment below must only be solved over pixels the body actually covers. Aligning over
    empty background would fit the affine parameters to nothing.
    """
    if tau is None:
        tau = tau_for_bleed(0.5, faces.shape[0])

    from silhouette import _inside_distance
    tri2d = cam.project(verts)[faces]                          # (F, 3, 2)
    v_cam = (cam.view.to(verts) @ torch.cat(
        [verts, torch.ones_like(verts[:, :1])], -1).T).T[:, :3]
    z_vert = v_cam[:, 2].clamp(min=1e-4)[faces]                # (F, 3) per-corner depth

    # Faces are reordered for screen locality so each block covers a small rectangle. The
    # depths must ride along, or every face would be paired with another face's depth -- a
    # reordering bug that renders a plausible map of the wrong body.
    order = _morton_order(tri2d) if cull else torch.arange(tri2d.shape[0], device=tri2d.device)
    tri2d, z_vert = tri2d[order], z_vert[order]
    z_face = z_vert.mean(-1)
    pad = influence_pad(tau, tri2d.shape[0])
    # `max_faces` and `budget` are the THROUGHPUT KNOB, not tuning noise. Smaller blocks have
    # tighter bounding boxes and do less arithmetic; larger blocks launch fewer kernels. The
    # first bounds work, the second bounds overhead, and which one dominates is measured
    # rather than assumed -- see the sweep recorded in `_controls`.
    if budget is None:
        budget = ELEM_BUDGET if chunk is None else float(chunk) * cam.height * cam.width

    n_px = cam.height * cam.width
    num = torch.zeros(n_px, device=verts.device, dtype=verts.dtype)
    den = torch.zeros_like(num)
    beta = max(float(z_face.detach().std()) if z_face.numel() > 1 else 1.0, 1e-3)
    zmin = z_face.min()
    for i0, i1, y0, y1, x0, x1 in _work_items(
            tri2d, pad, cam.height, cam.width, cull=cull, budget=budget,
            max_faces=max_faces):
        px, idx = _block_pixels(y0, y1, x0, x1, cam.width, verts.device, verts.dtype)
        t = tri2d[i0:i1]                                       # (f, 3, 2)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]

        def cross(u, v):
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        area = cross(b - a, c - a)[:, None]                    # (f, 1)
        safe = torch.where(area.abs() < 1e-12, torch.full_like(area, 1e-12), area)
        P = px[None]                                           # (1, P, 2)
        wa = cross(c[:, None] - b[:, None], P - b[:, None]) / safe
        wb = cross(a[:, None] - c[:, None], P - c[:, None]) / safe
        wc = 1.0 - wa - wb
        # Perspective-correct: 1/z is what is linear in screen space.
        zc = z_vert[i0:i1]
        inv = (wa / zc[:, 0, None] + wb / zc[:, 1, None] + wc / zc[:, 2, None])
        # RETRACTED: `z_px = 1.0 / inv.clamp(min=1e-9)`, which was unbounded.
        #
        # Every pixel is evaluated against every face in reach, and OUTSIDE a triangle the
        # barycentric weights are negative. So `inv` crosses zero somewhere on the plane, the
        # clamp turns that into a depth of 1e9, and those pixels enter the weighted mean.
        #
        # `cov` cannot suppress them and no `tau` makes it. The sigmoid is 0.5 ON the edge by
        # construction, so a pixel just outside contributes 0.5 * 1e9. Measured on ANNY at
        # 256x256: inside the triangles z_px was 2.873 .. 2.922 against a true face range of
        # 2.701 .. 3.318, and correct. Over all pixels it reached 1.000e+09, and the rendered
        # depth came back 2.0e5 .. 4.5e8 for a body 1.7 m tall at 5 m. Nothing raised.
        #
        # The bound is the triangle's OWN corner depths. A perspective-correct interpolation
        # is a convex combination of the corners, so inside the triangle the result already
        # lies in [min, max] and the clamp is a no-op there -- it changes no correct pixel and
        # no correct gradient. Outside, it saturates to a depth that face could actually have.
        z_px = (1.0 / inv.clamp(min=1e-9)).clamp(
            min=zc.min(-1).values[:, None], max=zc.max(-1).values[:, None])

        d = _inside_distance(px, t)
        cov = torch.sigmoid(d / tau)
        w = cov * torch.exp(-(z_face[i0:i1, None] - zmin) / beta)
        num = num.index_add(0, idx, (w * z_px).sum(0))
        den = den.index_add(0, idx, w.sum(0))
    depth = num / den.clamp(min=1e-9)
    return depth.reshape(cam.height, cam.width), den.reshape(cam.height, cam.width)


def align_affine(pred: torch.Tensor, ref: torch.Tensor,
                 mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares (a, b) minimising || a*pred + b - ref ||^2 over `mask`.

    Closed form, so it is differentiable and adds no optimisation of its own. This is what
    removes Marigold's affine ambiguity; without it the comparison scores the ambiguity.
    """
    w = mask.reshape(-1)
    p = pred.reshape(-1)
    r = ref.reshape(-1)
    sw = w.sum().clamp(min=1e-9)
    mp = (w * p).sum() / sw
    mr = (w * r).sum() / sw
    cov = (w * (p - mp) * (r - mr)).sum()
    var = (w * (p - mp) ** 2).sum().clamp(min=1e-9)
    a = cov / var
    b = mr - a * mp
    return a, b


def depth_loss(verts: torch.Tensor, faces: torch.Tensor, target_depth: torch.Tensor,
               cam: Camera, tau: float = 1.0,
               min_coverage: float = 1e-3, report_alignment: bool = False):
    """Scale-and-shift-invariant depth residual against a predicted depth map.

    `target_depth` is Marigold's output, affine-invariant. The alignment is solved here, on
    every call, over the rendered coverage mask only.

    Returns a NaN-free scalar. Where the body covers nothing the loss is zero-by-absence,
    which the caller must notice -- a body that has left the frame produces no depth
    disagreement at all, and that reads exactly like a perfect fit. `coverage_ok` exists so
    that case is caught rather than rewarded.
    """
    rendered, weight = soft_depth(verts, faces, cam, tau=tau)
    mask = weight / weight.max().clamp(min=1e-9)
    a, b = align_affine(target_depth, rendered, mask)
    resid = (a * target_depth + b - rendered) ** 2
    loss = (mask * resid).sum() / mask.sum().clamp(min=min_coverage)
    return (loss, a, b) if report_alignment else loss


def coverage_ok(verts: torch.Tensor, faces: torch.Tensor, cam: Camera,
                tau: float = 1.0, floor: float = 0.01) -> bool:
    """Did the body cover enough of the frame for the depth loss to mean anything?

    CLAUDE.md rule 3: an unmet precondition is a FAIL, not a skip. A depth loss computed over
    four pixels is not a small measurement, it is not a measurement.
    """
    S = soft_silhouette(verts, faces, cam, tau=tau)
    return float(S.mean()) > floor


# ---------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------

def _quad(cx: float, cy: float, r: float, z: float = 5.0):
    v = torch.tensor([[cx - r, cy - r, z], [cx + r, cy - r, z],
                      [cx + r, cy + r, z], [cx - r, cy + r, z]], dtype=torch.float64)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return v, f


def _wedge(z0: float, z1: float):
    """A quad tilted in depth, so the depth map is not constant."""
    v = torch.tensor([[-0.5, -0.5, z0], [0.5, -0.5, z1],
                      [0.5, 0.5, z1], [-0.5, 0.5, z0]], dtype=torch.float64)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return v, f


def _controls() -> None:
    cam = Camera(width=48, height=48, fx=160.0, fy=160.0, cx=24.0, cy=24.0)

    v, f = _wedge(4.5, 5.5)
    D, W = soft_depth(v, f, cam, tau=0.5)
    cov = W / W.max()
    inside = cov > 0.5
    print(f"  depth renders             -> range [{float(D[inside].min()):.3f}, "
          f"{float(D[inside].max()):.3f}] over {int(inside.sum())} px")
    assert float(D[inside].max()) - float(D[inside].min()) > 0.1, \
        "tilted surface produced a flat depth map"

    # 1. THE AFFINE SOLVE ACTUALLY REMOVES THE AMBIGUITY. Scaling and shifting the target
    #    must not change the loss. If it does, the solve is absorbing real error and every
    #    number this file reports is meaningless.
    target = D.detach().clone()
    l0 = float(depth_loss(v, f, target, cam, tau=0.5))
    l1 = float(depth_loss(v, f, target * 3.7 - 2.1, cam, tau=0.5))
    print(f"  loss under affine change  -> {l0:.3e} vs {l1:.3e}")
    assert abs(l0 - l1) < 1e-6 * max(1.0, abs(l0)) + 1e-9, \
        "affine solve failed: the loss moved when only scale/shift changed"

    # 2. IT STILL SEES REAL STRUCTURE. A target with the depth gradient REVERSED is not an
    #    affine transform of the original, so the loss must rise. Without this, a solve that
    #    trivially absorbed everything would pass control 1 and look perfect.
    flipped = torch.flip(target, dims=[1])
    l2 = float(depth_loss(v, f, flipped, cam, tau=0.5))
    print(f"  loss under reversed depth -> {l2:.3e}")
    assert l2 > l0 * 10 + 1e-9, "affine solve absorbed a genuine structural error"

    # 3. GRADIENT REACHES THE VERTICES.
    vg = v.clone().requires_grad_(True)
    depth_loss(vg, f, flipped, cam, tau=0.5).backward()
    g = float(vg.grad.abs().max())
    print(f"  gradient to vertices      -> {g:.3e}")
    assert g > 1e-12, "no gradient: the depth term cannot drive a fit"

    # 4. AN EMPTY FRAME IS A FAILED PRECONDITION, NOT A PERFECT FIT.
    far, ff = _quad(0.0, 0.0, 0.5, z=4000.0)
    print(f"  body off-frame coverage   -> ok={coverage_ok(far, ff, cam, tau=0.5)}")
    assert not coverage_ok(far, ff, cam, tau=0.5), \
        "a body covering nothing was reported as measurable"
    assert coverage_ok(v, f, cam, tau=0.5), "a visible body was reported unmeasurable"


if __name__ == "__main__":
    torch.manual_seed(0)
    print("soft depth controls:")
    _controls()
    print("\nall controls passed")
