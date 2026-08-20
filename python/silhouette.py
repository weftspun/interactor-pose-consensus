"""Differentiable soft silhouette: a second opinion on the LBFGS vertex fit, and the route
from pose to SHAPE.

## The gap this closes

Keypoints give pose. They do not give shape. ANNY carries 11 phenotype parameters and 256
local changes, and NO keypoint constrains any of them -- a heavy person and a light one can
put their joints in identical places. So a keypoint fit, however precise, returns a body of
the wrong build in the right pose, and nothing in that objective can notice.

A silhouette is the cheapest signal that does constrain build. Outline area, limb thickness
and torso width are exactly what phenotype moves and exactly what a projection preserves.

## Why it is a genuine second opinion

`anny-pose-retarget-work/lbfgs_polish.py` minimises distance to target VERTICES. It is
extremely precise when right -- the logbook records 1.7e-4 mm on a same-rig fit -- and its
failure mode is not imprecision. It is CORRESPONDENCE: the solver does exactly what it was
told, having been told that the wrong vertex is the right one. The candy-wrapper failure and
the never-root-caused finger chain are both that.

A second opinion is worth having only if it fails differently, which is the lesson the
estimator panel taught by not having it. This one passes that test because IT HAS NO
CORRESPONDENCE AT ALL. It sees an outline. A vertex mislabelled as its neighbour does not
move the outline, so a correspondence error has nothing here to corrupt.

The independence runs both ways, and the second direction is why this augments rather than
replaces:

    vertex fit  is blind to whether its correspondence is right
    silhouette  is blind to everything inside the outline, and to depth

A forearm rotating about its own axis moves vertices and does not move the silhouette. So
agreement means two unrelated objectives chose the same body, and the DIRECTION of a
disagreement names the failure: silhouette-only error implicates the correspondence,
vertex-only error means the pose is unobservable from that view.

## Why the rasterizer must be soft

Coverage is a step function of vertex position -- a pixel is inside a triangle or it is not.
The derivative is therefore zero almost everywhere and undefined on the boundary, so
attaching autodiff to an ordinary rasterizer yields no silhouette gradient at all. That is
not an optimisation to defer; it is the reason SoftRas and the antialiasing op in nvdiffrast
exist.

This uses the SoftRas formulation. Each face contributes smooth occupancy decaying with
signed distance to its edges, and the silhouette is their probabilistic union:

    sigma_i = sigmoid(d_i / tau)           per-face soft coverage
    S       = 1 - prod_i (1 - sigma_i)     union, differentiable everywhere

`tau` is real and not a nuisance parameter: too sharp and the gradient vanishes again, too
soft and coverage bleeds past the body so the fit shrinks to compensate. `tau_schedule`
anneals it, stated here rather than left for a caller to rediscover.

No z-buffer, no shading, no textures, no interpolation -- a silhouette is the union of every
face regardless of depth order. That is what keeps this small enough to write in plain torch
instead of building nvdiffrast against a mismatched CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch


@dataclass
class Camera:
    """Pinhole camera. Intrinsics in pixels, extrinsics world to camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    #: (4, 4) world-to-camera. Identity means the body is already in camera space.
    view: torch.Tensor = field(default_factory=lambda: torch.eye(4, dtype=torch.float64))

    def project(self, verts: torch.Tensor) -> torch.Tensor:
        """(V, 3) world to (V, 2) pixel. Differentiable in `verts`."""
        v = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
        cam = (self.view.to(verts) @ v.T).T[:, :3]
        # Depth is clamped rather than allowed through zero. A vertex at or behind the
        # pinhole projects to infinity and takes the whole fit with it; clamping keeps the
        # gradient finite and leaves the failure visible as a bad fit instead of a NaN.
        z = cam[:, 2].clamp(min=1e-4)
        return torch.stack([self.fx * cam[:, 0] / z + self.cx,
                            self.fy * cam[:, 1] / z + self.cy], dim=-1)


def _inside_distance(px: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """Signed distance from pixels to triangles, POSITIVE INSIDE. px (P,2), tri (F,3,2).

    The minimum of the three edge half-plane distances. Inside a convex triangle all three
    are positive so the minimum is positive exactly inside, and it decays smoothly outside --
    which is what gives the sigmoid something to differentiate.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    # Orient consistently so "inside" keeps a stable sign whatever the winding.
    area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    sign = torch.where(area >= 0, 1.0, -1.0)[:, None]

    out = []
    for p0, p1 in ((a, b), (b, c), (c, a)):
        e = p1 - p0
        n = torch.stack([-e[:, 1], e[:, 0]], dim=-1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-9)
        out.append(sign * ((px[None] - p0[:, None]) * n[:, None]).sum(-1))
    return torch.stack(out, dim=0).min(dim=0).values


def chunk_for(n_pixels: int, per_face_tensors: int = 17,
              budget_bytes: float = 8.0e9, bytes_per_elem: int = 4) -> int:
    """Faces per block, so peak memory follows the IMAGE SIZE instead of a constant.

    RETRACTED: `chunk` defaulted to a bare 2048, sized for a 256x256 image and stated as such
    in `soft_silhouette`'s docstring. It has no reference to the pixel count, so it is the
    same class of constant as the old `tau`: correct where it was measured and wrong
    elsewhere, by a factor of the thing it left out.

    Every block materialises several (chunk, P) intermediates -- the three barycentrics, the
    interpolated 1/z, the three edge distances, the coverage and the weight. At 2048 faces
    each one is

        256x256     2048 * 65,536 * 4 B  =  0.54 GB      fits
        1024x1024   2048 * 1,048,576 * 4 B  =  8.6 GB    ten of these is 86 GB

    Measured: at 1024x1024 the 4090 sat at 24,041 MiB of 24,564 at 100% utilisation and did
    not finish. It did not raise -- an allocator at its ceiling thrashes rather than failing,
    so the symptom is a render that never returns, not an error naming the cause.

    BOTH CONSTANTS ARE MEASURED, not counted off the source. The first version counted ten
    live intermediates and set a 1.5 GB budget, and that is not what ran: peak was 2.39 GiB at
    572 faces over 65,536 pixels, which is 68 bytes per element, so 17 float32s are live. The
    conservative budget then cost 8x in launch overhead -- 256x256 took 9.21 s at chunk 572
    against 1.09 s at chunk 2048. Being wrong about memory is loud and being wrong about
    throughput is silent, so the second one needs the measurement more.

    The budget leaves roughly two thirds of a 24 GiB card free, because this renders alongside
    a generator rather than alone.
    """
    return max(1, int(budget_bytes / (per_face_tensors * bytes_per_elem * max(n_pixels, 1))))


ELEM_BUDGET = 8.0e9 / (17 * 4)          # elements per block; see `chunk_for` for the 17


def influence_pad(tau: float, n_faces: int, leak: float = 1e-6) -> float:
    """Pixels beyond its own outline that a face can still move, to within `leak`.

    A face contributes `sigmoid(d / tau)`, which for a pixel outside decays as `exp(d / tau)`.
    F of them accumulate, so the total a cull discards is below `leak` once

        pad  =  tau * ln(F / leak)

    At ANNY's 27,420 faces and the default tau of 0.0472 px this is 1.13 px -- about a
    hundredth of the width of a credit card's edge on screen. A face barely reaches past
    itself, which is what makes culling worth doing at all.
    """
    return float(tau * math.log(max(n_faces, 1) / leak))


def _dilated_bbox(tri2d: torch.Tensor, t: float):
    """Bounding box of {x : min-of-edge-half-planes(x) >= -t}, exactly.

    RETRACTED: an earlier version padded the VERTEX bounding box by `t`, which under-covers.
    `_inside_distance` is the minimum of three EDGE HALF-PLANE distances, and an edge's
    half-plane is a line that runs to infinity, so a pixel out past a corner can be far from
    the triangle and still only a little way outside one of its edges.

    Measured on a right-isoceles quad: pixel (8, 13) is 8.5 px from the triangle and 3.54 px
    outside its nearest edge line, so it still scored sigmoid(-7.5) = 5.5e-04. The vertex-box
    pad excluded it and the culled silhouette disagreed with the reference by exactly that.

    The region is an intersection of three offset half-planes, which is itself a triangle. Its
    corners are the pairwise intersections of the offset lines, so they are computed rather
    than approximated. A sliver whose edges are nearly parallel gives a near-singular solve
    and a huge box, which culls nothing for that face and stays CORRECT -- degrading to no
    cull rather than to a wrong answer is the failure mode to want here.
    """
    a, b, c = tri2d[:, 0], tri2d[:, 1], tri2d[:, 2]
    area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    sign = torch.where(area >= 0, 1.0, -1.0)[:, None]

    ms, cs = [], []
    for p0, p1 in ((a, b), (b, c), (c, a)):
        e = p1 - p0
        n = torch.stack([-e[:, 1], e[:, 0]], dim=-1)
        m = sign * (n / (n.norm(dim=-1, keepdim=True) + 1e-9))   # inward unit normal
        ms.append(m)
        cs.append((m * p0).sum(-1) - t)                          # m . x = c is the offset line

    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        mi, mj, ci, cj = ms[i], ms[j], cs[i], cs[j]
        det = mi[:, 0] * mj[:, 1] - mi[:, 1] * mj[:, 0]
        det = torch.where(det.abs() < 1e-9, torch.full_like(det, 1e-9), det)
        pts.append(torch.stack([(mj[:, 1] * ci - mi[:, 1] * cj) / det,
                                (mi[:, 0] * cj - mj[:, 0] * ci) / det], dim=-1))
    corners = torch.stack(pts, dim=1)                            # (F, 3, 2)
    return corners.min(1).values, corners.max(1).values


def _morton_order(tri2d: torch.Tensor) -> torch.Tensor:
    """Sort faces by screen locality, so a run of faces covers a small rectangle.

    Without this the cull buys nothing: a chunk of arbitrary faces spans the whole body and
    its bounding box is the whole body. Interleaving the bits of the centroid's x and y keeps
    neighbours in the ordering neighbours on screen.
    """
    c = tri2d.detach().mean(1)
    lo, hi = c.min(0).values, c.max(0).values
    n = ((c - lo) / (hi - lo).clamp(min=1e-9) * 1023).long().clamp(0, 1023)

    def spread(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return torch.argsort(spread(n[:, 0]) | (spread(n[:, 1]) << 1))


def _work_items(tri2d, pad, height, width, cull=True, budget=ELEM_BUDGET, max_faces=4096):
    """Blocks of (faces x rectangle) to evaluate, each inside the element budget.

    WHY THIS REPLACED A FLAT FACE LOOP. Testing every face against every pixel is O(F * P):
    2.9e10 pairs for ANNY at 1024x1024. The body covers 9.3% of that frame and the average
    face covers about 7 pixels, so nearly all of it is arithmetic on pixels the face cannot
    reach. Measured 20.8 s per image, which is 4,628 GPU-hours for the 800k-image corpus and
    193 days on one card.

    Faces arrive in Morton order, so a contiguous run has a tight bounding box. A run whose
    cost exceeds the budget is halved until it fits, which also tightens the box.

    `cull=False` yields the old whole-frame blocks. It is not dead code: it is the reference
    the cull is checked against, and keeping it on the same code path stops the two drifting.
    """
    n_faces = tri2d.shape[0]
    if not cull:
        step = max(1, int(budget / max(height * width, 1)))
        return [(i, min(i + step, n_faces), 0, height, 0, width)
                for i in range(0, n_faces, step)]

    lo, hi = _dilated_bbox(tri2d.detach(), pad)
    big = float(max(height, width)) * 8.0
    lo = torch.nan_to_num(lo, nan=-big, posinf=big, neginf=-big).clamp(-big, big).cpu()
    hi = torch.nan_to_num(hi, nan=big, posinf=big, neginf=-big).clamp(-big, big).cpu()
    items, stack = [], [(0, n_faces)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0:
            continue
        x0 = max(int(math.floor(float(lo[i0:i1, 0].min()))), 0)
        x1 = min(int(math.ceil(float(hi[i0:i1, 0].max()))) + 1, width)
        y0 = max(int(math.floor(float(lo[i0:i1, 1].min()))), 0)
        y1 = min(int(math.ceil(float(hi[i0:i1, 1].max()))) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue                       # entirely off screen, and off screen is not drawn
        n = i1 - i0
        if n > 1 and (n * (x1 - x0) * (y1 - y0) > budget or n > max_faces):
            mid = (i0 + i1) // 2
            stack.append((i0, mid))
            stack.append((mid, i1))
            continue
        items.append((i0, i1, y0, y1, x0, x1))
    return items


def _block_pixels(y0, y1, x0, x1, width, device, dtype):
    """Pixel coordinates of a rectangle, and their indices into the flattened frame."""
    rows = torch.arange(y0, y1, device=device, dtype=dtype)
    cols = torch.arange(x0, x1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    idx = (yy.long() * width + xx.long()).reshape(-1)
    return coords, idx


def soft_silhouette(verts: torch.Tensor, faces: torch.Tensor, cam: Camera,
                    tau: float | None = None, chunk: int | None = None,
                    cull: bool = True, max_faces: int = 4096,
                    budget: float | None = None) -> torch.Tensor:
    """Differentiable silhouette in [0, 1], shape (height, width).

    `tau` defaults to half a pixel of bleed AT THIS MESH'S FACE COUNT, via `tau_for_bleed`.
    It was a bare 1.0, which on ANNY reported a silhouette ten times the body's true area.

    Faces are evaluated only against pixels they can reach, within `influence_pad`. `cull`
    turns that off and is what the culled path is measured against.

    `chunk` is accepted for callers that pin it, and overrides the budget-derived block size.
    """
    if tau is None:
        tau = tau_for_bleed(0.5, faces.shape[0])
    tri = cam.project(verts)[faces]
    order = _morton_order(tri) if cull else torch.arange(tri.shape[0], device=tri.device)
    tri = tri[order]
    pad = influence_pad(tau, tri.shape[0])
    # `max_faces` and `budget` are the THROUGHPUT KNOB, not tuning noise. Smaller blocks have
    # tighter bounding boxes and do less arithmetic; larger blocks launch fewer kernels. The
    # first bounds work, the second bounds overhead, and which one dominates is measured
    # rather than assumed -- see the sweep recorded in `_controls`.
    if budget is None:
        budget = ELEM_BUDGET if chunk is None else float(chunk) * cam.height * cam.width

    # Accumulate log(1 - sigma) so the union becomes a sum. A running product of tens of
    # thousands of terms underflows to exactly zero and takes the gradient with it.
    log_keep = torch.zeros(cam.height * cam.width, device=verts.device, dtype=verts.dtype)
    for i0, i1, y0, y1, x0, x1 in _work_items(
            tri, pad, cam.height, cam.width, cull=cull, budget=budget,
            max_faces=max_faces):
        coords, idx = _block_pixels(y0, y1, x0, x1, cam.width, verts.device, verts.dtype)
        d = _inside_distance(coords, tri[i0:i1])
        sigma = torch.sigmoid(d / tau)
        log_keep = log_keep.index_add(0, idx, torch.log1p(-sigma.clamp(max=1 - 1e-7)).sum(0))
    return (1.0 - torch.exp(log_keep)).reshape(cam.height, cam.width)


def tau_for_bleed(bleed_px: float, n_faces: int) -> float:
    """The `tau` that bleeds coverage `bleed_px` past the true silhouette, for `n_faces`.

    RETRACTED: `tau` used to be a bare constant, defaulting to 1.0, with no face count in it.
    It is not scale-free, and the error grows with mesh size.

    The union accumulates `sigma_i = sigmoid(d_i / tau)` over EVERY face, and a background
    pixel sits outside all of them at some negative distance. Each contributes about
    `exp(d / tau)`, small but never zero, and F of them sum. The union crosses 0.5 where that
    sum reaches ln 2, so the silhouette bleeds outward to

        d  =  -tau * ln(F / ln 2)

    which is LINEAR IN TAU AND LOGARITHMIC IN FACE COUNT. Solving it for tau is this function.

    WHY IT WENT UNNOTICED. Every self-test in this file and in `depth_term.py` uses `_quad`,
    which has two faces. At F = 2 the log term is 1.06, so tau and bleed are the same number
    and the constants look correct. ANNY has 27,420 faces, where the term is 10.6.

    Measured on ANNY at 256x256, against an exact rasterised silhouette of 6,047 pixels:

        tau      coverage           IoU
        4.00     65,536  (100.0%)   0.092
        1.00     60,480  ( 92.3%)   0.100     <- the old default
        0.30     18,331  ( 28.0%)   0.330     <- where the old schedule ENDED
        0.10      7,177  ( 11.0%)   0.843
        0.03      6,143  (  9.4%)   0.984
        0.01      6,058  (  9.2%)   0.998

    So the old default reported a body covering the whole frame, ten times its true area, and
    the annealing schedule finished before it became a measurement. Nothing raised: the map
    was smooth, differentiable and plausible.
    """
    return float(bleed_px / math.log(max(n_faces, 2) / math.log(2.0)))


def tau_schedule(step: int, total: int, n_faces: int,
                 start_px: float = 4.0, end_px: float = 0.3) -> float:
    """Anneal softness from blurry to sharp, geometrically. Bounds are PIXELS OF BLEED.

    Starting sharp is the common failure: a crisp silhouette has near-zero gradient more than
    a pixel or two from its boundary, so a body starting further out than that receives no
    signal and the optimiser reports convergence without having moved. Starting soft gives
    every pixel a gradient; ending sharp recovers the precision.

    `n_faces` is required rather than defaulted, because a default here is exactly the bug
    this signature was changed to remove. `start_px` and `end_px` keep their old numbers and
    change meaning: they were raw tau, and at the two-face quad they were tuned on the two
    readings coincide to within 6%, so the tuning carries over and now scales.
    """
    t = min(max(step / max(total - 1, 1), 0.0), 1.0)
    return tau_for_bleed(start_px * (end_px / start_px) ** t, n_faces)


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Soft intersection-over-union: the reported agreement number.

    IoU rather than pixel L2, because L2 is dominated by body AREA -- a fit uniformly too
    large and one that is displaced can score alike, and only one of those is a pose error.
    """
    inter = (a * b).sum()
    union = a.sum() + b.sum() - inter
    return float(inter / union.clamp(min=1e-9))


def silhouette_loss(verts: torch.Tensor, faces: torch.Tensor, target: torch.Tensor,
                    cam: Camera, tau: float) -> torch.Tensor:
    """1 - soft IoU, as a differentiable objective.

    IoU is used as the loss and not merely as the report, so the number that drives the fit
    is the number that scores it. Two different measures invite the fit to improve one while
    the other is quoted.
    """
    S = soft_silhouette(verts, faces, cam, tau=tau)
    inter = (S * target).sum()
    union = S.sum() + target.sum() - inter
    return 1.0 - inter / union.clamp(min=1e-9)


# ---------------------------------------------------------------------------------------
# Controls. CLAUDE.md rule 2: a check that passes on known-broken input certifies the defect,
# so each of these must fail in the specific way named.
# ---------------------------------------------------------------------------------------

def _quad(cx: float, cy: float, r: float, z: float = 5.0):
    v = torch.tensor([[cx - r, cy - r, z], [cx + r, cy - r, z],
                      [cx + r, cy + r, z], [cx - r, cy + r, z]], dtype=torch.float64)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return v, f


def _controls() -> None:
    cam = Camera(width=64, height=64, fx=160.0, fy=160.0, cx=32.0, cy=32.0)

    # r=0.5 at z=5 with fx=160 is a 32 px box in a 64 px frame. r=1.0 filled the frame
    # edge to edge, which made the displacement control unable to move anything.
    v, f = _quad(0.0, 0.0, 0.5)
    S = soft_silhouette(v, f, cam, tau=0.5)
    assert 0.0 <= float(S.min()) and float(S.max()) <= 1.0, "silhouette left [0,1]"
    covered = float((S > 0.5).sum())
    print(f"  renders                    -> {covered:.0f} px of {cam.width * cam.height}")
    assert 400 < covered < 2000, f"quad should fill part of frame, covered {covered}"

    # 1. A GRADIENT EXISTS AT THE BOUNDARY. The entire reason for soft rasterization; an
    #    ordinary rasterizer scores zero here, and that is the bug being guarded against.
    vg = v.clone().requires_grad_(True)
    soft_silhouette(vg, f, cam, tau=0.5).sum().backward()
    g = float(vg.grad.abs().max())
    print(f"  gradient at boundary       -> {g:.3e}")
    assert g > 1e-9, "no silhouette gradient: this rasterizer is hard, not soft"

    # 2. IoU FALLS WHEN THE SHAPE MOVES. A metric that does not drop for a displaced body
    #    cannot drive a fit, and would report success for any pose at all.
    A = soft_silhouette(*_quad(0.0, 0.0, 0.5), cam, tau=0.5)
    B = soft_silhouette(*_quad(0.30, 0.0, 0.5), cam, tau=0.5)
    # Soft self-IoU is BELOW 1 by construction: for values in (0,1), sum(a*a) < sum(a), so
    # the soft edge costs a few percent against itself. It is the floor to compare against,
    # not a defect -- which is why it is printed as the baseline rather than assumed to be 1.
    self_iou = iou(A, A)
    print(f"  IoU self={self_iou:.3f} (soft floor)  displaced={iou(A, B):.3f}")
    # The floor is tau- and size-dependent, not a constant: a 32 px box with tau=0.5 carries
    # a ~2 px soft band on each edge, which is why it lands near 0.85 rather than 1.0. It is
    # measured and used as the baseline every other number is compared against -- CLAUDE.md
    # rule 4, a number without a baseline is not a measurement.
    assert 0.7 < self_iou < 1.0, f"soft self-IoU outside plausible range: {self_iou}"
    assert iou(A, B) < self_iou - 0.15, "IoU does not discriminate displacement"

    # 3. IoU FALLS WHEN SIZE CHANGES. This is the shape signal specifically -- if a larger
    #    body scored the same, silhouette could not recover phenotype and the whole premise
    #    of this module would be wrong.
    C = soft_silhouette(*_quad(0.0, 0.0, 0.675), cam, tau=0.5)   # 35% larger
    print(f"  IoU under size change      -> {iou(A, C):.3f}  (shape signal)")
    assert iou(A, C) < self_iou - 0.1, "silhouette blind to size: cannot constrain phenotype"

    # 4. THE BLIND SPOT, ASSERTED RATHER THAN DOCUMENTED. Moving along the view axis with a
    #    compensating scale leaves the projection unchanged. If this ever fails, the
    #    independence claim above is wrong and the two opinions overlap more than advertised.
    # Twice as far and twice as large projects identically -- the depth blindness that makes
    # this a complement to the vertex fit rather than a replacement for it.
    D = soft_silhouette(*_quad(0.0, 0.0, 1.0, z=10.0), cam, tau=0.5)
    print(f"  IoU under depth+scale      -> {iou(A, D):.3f}  (blind, as designed)")
    assert iou(A, D) > self_iou - 0.03, "silhouette saw a change it cannot see"

    # 5. TAU SCALES WITH FACE COUNT. This is the control the file did not have.
    #
    # Every other control here uses `_quad`, which has two faces, and at two faces a bare tau
    # and a pixel of bleed are the same number. So a tau with no face count in it passed
    # everything above while reporting ten times a real body's area.
    #
    # ONE SQUARE, REFINED IN PLACE. The outline never moves, so the only variable is the face
    # count and any change in coverage comes from the accumulation.
    #
    #     faces   hard    default tau      retracted tau=1.0
    #         2   1,089   1,059  IoU 0.972  1,093  IoU 0.996   <- where it was tuned
    #        32   1,089   1,083  IoU 0.994  1,141  IoU 0.954
    #       288   1,089   1,089  IoU 1.000  1,363  IoU 0.799
    #     1,152   1,089   1,089  IoU 1.000  1,621  IoU 0.672
    #     4,608   1,089   1,089  IoU 1.000  2,041  IoU 0.534
    #
    # The old constant is at its BEST on the two-face quad and degrades monotonically from
    # there. That is why every control passed.
    for n, floor in ((12, 0.99), (24, 0.99), (48, 0.99)):
        gv, gf = _tessellated_quad(n)
        hard = _hard_coverage(gv, gf, cam)
        got = _iou_mask(soft_silhouette(gv, gf, cam) > 0.5, hard)
        bad = _iou_mask(soft_silhouette(gv, gf, cam, tau=1.0) > 0.5, hard)
        print(f"  {gf.shape[0]:5d} faces -> IoU {got:.3f}   (retracted tau=1.0: {bad:.3f})")
        assert got >= floor, f"soft coverage disagrees with the rasteriser at {gf.shape[0]} faces"
        # The negative control: the retracted default must FAIL here, or the fix above has
        # certified the defect rather than removed it.
        assert bad < 0.85, f"the old constant did not bleed at {gf.shape[0]} faces; control is dead"

    # 6. ANNEALING RUNS SOFT TO SHARP, never the reverse.
    taus = [tau_schedule(i, 10, f.shape[0]) for i in range(10)]
    print(f"  tau schedule               -> {taus[0]:.2f} .. {taus[-1]:.2f}")
    assert taus[0] > taus[-1] and all(a >= b for a, b in zip(taus, taus[1:])), \
        "tau must anneal monotonically from soft to sharp"


def _tessellated_quad(n: int):
    """One flat square cut into 2*n*n triangles. Same silhouette, many more faces.

    The face count is the variable under test and the outline is held fixed, so any change in
    measured coverage comes from the accumulation and not from the geometry.
    """
    import numpy as np
    g = np.linspace(-0.5, 0.5, n + 1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    v = torch.tensor(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 5.0)], -1),
                     dtype=torch.float64)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    f = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return v, torch.tensor(f, dtype=torch.long)


def _hard_coverage(verts, faces, cam, chunk: int | None = None) -> torch.Tensor:
    """Exact rasterised coverage. The baseline, because a number without one is not a
    measurement -- and a WRONG baseline is worse, because it convicts the thing under test.

    The tolerance is not decoration. Written as a strict `d > 0` this dropped every pixel whose
    centre lands exactly on an edge SHARED by two triangles, which on a regular tessellation
    aligns with the pixel lattice. Measured on one fixed square, refined in place: 930 px at 2
    faces falling to 142 px at 4,608, for an outline that never moved. It reported the soft
    silhouette as wrong at scale when the soft silhouette was reading the correct 1,089 px.
    """
    ys, xs = torch.meshgrid(
        torch.arange(cam.height, device=verts.device, dtype=verts.dtype),
        torch.arange(cam.width, device=verts.device, dtype=verts.dtype), indexing="ij")
    px = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    tri = cam.project(verts)[faces]
    chunk = chunk_for(px.shape[0]) if chunk is None else chunk
    out = torch.zeros(px.shape[0], dtype=torch.bool, device=verts.device)
    for i in range(0, tri.shape[0], chunk):
        out |= (_inside_distance(px, tri[i:i + chunk]) > -1e-6).any(0)
    return out.reshape(cam.height, cam.width)


def _iou_mask(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a & b).sum()) / max(float((a | b).sum()), 1.0)


def _recovers_shape() -> None:
    """The positive control: perturb a known shape and check the fit gets it back.

    Ground truth is known because the target was rendered from it, so this is a real
    measurement rather than a plausibility check.
    """
    cam = Camera(width=64, height=64, fx=160.0, fy=160.0, cx=32.0, cy=32.0)
    vt, f = _quad(0.0, 0.0, 0.5)
    target = soft_silhouette(vt, f, cam, tau=0.5).detach()

    scale = torch.tensor(1.6, dtype=torch.float64, requires_grad=True)  # 60% too large
    opt = torch.optim.LBFGS([scale], max_iter=30, line_search_fn="strong_wolfe")
    steps = 12
    for i in range(steps):
        tau = tau_schedule(i, steps, f.shape[0])

        def closure():
            opt.zero_grad()
            v = vt.clone(); v[:, :2] = vt[:, :2] * scale
            loss = silhouette_loss(v, f, target, cam, tau)
            loss.backward()
            return loss
        opt.step(closure)

    got = float(scale.detach())
    print(f"  shape recovery             -> started 1.600, recovered {got:.3f}, true 1.000")
    assert abs(got - 1.0) < 0.06, f"failed to recover scale from silhouette: {got}"


if __name__ == "__main__":
    torch.manual_seed(0)
    print("soft silhouette controls:")
    _controls()
    print("\npositive control:")
    _recovers_shape()
    print("\nall controls passed")
