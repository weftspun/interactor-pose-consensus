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


def soft_silhouette(verts: torch.Tensor, faces: torch.Tensor, cam: Camera,
                    tau: float = 1.0, chunk: int = 2048) -> torch.Tensor:
    """Differentiable silhouette in [0, 1], shape (height, width).

    `chunk` bounds peak memory. The naive form materialises (F, P), which for ANNY at 36,108
    faces and a 256x256 image is 2.4e9 entries; faces accumulate in blocks instead, in log
    space so the product stays stable.
    """
    ys, xs = torch.meshgrid(
        torch.arange(cam.height, device=verts.device, dtype=verts.dtype),
        torch.arange(cam.width, device=verts.device, dtype=verts.dtype), indexing="ij")
    px = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    tri = cam.project(verts)[faces]
    # Accumulate log(1 - sigma) so the union becomes a sum. A running product of tens of
    # thousands of terms underflows to exactly zero and takes the gradient with it.
    log_keep = torch.zeros(px.shape[0], device=verts.device, dtype=verts.dtype)
    for i in range(0, tri.shape[0], chunk):
        d = _inside_distance(px, tri[i:i + chunk])
        sigma = torch.sigmoid(d / tau)
        log_keep = log_keep + torch.log1p(-sigma.clamp(max=1 - 1e-7)).sum(0)
    return (1.0 - torch.exp(log_keep)).reshape(cam.height, cam.width)


def tau_schedule(step: int, total: int, start: float = 4.0, end: float = 0.3) -> float:
    """Anneal softness from blurry to sharp, geometrically.

    Starting sharp is the common failure: a crisp silhouette has near-zero gradient more than
    a pixel or two from its boundary, so a body starting further out than that receives no
    signal and the optimiser reports convergence without having moved. Starting soft gives
    every pixel a gradient; ending sharp recovers the precision.
    """
    t = min(max(step / max(total - 1, 1), 0.0), 1.0)
    return float(start * (end / start) ** t)


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

    # 5. ANNEALING RUNS SOFT TO SHARP, never the reverse.
    taus = [tau_schedule(i, 10) for i in range(10)]
    print(f"  tau schedule               -> {taus[0]:.2f} .. {taus[-1]:.2f}")
    assert taus[0] > taus[-1] and all(a >= b for a, b in zip(taus, taus[1:])), \
        "tau must anneal monotonically from soft to sharp"


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
        tau = tau_schedule(i, steps)

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
