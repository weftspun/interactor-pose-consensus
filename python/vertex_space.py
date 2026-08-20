"""Compare estimators in SOMA-X vertex space instead of a keypoint schema.

## The problem this replaces

The four backends speak incompatible languages: MediaPipe Holistic emits 33 pose + 468 face +
21x2 hand points, SDPose-Wholebody emits COCO-WholeBody 133, RF-DETR emits COCO-17, GEM-X emits
SOMA-X coefficients. Comparing them by keypoint index requires hand-authoring correspondences
-- deciding which of MediaPipe's 468 face vertices "is" COCO-WholeBody face point 23. Those
decisions are arbitrary, unverifiable, and silently wrong: a bad correspondence shows up as a
permanent disagreement that looks like a model defect.

## What replaces it

Associate every backend's keypoints to SOMA-X mesh vertices ONCE, offline. After that any two
backends are comparable wherever their vertex sets intersect, with no schema mapping at all.
A backend that constrains 17 vertices and one that constrains 133 simply share 17.

This is also the space the corpus already uses: `anny_render_schema.py`'s `KEYPOINTS_2D` is
keyed by `bone_id`, not by a keypoint index. The canonical pose space here was never COCO.

## Why % of stature, not OKS

`preflight_audit.py` already measures skeleton/mesh pairing as *median joint-to-nearest-vertex
distance as a percentage of stature*, and its comment records why: an absolute tolerance
"would pass every spot-check and ship", because the same millimetre error is negligible on an
adult and disqualifying on an infant. Stature normalisation is a physical measure and
transfers across phenotypes.

OKS normalises by bounding-box area -- an image-space quantity that conflates a distant figure
with a small one, and that has no meaning at all once the comparison moves to a mesh.

Reusing the existing definition rather than writing a second one is deliberate: two
definitions of "how far apart" drift, and then the audit and the panel disagree about whether
the same pose is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - surfaced by the caller, not swallowed
    cKDTree = None


@dataclass(frozen=True)
class VertexMap:
    """A backend's keypoints, associated to SOMA-X vertices.

    Built once per backend against a neutral-phenotype mesh, then reused. `vertex_id[k]` is
    the SOMA-X vertex that backend keypoint `k` sits on; `residual[k]` is how far it sat from
    that vertex when the map was built, as a fraction of stature.

    The residual is kept because it is the honest quality of the correspondence. A keypoint
    that never sat closer than 4% of stature to any vertex is not really associated with the
    mesh, and comparisons through it are noise -- `usable()` exists to exclude those rather
    than let them quietly widen every disagreement.
    """

    backend: str
    vertex_id: np.ndarray      # (n_keypoints,) int
    residual: np.ndarray       # (n_keypoints,) float, fraction of stature
    #: association residual above which a keypoint is not trusted as a correspondence.
    #: 1% of stature is preflight_audit's own pairing tolerance; the same number is used here
    #: so the panel and the audit cannot disagree about what "paired" means.
    max_residual: float = 0.01

    def usable(self) -> np.ndarray:
        return self.residual <= self.max_residual


def stature(verts: np.ndarray) -> float:
    """Largest extent of the mesh. Matches preflight_audit's definition exactly."""
    return float((verts.max(0) - verts.min(0)).max())


def associate(joints: np.ndarray, verts: np.ndarray, backend: str,
              max_residual: float = 0.01) -> VertexMap:
    """Associate keypoints to their nearest SOMA-X vertices.

    `joints` (n, 3) and `verts` (v, 3) in the same frame. Uses the same cKDTree nearest-vertex
    query as the preflight audit, and reports residuals as a fraction of stature.
    """
    if cKDTree is None:
        raise ImportError("scipy is required for vertex association")
    if joints.shape[-1] != 3 or verts.shape[-1] != 3:
        raise ValueError("association happens in 3D; project after, never before")
    s = stature(verts)
    dist, idx = cKDTree(verts).query(joints)
    return VertexMap(backend=backend, vertex_id=idx,
                     residual=dist / max(s, 1e-9), max_residual=max_residual)


def shared_vertices(a: VertexMap, b: VertexMap) -> tuple[np.ndarray, np.ndarray]:
    """Indices into each backend's keypoints where both land on the same vertex AND both
    associations are trustworthy.

    Returning the intersection rather than a padded union is what makes a 17-point backend
    comparable with a 133-point one without inventing correspondences for the other 116.
    """
    a_ok, b_ok = a.usable(), b.usable()
    common = np.intersect1d(a.vertex_id[a_ok], b.vertex_id[b_ok])
    ai = np.array([np.where(a.vertex_id == v)[0][0] for v in common], dtype=int)
    bi = np.array([np.where(b.vertex_id == v)[0][0] for v in common], dtype=int)
    return ai, bi


def disagreement(
    a_pts: np.ndarray, b_pts: np.ndarray,
    a_map: VertexMap, b_map: VertexMap,
    subject_stature: float,
) -> tuple[float, int]:
    """Median distance between two backends at their shared vertices, as % of stature.

    Returns (percent, n_compared). `n_compared` is returned rather than folded away because a
    tiny overlap makes the percentage meaningless, and a caller that cannot see the count will
    treat two shared vertices as confidently as forty.

    Median, not mean: one badly-placed limb should not be able to average itself away against
    a well-placed torso, nor to condemn an otherwise agreeing pose. The audit this mirrors
    uses the median for the same reason.
    """
    ai, bi = shared_vertices(a_map, b_map)
    if len(ai) == 0:
        # No shared trustworthy vertices is not agreement. Returning 0% here -- "they differ
        # by nothing" -- would pass every pair that has nothing in common.
        return float("inf"), 0
    d = np.linalg.norm(a_pts[ai] - b_pts[bi], axis=-1)
    return 100.0 * float(np.median(d)) / max(subject_stature, 1e-9), int(len(ai))
