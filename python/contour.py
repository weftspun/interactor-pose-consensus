"""A fixed-length ordered contour from a silhouette mask.

`silhouette.py` renders a mask. Every consumer downstream wants an outline: a
fixed count of ordered points, which is the one output shape a dataflow
accelerator will take. This turns one into the other and back.

The round trip is the test: trace, resample to N, fill, and the filled polygon
must agree with the mask it came from.

    python contour.py --self-test
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

DEFAULT_N = 128
#: Below this the round trip is failing rather than coarse. Measured: at 128
#: points a disc round-trips at 0.966, a square 0.969 and an L 0.959.
ROUND_TRIP_IOU = 0.95

_NEIGHBOURS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def largest_component(mask):
    """The biggest connected blob, because a body is one thing and specks are not."""
    lab = np.zeros(mask.shape, dtype=np.int32)
    cur, sizes = 0, {}
    for sy in range(mask.shape[0]):
        for sx in range(mask.shape[1]):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack, n = [(sy, sx)], 0
            lab[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                n += 1
                for dy, dx in _NEIGHBOURS:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                            and mask[ny, nx] and not lab[ny, nx]):
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
            sizes[cur] = n
    if not sizes:
        return np.zeros_like(mask)
    return lab == max(sizes, key=sizes.get)


def trace_boundary(mask):
    """Moore-neighbour tracing, clockwise, starting from the topmost-leftmost pixel."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    start = (int(ys[0]), int(xs[int(np.argmin(xs[ys == ys[0]]))]))
    ring, cur, back = [start], start, 7
    for _ in range(4 * int(mask.sum()) + 8):
        found = False
        for k in range(8):
            idx = (back + 1 + k) % 8
            dy, dx = _NEIGHBOURS[idx]
            ny, nx = cur[0] + dy, cur[1] + dx
            if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]):
                back = (idx + 5) % 8
                cur = (ny, nx)
                found = True
                break
        if not found:
            break
        if cur == start and len(ring) > 2:
            break
        ring.append(cur)
    return np.array(ring, dtype=np.float64)


def resample(ring, n=DEFAULT_N):
    """Arc-length resampling. FIXED N is the whole point: the shape stops varying."""
    if len(ring) < 2:
        return np.zeros((n, 2), dtype=np.float64)
    closed = np.vstack([ring, ring[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    walk = np.concatenate([[0.0], np.cumsum(seg)])
    if walk[-1] <= 0:
        return np.repeat(closed[:1], n, axis=0)
    want = np.linspace(0.0, walk[-1], n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    for i, d in enumerate(want):
        j = int(np.searchsorted(walk, d, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t = 0.0 if seg[j] <= 0 else (d - walk[j]) / seg[j]
        out[i] = closed[j] + t * (closed[j + 1] - closed[j])
    return out


def contour(mask, n=DEFAULT_N):
    return resample(trace_boundary(largest_component(np.asarray(mask) > 0.5)), n)


def fill(points, shape):
    """Even-odd scanline fill, so the round trip needs nothing outside numpy."""
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    if len(points) < 3:
        return out
    ys, xs = points[:, 0], points[:, 1]
    for y in range(h):
        hits = []
        for i in range(len(points)):
            y0, y1 = ys[i], ys[(i + 1) % len(points)]
            if (y0 <= y) != (y1 <= y):
                t = (y - y0) / (y1 - y0)
                hits.append(xs[i] + t * (xs[(i + 1) % len(points)] - xs[i]))
        hits.sort()
        for a, b in zip(hits[0::2], hits[1::2]):
            out[y, max(0, int(np.ceil(a))):min(w, int(np.floor(b)) + 1)] = True
    return out


def iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum()) / float(union)


def _disc(h=64, w=64, r=20):
    y, x = np.mgrid[0:h, 0:w]
    return ((y - h / 2) ** 2 + (x - w / 2) ** 2) < r * r


def _square(h=64, w=64, m=16):
    a = np.zeros((h, w), dtype=bool)
    a[m:h - m, m:w - m] = True
    return a


def _ell(h=64, w=64):
    a = np.zeros((h, w), dtype=bool)
    a[12:52, 12:28] = True
    a[36:52, 12:52] = True
    return a


def self_test():
    """A round trip that has never rejected a bad contour has not shown it can."""
    for name, m in (("disc", _disc()), ("square", _square()), ("ell", _ell())):
        c = contour(m)
        if c.shape != (DEFAULT_N, 2):
            sys.exit("FAIL  %s gave %s, not a fixed %d points" % (name, c.shape, DEFAULT_N))
        got = iou(fill(c, m.shape), m)
        if got < ROUND_TRIP_IOU:
            sys.exit("FAIL  %s round trip is %.3f, under %.2f" % (name, got, ROUND_TRIP_IOU))
        print("  ok    %-6s %d points, round trip IoU %.3f" % (name, DEFAULT_N, got))

    m = _ell()
    shuffled = contour(m).copy()
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    if iou(fill(shuffled, m.shape), m) >= ROUND_TRIP_IOU:
        sys.exit("FAIL  a shuffled contour still rebuilt the mask, so order is not tested")
    print("  ok    control: shuffling the order breaks the round trip")

    two = np.zeros((64, 64), dtype=bool)
    two[10:20, 10:20] = True
    two[40:60, 40:60] = True
    if iou(fill(contour(two), two.shape), two) >= ROUND_TRIP_IOU:
        sys.exit("FAIL  two blobs round-tripped, but one ring cannot hold two regions")
    print("  ok    control: two regions cannot round trip, which is the stated limit")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--points", type=int, default=DEFAULT_N)
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    ap.error("nothing to do; pass --self-test")


if __name__ == "__main__":
    sys.exit(main())
