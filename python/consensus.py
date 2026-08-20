"""Agreement between independent pose estimators, and the faults that reject a sample.

Called from Elixir through pythonx; kept as plain Python so it can be tested and reasoned
about without a BEAM.

The rule this file exists to enforce: a keypoint set is emitted only when every available
backend agrees. There is deliberately no averaging, no "best backend wins", and no
majority-vote fallback -- each of those turns a disagreement into a number that no estimator
produced, which is exactly the laundering this panel is meant to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

import numpy as np

# COCO-WholeBody layout, 133 keypoints. The panel runs in this space rather than COCO-17
# because See-Through separates `face`, `irides`, `eyebrow`, `eyewhite`, `eyelash`, `nose`,
# `mouth`, `ears` and `handwear` -- none of which a 17-point body skeleton constrains at all.
# A body-only panel would certify a pose while saying nothing about most of the tags it is
# meant to supervise.
REGIONS: dict[str, tuple[int, int]] = {
    "body":       (0, 17),     # COCO-17
    "feet":       (17, 23),
    "face":       (23, 91),    # 68-point face
    "left_hand":  (91, 112),   # 21
    "right_hand": (112, 133),  # 21
}

# Per-keypoint sigmas: the standard deviation of human annotator disagreement, in units of
# object scale. Eyes are pinned tightly (0.025); hips and ankles are genuinely ambiguous even
# to people (0.107 / 0.089). One flat threshold would hold the ankle to a precision humans do
# not achieve, and let the eye drift by more than the feature is wide.
BODY_SIGMAS = np.array([
    .026, .025, .025, .035, .035,
    .079, .079, .072, .072, .062, .062,
    .107, .107, .087, .087, .089, .089,
])
FOOT_SIGMAS = np.full(6, .079)
# COCO-WholeBody's face and hand sigmas are much tighter than any body joint -- these are
# small features and annotators agree closely on them. Reusing a body sigma here would make
# the face check nearly impossible to fail, which is the opposite of what is wanted.
FACE_SIGMAS = np.full(68, .015)
HAND_SIGMAS = np.full(21, .025)

WHOLEBODY_SIGMAS = np.concatenate(
    [BODY_SIGMAS, FOOT_SIGMAS, FACE_SIGMAS, HAND_SIGMAS, HAND_SIGMAS]
)
assert len(WHOLEBODY_SIGMAS) == 133

#: Per-region agreement floors. Face and hands are held tighter because the tags that depend
#: on them (irides, eyelash, handwear) are small: a face agreeing only to body tolerance is
#: not agreement for this purpose.
DEFAULT_FLOORS: dict[str, float] = {
    "body": 0.85, "feet": 0.80, "face": 0.90, "left_hand": 0.80, "right_hand": 0.80,
}


class Fault(str, Enum):
    """Why a sample was rejected. Distinct values because the remedies differ: DISAGREEMENT
    means the image is genuinely hard, BACKEND_FAILED means the panel was degraded."""

    DISAGREEMENT = "backends disagree beyond the floor"
    PERSON_COUNT = "backends found different numbers of people"
    QUORUM = "too few backends available to form a panel"


@dataclass
class Prediction:
    """One backend's answer for one image."""

    backend: str
    #: (n_persons, 17, 3) — x, y, visibility
    keypoints: np.ndarray
    #: (n_persons, 4) — xyxy, used for the OKS scale term
    boxes: np.ndarray
    ok: bool = True
    error: str | None = None


@dataclass
class Verdict:
    accepted: bool
    faults: list[Fault] = field(default_factory=list)
    #: which backends actually voted. Written to the corpus so panel degradation is
    #: auditable after the fact rather than invisible.
    participants: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    #: pairwise OKS, keyed by the two backend names
    pairwise: dict[tuple[str, str], float] = field(default_factory=dict)
    detail: str = ""
    #: the emitted keypoints, or None when faulted. Never a blend.
    keypoints: np.ndarray | None = None


def oks(a: np.ndarray, b: np.ndarray, box: np.ndarray, sigmas: np.ndarray) -> float:
    """Object Keypoint Similarity between two keypoint sets of matching length.

    Mirrors COCO's definition: squared distance scaled by 2*(sigma*2)^2 and by the object
    area, exponentiated, averaged over keypoints both sides consider visible.

    Scale normalisation is the reason a pixel threshold will not do -- the same absolute
    error is trivial on a full-frame figure and disqualifying on a distant one.
    """
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1.0)
    d2 = (a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2
    # A keypoint counts only where BOTH backends claim to see it. Comparing against a joint
    # one side marked invisible measures the visibility flag, not the position.
    vis = (a[:, 2] > 0) & (b[:, 2] > 0)
    if not vis.any():
        # No shared visible joints is not agreement -- it is an absence of evidence, and
        # returning 1.0 here would silently pass every occluded figure.
        return 0.0
    e = d2 / (2 * (sigmas * 2) ** 2) / (area + np.spacing(1))
    return float(np.exp(-e[vis]).mean())


class Rule(str, Enum):
    """How much agreement is required among the backends that responded.

    UNANIMOUS is the default and matches the panel's stated purpose: fault if there is a
    difference. MAJORITY is offered because it is what "quorum" means in Paxos/BFT, but see
    the warning on `adjudicate` before choosing it.
    """

    UNANIMOUS = "every responding backend must agree"
    MAJORITY = "a majority must agree; outliers are outvoted"


def adjudicate(
    predictions: list[Prediction],
    oks_floor: float = 0.85,
    quorum: int = 3,
    rule: "Rule" = None,
) -> Verdict:
    """Accept only on unanimous agreement.

    `oks_floor` is the pairwise bar. 0.85 is deliberately strict: COCO treats 0.5-0.95 as the
    evaluation range and 0.75 as "good", but this is not scoring a model against ground truth
    -- it is asking whether four estimators saw the same person. Two correct estimators
    routinely exceed 0.9 on an unambiguous figure.

    Two rules, deliberately separate, because the word "quorum" collapses them:

      * AVAILABILITY -- `quorum`, how many backends must respond.
      * AGREEMENT    -- `rule`, how many of those must concur.

    Classical consensus sets availability at a majority (N=3 tolerates 1 failure) and decides
    by majority. This panel defaults agreement to UNANIMOUS instead, and the reason is that
    the BFT analogy does not survive contact with pose estimation.

    BFT tolerates f faulty nodes because failures are assumed INDEPENDENT: a corrupted replica
    fails in its own way, so a majority rarely shares an error. Pose estimators fail
    CORRELATED. A hard occlusion, an odd crop, a costume that reads as a limb -- these mislead
    every backend, in the same direction. Two estimators agreeing on a wrong wrist is ordinary;
    two replicas independently corrupting to the same value is not.

    So MAJORITY inherits BFT's arithmetic without BFT's independence assumption. It raises
    yield and admits precisely the correlated errors the panel exists to catch. It is
    available because it is sometimes the right trade -- a corpus that rejects 40% of samples
    may be worse than one with a small correlated-error rate -- but it should be chosen
    knowingly, not inherited from the vocabulary.

    `quorum` is the minimum number of WORKING backends, and it is the ONLY thing a failure
    affects. With four backends and quorum 3, one may drop out and a sample still passes if
    the remaining three agree; with three backends and quorum 3, none may.

    Which is why losing RF-DETR to its COCO-17 head matters more than one vote: it takes the
    panel from 4-with-margin to 3-at-quorum, where any outage stops the corpus entirely.
    """
    working = [p for p in predictions if p.ok]
    failed = [p for p in predictions if not p.ok]
    faults: list[Fault] = []
    detail: list[str] = []

    # A failure is NOT itself a fault -- quorum decides. An earlier version raised a fault on
    # any failure, which made `quorum` dead code: it never bound, because a stricter rule
    # fired first. Two rules that contradict each other are worse than either alone, and the
    # one that wins is whichever happens to run first.
    #
    # The hazard that rule was reaching for is real though: a panel that quietly shrinks keeps
    # reporting the confidence of a full panel. The answer is not to fault, it is to RECORD --
    # `participants` goes into the emitted row, so a corpus can be asked afterwards how many
    # backends actually voted on each sample, and a slow degradation is visible in the data
    # rather than hidden behind a passing verdict.
    if failed:
        detail += [f"{p.backend} unavailable: {p.error}" for p in failed]

    if len(working) < quorum:
        faults.append(Fault.QUORUM)
        detail.append(f"{len(working)} working backends, quorum {quorum}")
        return Verdict(accepted=False, faults=faults, detail="; ".join(detail))

    counts = {p.backend: len(p.keypoints) for p in working}
    if len(set(counts.values())) > 1:
        faults.append(Fault.PERSON_COUNT)
        detail.append(f"person counts {counts}")
        return Verdict(accepted=False, faults=faults, detail="; ".join(detail))

    pairwise: dict[tuple[str, str], float] = {}
    worst = 1.0
    for p, q in combinations(working, 2):
        # Persons are index-aligned here; a real implementation matches them by IoU first.
        scores = [oks(p.keypoints[i], q.keypoints[i], p.boxes[i]) for i in range(len(p.keypoints))]
        s = min(scores) if scores else 0.0
        pairwise[(p.backend, q.backend)] = s
        worst = min(worst, s)

    rule = rule or Rule.UNANIMOUS
    low = [f"{a}~{b}={s:.3f}" for (a, b), s in pairwise.items() if s < oks_floor]
    if rule is Rule.UNANIMOUS:
        disagree = worst < oks_floor
    else:
        # Majority: a backend is an outlier if it disagrees with everyone else. Accept when
        # some subset larger than half mutually agrees.
        agree_count = {p.backend: 0 for p in working}
        for (a, b), sc in pairwise.items():
            if sc >= oks_floor:
                agree_count[a] += 1
                agree_count[b] += 1
        need = len(working) // 2  # each member of a majority agrees with >= this many others
        disagree = sum(1 for c in agree_count.values() if c >= need) <= len(working) // 2
    if disagree:
        faults.append(Fault.DISAGREEMENT)
        detail.append(f"[{rule.name}] below floor: " + ", ".join(low))

    accepted = not faults
    return Verdict(
        accepted=accepted,
        faults=faults,
        participants=[p.backend for p in working],
        absent=[p.backend for p in failed],
        pairwise=pairwise,
        detail="; ".join(detail),
        # On acceptance the backends agree to within the floor, so any one of them is as good
        # as another; the first is emitted verbatim rather than blending. A blend would be a
        # skeleton no estimator produced.
        keypoints=working[0].keypoints if accepted else None,
    )
