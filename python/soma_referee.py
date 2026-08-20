"""The parametric referee: fit ANNY/SOMA-X to a backend's keypoints and judge the fit.

## Why a referee and not a fourth backend

The panel wanted a parametric member because a parametric model CANNOT REPRESENT AN
IMPOSSIBLE SKELETON. Discriminative estimators can: a heatmap peak lands where it lands, and
nothing in the architecture forbids an elbow bending backwards or a limb changing length
between frames. That property -- not the vote -- was the whole reason GEM-X was wanted.

GEM-X is NVlabs and its terms are unresolved, so it cannot be seated. But the property does
not need GEM-X. It needs a parametric human, and ANNY is one we own outright: no licence
question, no gated corpus, no upstream checkpoint.

A referee is also STRICTLY STRONGER than a third voter, which is the part worth being clear
about. A third backend can be outvoted 2-to-1 by two estimators making the same correlated
mistake -- precisely the failure this panel exists to catch, since the remaining backends are
all discriminative and COCO-trained, so their errors correlate. The referee cannot be
outvoted: an impossible pose is rejected even when every backend agrees on it, because the
question it answers is not "do they concur" but "could a human body do this at all".

## The fit residual IS the measurement

`AnnyInverter` solves for the pose and phenotype that best explain the target. The residual
left over is the physical quantity: how far the keypoints sit from ANY pose the rig can reach.
Reported as a percentage of stature, matching `preflight_audit.py` -- an absolute tolerance
"would pass every spot-check and ship", because the same millimetre error is negligible on an
adult and disqualifying on a child.

## Hands are refereed, but the finger path must earn it first

`anny-pose-retarget-work/README.md` records that per-joint extraction was verified for arms,
legs and spine and NEVER INDEPENDENTLY VERIFIED FOR FINGERS: 28 small chained joints, 3 deep
per chain, where a per-joint convention error compounds down the chain instead of staying
local. `LeftHandPinky1` extracted as [-65.2, -55.0, 75.2] -- all three axes large -- while
every sibling MCP had near-zero X/Z and a moderate Y-only curl. The logbook calls that a
strong circumstantial signal of gimbal lock or an unaccounted bind-pose skew, never
root-caused.

Abstaining on hands was the first design here and it was wrong. "Never verified" is not
"known broken", and `handwear` is one of the 24 tags See-Through must separate -- an
unrefereed hand is a supervised output with nothing checking it. Declining to measure does not
make the hand correct; it makes the defect invisible, which is the failure mode rule 3 exists
to name.

So the referee judges hands, gated on a check aimed at the specific defect. The distinguishing
property of a per-joint convention error is not that it is large but that it GROWS WITH DEPTH:
a local error stays local, while a convention error compounds MCP -> PIP -> DIP. `chain_gate`
fits a known curl and compares residual by depth. Error that is flat across depths is ordinary
fit noise and the hand numbers count; error that climbs is the logbook's bug still present,
and the hands are marked UNTRUSTED -- reported and counted, never silently averaged in.

That is the second chance: the finger path is measured rather than assumed, and it either
passes a test targeting its known failure or fails visibly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

#: Every region the referee judges. Hands included -- see the module docstring.
REFEREED = ("body", "feet", "face", "left_hand", "right_hand")
HANDS = ("left_hand", "right_hand")

#: Residual above which no ANNY pose explains the keypoints, as a fraction of stature.
#: 2% of stature on a 1.7 m adult is about 34 mm -- roughly a golf ball (42.7 mm) -- and
#: comfortably above the ~1% pairing tolerance preflight_audit uses for a genuinely paired
#: mesh. A pose needing more than that is not a pose the rig can reach.
IMPOSSIBLE_RESIDUAL = 0.02

#: Hands are small, so the same fraction-of-stature bar would be far looser on a finger than
#: on a femur. A whole hand spans roughly 11% of stature; holding it to the body tolerance
#: would permit a fingertip error of a fifth of the hand. 0.6% of stature is ~10 mm on a
#: 1.7 m adult -- about a AAA battery's diameter (10.5 mm).
HAND_RESIDUAL = 0.006

#: How much residual may grow from the base of a finger chain to its tip before the growth is
#: read as the logbook's compounding-convention defect rather than ordinary fit noise. A
#: convention error roughly compounds per joint; noise does not. 1.6x across two joints of
#: depth is comfortably above measurement scatter and well below true compounding.
CHAIN_GROWTH_LIMIT = 1.6


class RefereeCall(str, Enum):
    FITS = "a SOMA-X pose explains these keypoints"
    IMPOSSIBLE = "no SOMA-X pose reaches these keypoints"
    HANDS_UNTRUSTED = "body fits, but the finger chain failed its depth gate"
    NOT_RUN = "referee did not run -- a failure, never a pass"


@dataclass
class ChainGate:
    """Result of the finger-chain verification, per hand."""

    passed: bool
    #: median residual at each depth: 1 = MCP, 2 = PIP, 3 = DIP.
    by_depth: dict[int, float] = field(default_factory=dict)
    growth: float = float("nan")
    detail: str = ""


@dataclass
class RefereeVerdict:
    call: RefereeCall
    #: median residual as a PERCENTAGE of stature, per region.
    residual_pct: dict[str, float] = field(default_factory=dict)
    #: hand regions whose numbers did not earn trust. Carried into the corpus so they can be
    #: counted afterwards rather than discovered later.
    untrusted: tuple[str, ...] = ()
    gates: dict[str, ChainGate] = field(default_factory=dict)
    detail: str = ""

    def household(self, region: str, stature_m: float = 1.7) -> str:
        """Residual as a household object, per the workspace's reporting rule.

        A bare "2.1%" does not tell a reader whether the error matters.
        """
        mm = self.residual_pct.get(region, float("nan")) / 100.0 * stature_m * 1000.0
        # NEAREST anchor, and multiples of it. First-that-fits was the earlier version and it
        # printed "about a pencil" for both 3.4 mm and 6.8 mm -- two numbers that differ by
        # 2x reading identically is exactly the failure the household rule exists to prevent.
        anchors = (("a credit card's thickness", 0.76), ("a penny", 1.52),
                   ("a pencil", 7.0), ("a AAA battery", 10.5), ("a AA battery", 14.5),
                   ("a nickel", 21.2), ("a golf ball", 42.7), ("an adult wrist", 57.0),
                   ("a soda can", 66.0))
        best, best_err = None, float("inf")
        for name, size in anchors:
            for mult in (1, 2, 3, 4, 5):
                err = abs(mm - size * mult) / max(mm, 1e-9)
                if err < best_err:
                    best_err, best = err, (name, size, mult)
        name, _, mult = best
        stack = name if mult == 1 else f"{mult} stacked {name.split(' ', 1)[-1]}s"
        return f"{mm:.1f} mm (about {stack})"


def chain_gate(residual_by_depth: dict[int, np.ndarray]) -> ChainGate:
    """Test the finger chain for the logbook's compounding-convention defect.

    A per-joint convention error does not stay put: it accumulates as the chain is composed,
    so DIP residual runs well above MCP residual. Ordinary fit noise is roughly flat with
    depth. Comparing depths therefore separates the two, where comparing magnitudes alone
    cannot -- a large flat error is a bad fit, a small growing one is the bug.

    Requires all three depths. Two points cannot distinguish a trend from a fluctuation, and
    guessing from two would be the convenient proxy rather than the physical quantity.
    """
    missing = [d for d in (1, 2, 3) if d not in residual_by_depth]
    if missing:
        return ChainGate(passed=False, detail=f"no residual at depth {missing} -- cannot test")

    med = {d: float(np.median(residual_by_depth[d])) for d in (1, 2, 3)}
    base = max(med[1], 1e-9)
    growth = med[3] / base
    if growth > CHAIN_GROWTH_LIMIT:
        return ChainGate(
            passed=False, by_depth=med, growth=growth,
            detail=f"residual grows {growth:.2f}x from MCP to DIP (limit "
                   f"{CHAIN_GROWTH_LIMIT}) -- the compounding signature the logbook flagged",
        )
    return ChainGate(passed=True, by_depth=med, growth=growth,
                     detail=f"flat with depth ({growth:.2f}x): local error, not compounding")


def referee(
    fit_residuals: dict[str, np.ndarray],
    stature: float,
    finger_depths: dict[str, dict[int, np.ndarray]] | None = None,
) -> RefereeVerdict:
    """Judge a completed ANNY fit across all five regions.

    `fit_residuals` maps each region to per-keypoint distances in the rig's units, as produced
    by an AnnyInverter solve. `finger_depths` carries the per-depth residuals the hand gate
    needs, keyed by hand then by depth.

    Median, not mean and not max: one badly-placed joint should neither condemn an otherwise
    reachable pose nor average itself away against a well-fitted torso. `preflight_audit`
    chooses the median for the same reason.

    A region in REFEREED but missing from `fit_residuals` yields NOT_RUN rather than a pass.
    An absent measurement is the case where silence is most easily mistaken for success.
    """
    missing = [r for r in REFEREED if r not in fit_residuals]
    if missing:
        return RefereeVerdict(
            call=RefereeCall.NOT_RUN,
            detail=f"no fit for {', '.join(missing)} -- treated as failure, not skipped",
        )

    pct = {
        r: 100.0 * float(np.median(fit_residuals[r])) / max(stature, 1e-9)
        for r in REFEREED
    }

    # Reachability first: an impossible pose is rejected whatever the hands say.
    for r in ("body", "feet", "face"):
        if pct[r] / 100.0 > IMPOSSIBLE_RESIDUAL:
            return RefereeVerdict(
                call=RefereeCall.IMPOSSIBLE, residual_pct=pct,
                detail=f"{r} residual {pct[r]:.2f}% of stature exceeds "
                       f"{IMPOSSIBLE_RESIDUAL * 100:.0f}%",
            )
    for h in HANDS:
        if pct[h] / 100.0 > HAND_RESIDUAL:
            return RefereeVerdict(
                call=RefereeCall.IMPOSSIBLE, residual_pct=pct,
                detail=f"{h} residual {pct[h]:.2f}% of stature exceeds "
                       f"{HAND_RESIDUAL * 100:.1f}%",
            )

    # The hand numbers are only admissible if the finger path passed its depth gate. Running
    # the gate is mandatory: absent depth data is a failed gate, not a waived one.
    gates: dict[str, ChainGate] = {}
    untrusted: list[str] = []
    for h in HANDS:
        g = chain_gate((finger_depths or {}).get(h, {}))
        gates[h] = g
        if not g.passed:
            untrusted.append(h)
    if untrusted:
        return RefereeVerdict(
            call=RefereeCall.HANDS_UNTRUSTED, residual_pct=pct,
            untrusted=tuple(untrusted), gates=gates,
            detail="; ".join(f"{h}: {gates[h].detail}" for h in untrusted),
        )
    return RefereeVerdict(call=RefereeCall.FITS, residual_pct=pct, gates=gates)


def _flat(n: int, v: float) -> np.ndarray:
    return np.full(n, v)


def _good_hands() -> dict[str, dict[int, np.ndarray]]:
    """Depth residuals that are flat -- ordinary noise, no compounding."""
    return {h: {1: _flat(5, 0.0030), 2: _flat(5, 0.0032), 3: _flat(5, 0.0034)}
            for h in HANDS}


def negative_controls() -> None:
    """Inputs the referee MUST reject.

    CLAUDE.md rule 2 -- a check that passes on known-broken input is decoration, certifying
    the defect. Each control targets a distinct failure, because one control only ever proves
    the referee is not uniformly permissive.
    """
    # 1. A pose no human reaches.
    broken = {r: _flat(n, 0.09) for r, n in
              (("body", 17), ("feet", 6), ("face", 68), ("left_hand", 21), ("right_hand", 21))}
    v = referee(broken, stature=1.0, finger_depths=_good_hands())
    assert v.call is RefereeCall.IMPOSSIBLE, "referee accepted an impossible pose"
    print(f"  impossible pose   -> {v.call.name}: {v.detail}")

    # 2. The logbook's compounding finger error -- small at the knuckle, large at the tip.
    ok = {"body": _flat(17, 0.004), "feet": _flat(6, 0.004), "face": _flat(68, 0.002),
          "left_hand": _flat(21, 0.003), "right_hand": _flat(21, 0.003)}
    compounding = {h: {1: _flat(5, 0.0015), 2: _flat(5, 0.0032), 3: _flat(5, 0.0068)}
                   for h in HANDS}
    v = referee(ok, stature=1.0, finger_depths=compounding)
    assert v.call is RefereeCall.HANDS_UNTRUSTED, "referee trusted a compounding finger chain"
    assert set(v.untrusted) == set(HANDS)
    print(f"  compounding chain -> {v.call.name}: {v.detail}")

    # 3. A hand within the body tolerance but outside the hand tolerance. Without a separate
    #    hand bar this passes, which is the whole reason HAND_RESIDUAL exists.
    loose = dict(ok, left_hand=_flat(21, 0.012))
    v = referee(loose, stature=1.0, finger_depths=_good_hands())
    assert v.call is RefereeCall.IMPOSSIBLE, "hand judged at the body tolerance"
    print(f"  hand at body bar  -> {v.call.name}: {v.detail}")

    # 4. No depth data at all. Must fail the gate rather than waive it.
    v = referee(ok, stature=1.0, finger_depths=None)
    assert v.call is RefereeCall.HANDS_UNTRUSTED, "a missing gate was waived"
    print(f"  gate not run      -> {v.call.name}")

    # 5. A missing region must not read as a pass.
    v = referee({"body": _flat(17, 0.004)}, stature=1.0, finger_depths=_good_hands())
    assert v.call is RefereeCall.NOT_RUN, "a missing fit was treated as a pass"
    print(f"  missing region    -> {v.call.name}: {v.detail}")


if __name__ == "__main__":
    print("negative controls (each MUST be rejected):")
    negative_controls()

    ok = {"body": _flat(17, 0.004), "feet": _flat(6, 0.004), "face": _flat(68, 0.002),
          "left_hand": _flat(21, 0.003), "right_hand": _flat(21, 0.003)}
    v = referee(ok, stature=1.0, finger_depths=_good_hands())
    assert v.call is RefereeCall.FITS, "referee rejected a reachable pose"
    print("\npositive control (MUST be accepted):")
    print(f"  reachable pose    -> {v.call.name}")
    for r in REFEREED:
        print(f"    {r:<11} {v.residual_pct[r]:.3f}% of stature = {v.household(r)}")
    print(f"  finger gate       -> {v.gates['left_hand'].detail}")
