# pose-consensus

**The consensus panel is dead.** What remains is the parametric referee: fit ANNY/SOMA-X to a
set of keypoints and judge whether a human body could hold that pose.

The name no longer describes the contents. It is kept for now because the repository was never
placed in `default.xml`, so renaming is free whenever the surviving part gets a settled scope.

## What was removed, and why

Retractions stay next to what they retract, so this is recorded rather than quietly deleted.
`git log` has the code if it is ever wanted back.

The panel ran three to five independent pose estimators over the same image and emitted
keypoints only when they agreed. It was built to answer: *what pose is in this picture?* — for
pictures we did not author. Two things killed it.

**It was solving a problem we chose to stop having.** The corpus plan inverted: instead of
estimating poses from images, ANNY *originates* the pose, renders it, and a diffusion model
stylizes the render with the geometry pinned. The pose label is then true by construction. You
do not convene three estimators to adjudicate a pose you authored, so the panel had no question
left to answer.

**And it could not be assembled anyway.** Licence checking removed every candidate for the
third seat:

| backend | outcome |
|---|---|
| MediaPipe | Apache-2.0, verified from the release — the one survivor |
| ViTPose | Apache-2.0, verified from vendored `transformers` source |
| Sapiens | CC-BY-NC — the exact class `filter_coco_licenses.py` drops |
| DWPose / RTMW | Apache-2.0 weights, but trained on UBody, distributed only behind a form |
| GEM-X | NVlabs, terms unresolved |
| SDPose | OpenRAIL-M use-restrictions, undecided |
| OpenPose, AlphaPose | non-commercial |
| RF-DETR | COCO-17 head, no wholebody checkpoint |

Two verified backends is not a panel. Quorum needs three to tolerate a single outage.

**A third finding made the survivors weaker than they looked.** Both remaining backends are
discriminative and COCO-trained, so their errors correlate — a hard occlusion misleads them in
the same direction. Agreement between two COCO-trained networks on a COCO-like image is much
weaker evidence than it appears, which is why majority voting was never the right rule here
even when there were enough voters for it.

## What survives: the referee

`python/soma_referee.py`. A parametric model **cannot represent an impossible skeleton** —
that property, not the vote, was the whole reason a parametric member was wanted. It does not
need GEM-X. It needs a parametric human, and ANNY is one we own outright: no licence question,
no gated corpus, no upstream checkpoint.

A referee is also stronger than a voter. A voter can be outvoted by two estimators making the
same correlated mistake. The referee cannot: an unreachable pose is rejected even when
everything agrees on it, because the question is not *do they concur* but *could a body do
this at all*.

The fit residual is the measurement, reported as a percentage of stature — matching
`preflight_audit.py`, where an absolute tolerance "would pass every spot-check and ship"
because the same millimetre error is negligible on an adult and disqualifying on a child.

**Hands are refereed.** An earlier draft abstained on them, citing
`anny-pose-retarget-work`'s record that the finger chain was never independently verified.
That was wrong: "never verified" is not "known broken", `handwear` is one of the 24 tags
See-Through must separate, and declining to measure does not make a hand correct — it makes
the defect invisible. Instead the finger path is gated on the specific defect the logbook
described. A per-joint convention error *compounds with depth* while ordinary fit noise stays
flat, so `chain_gate` compares residual at MCP, PIP and DIP. Flat passes. Climbing is marked
`HANDS_UNTRUSTED` and counted.

Five negative controls ship with it, each targeting a distinct failure, because one control
only proves the referee is not uniformly permissive:

```
impossible pose   -> IMPOSSIBLE       body residual 9.00% of stature exceeds 2%
compounding chain -> HANDS_UNTRUSTED  grows 4.53x from MCP to DIP (limit 1.6)
hand at body bar  -> IMPOSSIBLE       1.20% exceeds the 0.6% hand bar
gate not run      -> HANDS_UNTRUSTED  a missing gate is failed, never waived
missing region    -> NOT_RUN          an absent fit is not a pass
```

## What the referee is for now

Not ground truth — that comes from the authored pose. The referee measures whether
**generation preserved the geometry**:

```
ANNY pose (licence-clean mocap)
  -> depth render from the mesh      exact geometry, not an estimate
  -> ControlNet + JuggernautXL       captions supply language, ANNY supplies shape
  -> MediaPipe reads it back         pose drift
  -> referee                         reachable? hands intact?
```

One backend, not a panel: drift is a metric, and a metric needs one instrument. MediaPipe is
Apache-2.0, verified, and runs on CPU without contending with the generator for the GPU.

The hand gate matters most here. A corpus was blocklisted for malformed hands. Generating its
replacement with SDXL, whose hands are a known weak point, would reproduce the same defect with
our own provenance on it. ControlNet pins limb geometry but a hand is a few dozen pixels in the
depth buffer, so fingers are effectively unconstrained.

## Open, and load-bearing

**The pose corpus is locomotion-only.** 810 clips across eight motion types — idle, forward,
backward and sideways walk and run, and turn — plus 22 O3DE clips and two UE4 getups. Anime
illustrations are seldom mid-stride. Whether these poses are useful to See-Through at all is
unsettled, and it sits upstream of everything above: if the answer is no, the pose corpus is
the wrong corpus and generating from it produces the wrong distribution.

`python/backend_licenses.py` survives, retargeted from panel backends to the remaining
dependencies. JuggernautXL's terms for producing a *training corpus* — as against producing
images — are unread, and several RAIL-family and Civitai licences distinguish the two. The
ControlNet weights need their own check.

## Status

Referee implemented, controls passing. No generation has been run and no drift has been
measured.
