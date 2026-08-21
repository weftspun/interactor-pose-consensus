# pose-consensus

**The consensus panel is dead.** What remains is the parametric referee: fit ANNY/SOMA-X to a
set of keypoints and judge whether a human body could hold that pose.

The name no longer describes the contents. It is kept for now because the repository was never
placed in `default.xml`, so renaming is free whenever the surviving part gets a settled scope.

## What was removed, and why

Retractions stay next to what they retract, so this is recorded rather than quietly deleted.
`git log` has the code if it is ever wanted back.

The panel ran three to five independent pose estimators over the same image and emitted
keypoints only when they agreed. It was built to answer: _what pose is in this picture?_ — for
pictures we did not author. Two things killed it.

**It was solving a problem we chose to stop having.** The corpus plan inverted: instead of
estimating poses from images, ANNY _originates_ the pose, renders it, and a diffusion model
stylizes the render with the geometry pinned. The pose label is then true by construction. You
do not convene three estimators to adjudicate a pose you authored, so the panel had no question
left to answer.

**And it could not be assembled anyway.** Licence checking removed every candidate for the
third seat:

| backend             | outcome                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| MediaPipe           | Apache-2.0, verified from the release — the one survivor                 |
| ViTPose             | Apache-2.0, verified from vendored `transformers` source                 |
| Sapiens             | CC-BY-NC — the exact class `filter_coco_licenses.py` drops               |
| DWPose / RTMW       | Apache-2.0 weights, but trained on UBody, distributed only behind a form |
| GEM-X               | NVlabs, terms unresolved                                                 |
| SDPose              | OpenRAIL-M — decided by role since; passthrough, so FLAGGED not DENIED   |
| OpenPose, AlphaPose | non-commercial                                                           |
| RF-DETR             | COCO-17 head, no wholebody checkpoint                                    |

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
everything agrees on it, because the question is not _do they concur_ but _could a body do
this at all_.

The fit residual is the measurement, reported as a percentage of stature — matching
`preflight_audit.py`, where an absolute tolerance "would pass every spot-check and ship"
because the same millimetre error is negligible on an adult and disqualifying on a child.

**Hands are refereed.** An earlier draft abstained on them, citing
`anny-pose-retarget-work`'s record that the finger chain was never independently verified.
That was wrong: "never verified" is not "known broken", `handwear` is one of the 24 tags
See-Through must separate, and declining to measure does not make a hand correct — it makes
the defect invisible. Instead the finger path is gated on the specific defect the logbook
described. A per-joint convention error _compounds with depth_ while ordinary fit noise stays
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
  -> depth-conditioned generator     captions supply language, ANNY supplies shape
  -> MediaPipe reads it back         pose drift
  -> referee                         reachable? hands intact?
```

**Retracted: that row read `ControlNet + JuggernautXL`.** CLAUDE.md closes it, and not on a
technicality. JuggernautXL is OpenRAIL-M, and the OpenRAIL rule turns on what the model is
_for_: rendering an ANNY pose and generating over it is _operationally_ passthrough — our own
asset in, geometry preserved, appearance changed — but its destination is a training corpus,
which is the generator case. Destination wins, because destination is what the restriction is
about. Once the terms are inside somebody's weights there is nothing left to inspect.

So the route is closed and the referee is unaffected. Drift is measured against the render, not
against the generator, so replacing the generator does not change what is measured. What it
changes is the interface: `python/backend_licenses.py` now admits three depth-conditionable,
licence-clean candidates — `z-image-turbo`, `qwen-image`, `kolors` — and choosing among them is
open. `z-image-turbo` is the only one whose weights fit an 8 GB device at INT4, and CLAUDE.md
records that the first two are one lineage wearing two names, so `kolors` is the only row that
addresses common-mode exposure at all.

One backend, not a panel: drift is a metric, and a metric needs one instrument. MediaPipe is
Apache-2.0, verified, and runs on CPU without contending with the generator for the GPU.

The hand gate matters most here, and it survives the change of generator intact. A corpus was
blocklisted for malformed hands. Generating its replacement with a model that malforms them
differently is not an improvement, it is the same defect with our provenance on it. That
argument was written about SDXL and does not depend on SDXL: no candidate above has had its
hand quality measured, so the gate is what stands between an unmeasured generator and a corpus.
A depth control pins limb geometry, but a hand is a few dozen pixels in the depth buffer, so
fingers are effectively unconstrained whatever conditions the rest of the body.

## Open, and load-bearing

**The pose corpus is locomotion-only.** 810 clips across eight motion types — idle, forward,
backward and sideways walk and run, and turn — plus 22 O3DE clips and two UE4 getups. Anime
illustrations are seldom mid-stride. Whether these poses are useful to See-Through at all is
unsettled, and it sits upstream of everything above: if the answer is no, the pose corpus is
the wrong corpus and generating from it produces the wrong distribution.

`python/backend_licenses.py` survives, retargeted from panel backends to the remaining
dependencies. This paragraph used to leave JuggernautXL's terms for producing a _training
corpus_ — as against producing images — recorded as unread. They are read now, and the reading
is above: OpenRAIL-M is blocked as a generator and permitted as passthrough, so the question was
never about JuggernautXL's terms specifically but about what the output is for.

**The ControlNet weights now have their own check.** That sentence used to end this paragraph
as a to-do, and the to-do was the defect: `Backend` carried two licences and a generator has
three. Control weights are a separate release under separate terms, so a base-licence check
admits FLUX.1 [schnell] (Apache-2.0) and HiDream-I1 (MIT), neither of which has a readable way
to be conditioned. Every corpus use renders an ANNY pose and requires the generated image to
keep that geometry, so the control is the term that decides.

`control_license` is now a third axis on `Backend`, `gate()` takes the worst of three, and
`GENERATORS` is gated rather than merely listed. Running the module prints:

```
depth-conditionable and licence-clean (3): z-image-turbo, qwen-image, kolors
```

which re-derives CLAUDE.md's survey from the table instead of recalling it. Four rows fail on
the control alone with a clean base — `qwen-image-edit`, `flux-schnell`, `hidream-i1`, `sana`
— and `sana` fails on `NO-DEPTH-CHECKPOINT` rather than on terms, because its licence is clean
end to end and its released weights are HED only. That distinction is kept because the remedy
differs: a licence failure is permanent, a missing checkpoint is a training job someone could
cost.

Corpus terms are `UNVERIFIED` for **every** generator, so none is admitted. That is condition 1
of the synthetic-data rule going unanswered by the whole field at once, not an oversight in one
row, and the output says so rather than leaving it to be inferred.

**Two models one prefix apart have opposite verdicts.** `juggernaut-z-image` is CC-BY-NC-4.0,
RunDiffusion's finetune. `z-image-turbo` is Apache-2.0, Tongyi-MAI's base. A search for
`z-image` returns one of each. Control 1 asserts they classify differently, so the distinction
is executed rather than commented.

Six negative controls ship with it, in `python/test_backend_licenses.py`. Control 2 runs the
retracted two-axis gate and asserts it MISSES what the three-axis gate catches — it still
refuses `flux-schnell`, but on the corpus, so a reader who later verified the corpus would
admit an unconditionable model. Each control was checked against a re-introduced defect:

```
control axis ignored                  -> 2, 4, 6 fail
z-image-turbo read as the NC finetune -> 1 fails
untagged control read as clear        -> 5 fails
unsurveyed control renders OK         -> 3 fails
role not passed to gate()             -> 7 fails
one generator row forgets its role    -> 8 fails
```

Six defects, eight controls, and no control fires on a defect it does not target.

**Controls 7 and 8 close a rule that was implemented and unreachable.** `classify` has always
taken a role, `Role` has always been defined, and `gate` — its only caller — never passed one,
because `Backend` had no field to carry it. So every OpenRAIL row took the strict branch by
accident rather than by the rule: `sdpose` read DENIED for the right verdict and the wrong
reason, and would have kept reading DENIED if the rule had later gone the other way. `Backend`
carries `role` now and `gate` passes it on all three axes. No verdict in the table changed,
which is the point — the reasons did.

## Status

Referee implemented, controls passing. No generation has been run and no drift has been
measured.
