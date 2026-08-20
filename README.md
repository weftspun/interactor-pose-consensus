# pose-consensus

Four independent pose estimators behind one interface. A keypoint set is emitted only when
they agree; **disagreement is a fault, not an average.**

## Why

Estimated keypoints are model output. Under the workspace's data-hygiene rule they are
*generated* synthetic, and the hazard that rule exists to prevent is a corpus that inherits a
model's errors as if they were facts. One estimator's guess is a claim. Four independent
estimators landing in the same place is evidence.

The alternative — averaging, or trusting a single "best" model — launders disagreement into a
plausible number. A mean of two confident and incompatible skeletons is a skeleton no estimator
believed. That is the failure this exists to make impossible.

## The estimators

Face/head/body throughout — body-only is not enough, because `face`, `irides`, `eyebrow`,
`eyewhite`, `eyelash`, `nose`, `mouth`, `ears` and `handwear` are nine of the 24 tags and a
17-point skeleton constrains none of them.

| # | backend | keypoints | lineage |
|---|---|---|---|
| 1 | **SDPose-Wholebody** | 133 (17+6 feet+68 face+42 hands) | Stable Diffusion priors |
| 2 | **MediaPipe Holistic** | 33 + 468 face + 21x2 hands | classical CNN, on-device |
| 3 | **GEM-X** | SOMA-X coefficients | **parametric**, not discriminative |
| 4 | **DWPose / RTMW** | 133 | MMPose, distillation-trained |
| 5 | **Sapiens** | 308 | Meta ViT, Humans-300M pretraining |

**RF-DETR is excluded**, not forgotten. `rf-detr-cpp/models/heads/keypoints.py` is
`num_keypoints_per_class`, COCO-17 by default, and `docs/architecture.md` records the keypoint
training path as "training-only, not needed for the inference port". A wholebody head means a
new output layer, a retrain upstream, and a re-port -- and no public wholebody checkpoint
exists. It is listed here so the exclusion is a decision on the record rather than an omission.

Lineage diversity is the whole point, not count. Correlated failure is what defeats a panel:
five models sharing an ancestor agree on their shared mistakes and certify them. Hence one
diffusion-prior model, one classical CNN, one parametric fit, one distillation-trained, one
large-pretraining ViT.

## Panel size

Odd sizes, because even ones tie:

| N | may be unavailable | quorum |
|---|---|---|
| 3 | 1 | 2 |
| 5 | 2 | 3 |
| 7 | 3 | 4 |

**This ladder governs availability, not agreement.** Under `Rule.UNANIMOUS` every backend that
responds must still concur -- N=5/quorum=3 means "at least three voted, and all who voted
agreed", not "three outvoted two". `Rule.MAJORITY` gives the classical behaviour and carries
the correlated-failure caveat documented on `adjudicate`.

## Agreement is OKS, not pixels

Object Keypoint Similarity — COCO's own metric — with per-keypoint sigmas and normalisation by
person scale. A pixel threshold would call a wrist misplaced by 10px on a 100px figure the same
as on a 2000px one, and would treat an ankle (loosely localised even by humans, sigma 0.089)
as strictly as an eye (sigma 0.025).

`rf-detr-cpp/keypoint_oks.py` already implements the matching cost; this reuses it rather than
writing a second definition that can drift from the first.

**Fault conditions**, any of which rejects the sample:

1. pairwise OKS between any two backends below the floor
2. a backend finds a different number of people
3. a backend fails or times out — a missing opinion is not a passing vote
4. fewer than the configured quorum of backends available at all

Condition 3 matters most and is the easiest to get wrong: a panel that silently proceeds when
one member is unavailable is a smaller panel that still reports full confidence.

## Why pythonx

The four backends are Python (torch, mediapipe, ONNX). Shelling out four times per image pays
four interpreter starts and four model loads. `pythonx` embeds one CPython in the BEAM, so all
four stay resident and warm.

One CPython means one GIL, so calls are serialised by a GenServer rather than contended. That
is correct for this workload anyway: the panel is a batch process over a corpus, not a
low-latency service, and the four backends compete for one GPU regardless.

## Status

Scaffold. No backend is wired yet, and no measurement has been taken.
