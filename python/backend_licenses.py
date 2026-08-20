"""License gate for panel backends, mirroring `filter_coco_licenses.py`.

The image corpus is already license-filtered: COCO photos whose own licence forbids
commercial use or derivatives are dropped before anything trains on them. A model's WEIGHTS
are the same kind of input and were not being filtered at all -- the first roster proposed
here included Sapiens, which is CC-BY-NC, the exact class that filter rejects. Recall is not
a licence check, so this makes it a command that fails.

Two licences, not one, because a checkpoint carries both:

  * CODE/WEIGHTS -- what the release says about the artefact itself.
  * TRAINING CORPUS -- what the artefact inherits. A permissively-licensed checkpoint trained
    on a research-only corpus is the `DeepFashion` case already blocklisted upstream: a
    re-export of terms that do not permit the use.

`UNVERIFIED` is a FAIL, never a skip. CLAUDE.md rule 3: a silent skip reads exactly like a
pass, so an unchecked backend is reported and counted rather than quietly admitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Mirrors filter_coco_licenses.COMMERCIAL_OK. Commercial use and derivatives must both be
# permitted -- training produces a derivative, so ND fails as surely as NC.
ALLOWED = {"Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0", "COCO-WholeBody"}
#: Permitted but flagged. Share-alike obligations on derived MODELS are legally unsettled;
#: the COCO filter keeps these behind a `share_alike` column rather than deciding for the org.
FLAGGED = {"CC-BY-SA-4.0", "OpenRAIL-M", "CreativeML-OpenRAIL-M", "CreativeML-OpenRAIL++-M"}
#: Distributed only behind a registration form. The terms cannot be read without accepting
#: them, so they cannot be gated on -- treated as DENIED until someone accepts and reports.
GATED = {"GATED-UBody", "GATED-acceptance"}


class Status(str, Enum):
    OK = "permits commercial use and derivatives"
    FLAGGED = "permitted but carries obligations or use-restrictions; needs a decision"
    DENIED = "forbids commercial use or derivatives"
    UNVERIFIED = "not checked against the release -- treated as a failure, not a skip"


@dataclass(frozen=True)
class Backend:
    name: str
    weights_license: str
    corpus_license: str
    #: True only when a human read the actual LICENSE file, not a recollection of it.
    checked: bool = False
    note: str = ""


#: The roster. `checked=False` everywhere until each is read from its release, which is the
#: point: this table starts out failing.
ROSTER = [
    # --- the panel ---
    Backend("mediapipe", "Apache-2.0", "Apache-2.0", checked=True,
            note="VERIFIED from the release LICENSE. One vendored exception under "
                 "tasks/cc/text/.../utf/ (Lucent/Plan9), unrelated to pose."),
    Backend("vitpose", "Apache-2.0", "COCO-WholeBody", checked=True,
            note="VERIFIED from vendored transformers source (Univ. of Sydney + HuggingFace). "
                 "Trained on COCO -- see the val2017 note below."),
    Backend("dwpose", "Apache-2.0", "GATED-UBody", checked=True,
            note="Weights Apache-2.0, VERIFIED. Corpus is COCO-WholeBody + UBody, and UBody "
                 "is distributed only through a Google Form -- terms cannot be read without "
                 "accepting them. Registration-gated corpora are the DeepFashion pattern "
                 "already blocklisted upstream: a permissive checkpoint re-exporting terms "
                 "nobody has read."),
    Backend("rtmw", "Apache-2.0", "GATED-UBody", checked=True,
            note="MMPose model zoo. Same UBody dependency as DWPose, and NOT independent of "
                 "it -- DWPose distils from an RTMPose teacher. Seating both would seat one "
                 "opinion twice."),

    # --- excluded, kept with verdicts attached ---
    Backend("gemx", "NVIDIA-UNVERIFIED", "UNVERIFIED",
            note="NVlabs/GEM-X. NVIDIA research code often ships under the NVIDIA Source "
                 "Code License (non-commercial). UNRESOLVED AND USED IN ANGER: "
                 "run_gemx_batch.sh already runs commercially-filtered COCO through it."),
    Backend("sdpose", "OpenRAIL-M", "UNVERIFIED",
            note="Stable Diffusion derived. OpenRAIL-M is neither NC nor ND, so the COCO "
                 "filter's categories do not decide it -- a policy call, not a lookup."),
    Backend("sapiens", "CC-BY-NC-4.0", "UNVERIFIED", checked=True,
            note="EXCLUDED: NC is the exact class filter_coco_licenses drops"),
    Backend("openpose", "NON-COMMERCIAL", "UNVERIFIED", checked=True,
            note="EXCLUDED: CMU academic licence; CMU also blocklisted for provenance"),
    Backend("alphapose", "NON-COMMERCIAL", "UNVERIFIED", checked=True,
            note="EXCLUDED: commercial use requires a separate licence"),
    Backend("rfdetr", "Apache-2.0", "UNVERIFIED", checked=True,
            note="EXCLUDED on capability, not licence: COCO-17 head, no wholebody checkpoint"),
]

#: Generators, checked against the HuggingFace API rather than model cards, 2026-08-19.
#: The licence cliff falls exactly at XI: every SDXL Juggernaut up to X v10 is OpenRAIL-M,
#: and the whole XI and Z line is non-commercial. So the vendor's own "why upgrade" table is,
#: licence-wise, a list of models that cannot be used -- newer is strictly worse here.
GENERATORS = [
    Backend("juggernaut-xl-v9", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            note="216,745 downloads -- the actually-supported successor. Best quality pick."),
    Backend("juggernaut-x-v10", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            note="17,255 downloads. X-generation, same licence class as v6."),
    Backend("juggernaut-xl-lightning", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            note="12,948 downloads. FEW-STEP -- lands on the step-reduction lever, which was "
                 "ranked the largest single payoff. Best-exercised of the distilled options."),
    Backend("juggernaut-x-hyper", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            note="352 downloads. FEW-STEP and X-generation, but thinly used next to Lightning."),
    Backend("juggernaut-xl-v6", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            note="CURRENT. See-Through's LayerDiffuse base, via the frankjoshua mirror."),
    Backend("juggernaut-xi-v11", "CC-BY-NC-ND-4.0", "UNVERIFIED", checked=True,
            note="DENIED twice. NC is the class that dropped Sapiens; ND is rejected by "
                 "filter_coco_licenses' own comment -- training/derived work is a derivative."),
    Backend("juggernaut-xi-lightning", "CC-BY-NC-ND-4.0", "UNVERIFIED", checked=True,
            note="DENIED. The newest few-step model, and unusable -- Lightning/Hyper below XI "
                 "are the only distilled options that survive."),
    Backend("juggernaut-z-image", "CC-BY-NC-4.0", "UNVERIFIED", checked=True,
            note="DENIED. Also Lumina-Image-2, which would strand the SDXL ggml port."),
    Backend("juggernaut-z-image-fast", "CC-BY-NC-4.0", "UNVERIFIED", checked=True,
            note="DENIED, same line."),
    Backend("juggernaut-pro-flux", "GATED-acceptance", "UNVERIFIED", checked=True,
            note="DENIED. HF returns 401 -- terms unreadable without accepting them, the UBody "
                 "situation again. FLUX.1-dev underneath is non-commercial besides."),
]

#: OpenRAIL-M stays FLAGGED rather than ALLOWED, and that is not a formality. It is neither NC
#: nor ND, so the COCO filter's categories do not decide it, and every RunDiffusion model adds
#: "may not be deployed behind paid API services without explicit licensing". That policy call
#: has not been made -- and it already applies to v6, which is in use today.
#:
#: Unresolved and separate: whether generating a TRAINING CORPUS is permitted, as against
#: generating images. Several RAIL-family licences draw that line explicitly.


def classify(license_id: str) -> Status:
    if license_id == "UNVERIFIED":
        return Status.UNVERIFIED
    if license_id in ALLOWED:
        return Status.OK
    if license_id in FLAGGED:
        return Status.FLAGGED
    if license_id in GATED:
        return Status.DENIED
    return Status.DENIED


def gate(roster=ROSTER) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (admitted, [(name, reason)]) for the rest.

    A backend is admitted only when BOTH its weights and its training corpus clear, and only
    when someone actually checked. The worse of the two statuses decides, because a permissive
    checkpoint trained on a restricted corpus is restricted.
    """
    admitted, refused = [], []
    for b in roster:
        w, c = classify(b.weights_license), classify(b.corpus_license)
        worst = max((w, c), key=lambda s: [Status.OK, Status.FLAGGED,
                                           Status.UNVERIFIED, Status.DENIED].index(s))
        if worst is Status.OK and b.checked:
            admitted.append(b.name)
        elif worst is Status.OK:
            refused.append((b.name, "licences look clear but were never verified against the release"))
        else:
            refused.append((b.name, f"{worst.name}: weights={b.weights_license} corpus={b.corpus_license}"))
    return admitted, refused


if __name__ == "__main__":
    ok, no = gate()
    print(f"admitted ({len(ok)}): {', '.join(ok) or 'NONE'}")
    for name, why in no:
        print(f"  refused  {name}: {why}")
    quorum = len(ok) // 2 + 1
    print(f"\npanel N={len(ok)}, majority quorum={quorum}")
    if len(ok) < 3:
        raise SystemExit(
            f"FAIL: {len(ok)} licence-clear backends. A panel needs at least 3 to tolerate "
            f"one outage. This is the intended initial state -- verify the roster above."
        )
