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
FLAGGED = {"CC-BY-SA-4.0"}
#: OpenRAIL-M depends on ROLE, not on the weights. Blocked as a generator, permitted as
#: passthrough -- see CLAUDE.md. `classify` therefore takes a role, and a caller that does not
#: state one gets the strict answer rather than a convenient default.
#:
#: RETRACTED, kept in place: for as long as this comment has existed the role was UNREACHABLE.
#: `classify` accepted one, `Role` was defined, and `gate` -- the only caller -- never passed
#: it, because `Backend` had no field to carry it. Every OpenRAIL row therefore took the strict
#: branch by accident rather than by the rule, and `sdpose` read DENIED for the right verdict
#: and the wrong reason. A rule that looks enforced and is not is the failure this module was
#: written to catch, one list further down the same file.
OPENRAIL = {"OpenRAIL-M", "CreativeML-OpenRAIL-M", "CreativeML-OpenRAIL++-M"}
#: Distributed only behind a registration form. The terms cannot be read without accepting
#: them, so they cannot be gated on -- treated as DENIED until someone accepts and reports.
GATED = {"GATED-UBody", "GATED-acceptance"}
#: Terms exist but are not stated: `license:other` on a HuggingFace repo, or no licence tag at
#: all. Distinct from GATED -- nothing is being withheld behind a form, there is simply nothing
#: to read. Same verdict for the same reason: a term nobody can read cannot be gated on. This
#: is what disqualifies every FLUX ControlNet and HiDream-I1's only conditioning.
UNREADABLE = {"license:other", "NO-LICENCE-TAG"}
#: Not a licence failure at all. The control checkpoint the pipeline needs was never published,
#: so the licence question does not arise. SANA's architecture supports depth and its released
#: weights are HED only; Qwen-Image-Edit's packaged interface takes `strength` and no control.
#: CLAUDE.md: a generator that cannot take a depth control is not a generator for this pipeline,
#: however clean its terms are. Kept apart from DENIED because the remedy differs -- a licence
#: failure is permanent, a missing checkpoint is a training job someone could cost.
MISSING = {"NO-DEPTH-CHECKPOINT"}
#: Backends that condition nothing. The pose estimators in ROSTER take an image, not a control.
NOT_APPLICABLE = "NOT-APPLICABLE"


class Status(str, Enum):
    OK = "permits commercial use and derivatives"
    FLAGGED = "permitted but carries obligations or use-restrictions; needs a decision"
    DENIED = "forbids commercial use or derivatives"
    UNVERIFIED = "not checked against the release -- treated as a failure, not a skip"


class Role(str, Enum):
    """What the model is FOR. OpenRAIL-M turns on this and nothing else.

    PASSTHROUGH transforms an input the user supplied and returns it -- the provenance stays
    with one artefact. GENERATOR samples content that becomes a corpus, which propagates the
    terms into weights where no later check can see them.
    """

    PASSTHROUGH = "transforms a supplied input and returns it"
    GENERATOR = "samples content that will be trained on"


@dataclass(frozen=True)
class Backend:
    name: str
    weights_license: str
    corpus_license: str
    #: True only when a human read the actual LICENSE file, not a recollection of it.
    checked: bool = False
    note: str = ""
    #: A THIRD licence, and the CLAUDE.md survey turns on it rather than on the base. Control
    #: weights are a separate release under separate terms, and one control says nothing about
    #: another even under the same owner -- Kolors ships `-Depth` as Apache-2.0 while its
    #: sibling `-Pose` carries no tag at all. A base licence alone admits FLUX.1 [schnell] and
    #: HiDream-I1, neither of which has a readable way to be conditioned.
    control_license: str = NOT_APPLICABLE
    #: What the model is FOR. Decides OpenRAIL-M and nothing else. `None` is the strict answer:
    #: a row that has not said what it is for has not earned the passthrough exemption.
    role: "Role" = None


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
    Backend("sdpose", "OpenRAIL-M", "UNVERIFIED", role=None,
            note="Stable Diffusion derived. OpenRAIL-M is neither NC nor ND, so the COCO "
                 "filter's categories do not decide it. The note here used to end 'a policy "
                 "call, not a lookup' -- CLAUDE.md has since made the call, so it IS a lookup "
                 "now: blocked as a generator, permitted as passthrough. Reading a pose off a "
                 "supplied image is passthrough, so `role=Role.PASSTHROUGH` would make this "
                 "FLAGGED rather than DENIED. Left at None deliberately: the panel is dead, "
                 "nothing calls SDPose, and asserting a role for a backend nobody runs would "
                 "be recording a decision that was never needed. The field is the point."),
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
            role=Role.GENERATOR,
            note="216,745 downloads -- the actually-supported successor. Best quality pick."),
    Backend("juggernaut-x-v10", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="17,255 downloads. X-generation, same licence class as v6."),
    Backend("juggernaut-xl-lightning", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="12,948 downloads. FEW-STEP -- lands on the step-reduction lever, which was "
                 "ranked the largest single payoff. Best-exercised of the distilled options."),
    Backend("juggernaut-x-hyper", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="352 downloads. FEW-STEP and X-generation, but thinly used next to Lightning."),
    Backend("juggernaut-xl-v6", "CreativeML-OpenRAIL-M", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="CURRENT. See-Through's LayerDiffuse base, via the frankjoshua mirror."),
    Backend("juggernaut-xi-v11", "CC-BY-NC-ND-4.0", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="DENIED twice. NC is the class that dropped Sapiens; ND is rejected by "
                 "filter_coco_licenses' own comment -- training/derived work is a derivative."),
    Backend("juggernaut-xi-lightning", "CC-BY-NC-ND-4.0", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="DENIED. The newest few-step model, and unusable -- Lightning/Hyper below XI "
                 "are the only distilled options that survive."),
    Backend("juggernaut-z-image", "CC-BY-NC-4.0", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="DENIED. Also Lumina-Image-2, which would strand the SDXL ggml port. "
                 "NOT `z-image-turbo` BELOW: this is RunDiffusion's NC finetune, that is "
                 "Tongyi-MAI's Apache-2.0 base. The names collide and the verdicts invert, "
                 "so a search for 'z-image' returns one of each."),
    Backend("juggernaut-z-image-fast", "CC-BY-NC-4.0", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="DENIED, same line. Same collision as the row above."),
    Backend("juggernaut-pro-flux", "GATED-acceptance", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            note="DENIED. HF returns 401 -- terms unreadable without accepting them, the UBody "
                 "situation again. FLUX.1-dev underneath is non-commercial besides."),

    # --- the depth-conditioning survey, a DIFFERENT question from the Juggernaut line above.
    # Every corpus use renders an ANNY pose and requires the generated image to keep that
    # geometry, so `control_license` decides these rows and the base licence does not. Corpus
    # is UNVERIFIED across the board and that is not laziness: none of these publish their
    # training data terms, which is condition 1 of the synthetic-data rule going unanswered by
    # every candidate at once. Recorded so the uniformity is visible rather than inferred.
    Backend("z-image-turbo", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="Apache-2.0",
            note="6B S3-DiT, 8 NFEs, Tongyi-MAI. Control is "
                 "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 -- Apache-2.0, DEPTH and "
                 "pose among its conditions, 15 blocks + 2 refiner, 70k steps. UNCHECKED: "
                 "model card read 2026-08-21, LICENSE file was not. The only candidate whose "
                 "weights fit an 8 GB device at INT4, and the only one that restores the depth "
                 "control DETAILS.md records as absent on the Qwen-Image-Edit path."),
    Backend("qwen-image", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="Apache-2.0",
            note="20B. Union plus a DEDICATED depth model, several independent maintainers. "
                 "Clears on terms and does not fit 8 GB: the DiT alone is ~10 GB at pure INT4."),
    Backend("qwen-image-edit", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="NO-DEPTH-CHECKPOINT",
            note="CURRENT, as `qwen_q4_k_m_image_edit`. The packaged interface takes image, "
                 "instruction, strength, steps, seed -- geometry preservation rests on "
                 "`strength` alone. DETAILS.md accepts this as a cost; the gate does not, "
                 "because RFD 1079 requires a depth control wherever generated geometry must "
                 "match an authored pose. Same base licence as `qwen-image`, opposite verdict."),
    Backend("kolors", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="Apache-2.0",
            note="The only non-Alibaba row here, so the only one that addresses common-mode "
                 "exposure. Costs: `Kolors-ControlNet-Depth` declares `ControlNetModel_JQ`, a "
                 "bespoke class diffusers has no loader for, at ~150 downloads. Its sibling "
                 "`-Pose` carries NO licence tag -- the reason `control_license` is per-row "
                 "and not per-owner. Cannot borrow SDXL's controls: "
                 "projection_class_embeddings_input_dim 5632 vs 2816."),
    Backend("flux-schnell", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="license:other",
            note="The row that proves a base licence is not sufficient. Apache-2.0 and 4-step "
                 "distilled, and every FLUX ControlNet -- InstantX Union and Canny, "
                 "Shakker-Labs Union-Pro and Depth -- is tagged `license:other` AND trained "
                 "against [dev], which is non-commercial. Loading one onto schnell propagates "
                 "[dev]'s terms and mismatches guidance behaviour besides."),
    Backend("flux-dev", "CC-BY-NC", "UNVERIFIED", checked=True,
            role=Role.GENERATOR,
            control_license="license:other",
            note="DENIED on the base before the control matters. Ordinary NC exclusion, the "
                 "Sapiens class."),
    Backend("hidream-i1", "MIT", "UNVERIFIED", role=Role.GENERATOR,
            control_license="license:other",
            note="The most permissive BASE of any candidate reviewed, and unusable. Its only "
                 "conditioning is ControlNetLoRA/hidream-i1: one LoRA, not a family, "
                 "`license:other`, 14 downloads. Published under a different org, so a "
                 "name-scoped search missed it on the first pass."),
    Backend("sana", "Apache-2.0", "UNVERIFIED", role=Role.GENERATOR,
            control_license="NO-DEPTH-CHECKPOINT",
            note="Clean end to end, base AND control, and still refused. SanaControlNetModel "
                 "is in diffusers and the released weights are HED only. Edge conditioning "
                 "carries silhouette and contours with no depth ordering, so it cannot say "
                 "which limb is in front, and limb overlap is the hard part for a body. The "
                 "one row whose gap is WORK rather than terms -- cost it as a training job."),
]

#: OpenRAIL-M stays FLAGGED rather than ALLOWED, and that is not a formality. It is neither NC
#: nor ND, so the COCO filter's categories do not decide it, and every RunDiffusion model adds
#: "may not be deployed behind paid API services without explicit licensing". That policy call
#: has not been made -- and it already applies to v6, which is in use today.
#:
#: Unresolved and separate: whether generating a TRAINING CORPUS is permitted, as against
#: generating images. Several RAIL-family licences draw that line explicitly.


def classify(license_id: str, role: "Role" = None) -> Status:
    if license_id == "UNVERIFIED":
        return Status.UNVERIFIED
    if license_id in OPENRAIL:
        # No role stated is the strict answer, not the permissive one. A caller that has not
        # said what the model is for has not earned the exemption.
        return Status.FLAGGED if role is Role.PASSTHROUGH else Status.DENIED
    if license_id in ALLOWED:
        return Status.OK
    if license_id in FLAGGED:
        return Status.FLAGGED
    if license_id in GATED:
        return Status.DENIED
    if license_id in UNREADABLE:
        return Status.DENIED
    if license_id in MISSING:
        # Not a licence verdict. Reported through the same channel because the pipeline is
        # equally blocked either way, and separated in the message so the remedy stays legible.
        return Status.DENIED
    if license_id == NOT_APPLICABLE:
        # A backend that conditions nothing cannot fail a control check. Distinguished from
        # the fallthrough below on purpose: silently denying it would read as a verdict.
        return Status.OK
    return Status.DENIED


def gate(roster=ROSTER) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (admitted, [(name, reason)]) for the rest.

    A backend is admitted only when ALL THREE of its weights, its training corpus and its
    control weights clear, and only when someone actually checked. The worst of the three
    decides, because a permissive checkpoint trained on a restricted corpus is restricted, and
    a permissive base with an unreadable control cannot be conditioned.

    The third term is the one a base-licence check misses. FLUX.1 [schnell] is Apache-2.0 and
    HiDream-I1 is MIT; both are refused here, and only on the control.
    """
    admitted, refused = [], []
    for b in roster:
        w = classify(b.weights_license, b.role)
        c = classify(b.corpus_license, b.role)
        k = classify(b.control_license, b.role)
        worst = max((w, c, k), key=lambda s: [Status.OK, Status.FLAGGED,
                                              Status.UNVERIFIED, Status.DENIED].index(s))
        if worst is Status.OK and b.checked:
            admitted.append(b.name)
        elif worst is Status.OK:
            refused.append((b.name, "licences look clear but were never verified against the release"))
        else:
            # Name which of the three failed. A bare verdict sends the reader to the wrong
            # licence, and for the FLUX and HiDream rows the wrong licence is the clean one.
            failed = [f"{label}={lic}" for label, st, lic in
                      (("weights", w, b.weights_license),
                       ("corpus", c, b.corpus_license),
                       ("control", k, b.control_license)) if st is worst]
            refused.append((b.name, f"{worst.name} on {', '.join(failed)}"))
    return admitted, refused


def survey(roster=GENERATORS) -> list[tuple[str, Status, Status]]:
    """Return (name, base verdict, control verdict) per generator.

    Reported apart from `gate` because the corpus term is UNVERIFIED for every generator at
    once -- none publishes its training data terms -- so a single worst-of verdict says
    `UNVERIFIED` for all of them and distinguishes nothing. The two axes that DO differ are
    the base and the control, and the second is the one CLAUDE.md's survey turns on.

    The control LICENCE TOKEN is returned alongside its verdict, and printing the token is not
    decoration. `NOT-APPLICABLE` classifies OK so that a pose estimator is not failed for
    conditioning nothing -- but rendered as `OK` in a generator table it reads as "the control
    was checked and cleared", which is the silent-skip-reads-as-a-pass failure. The SDXL rows
    below have controls; nobody surveyed them.
    """
    return [(b.name, classify(b.weights_license, b.role),
             classify(b.control_license, b.role), b.control_license) for b in roster]


if __name__ == "__main__":
    ok, no = gate()
    print(f"admitted ({len(ok)}): {', '.join(ok) or 'NONE'}")
    for name, why in no:
        print(f"  refused  {name}: {why}")
    quorum = len(ok) // 2 + 1
    print(f"\npanel N={len(ok)}, majority quorum={quorum}")

    # The generator table used to be prose no command read. It is gated now, because
    # `juggernaut-z-image` (NC) and `z-image-turbo` (Apache-2.0) differ by a prefix and the
    # difference was carried by a comment.
    print("\ngenerators -- base and CONTROL, which is the term a base-licence check misses:")
    conditionable = []
    for name, base, control, token in survey():
        surveyed = token != NOT_APPLICABLE
        clear = base is Status.OK and control is Status.OK and surveyed
        if clear:
            conditionable.append(name)
        shown = control.name if surveyed else "NOT SURVEYED"
        print(f"  {'  ' if clear else 'X '}{name:<24} base={base.name:<10} control={shown}")
    print(f"\ndepth-conditionable and licence-clean ({len(conditionable)}): "
          f"{', '.join(conditionable) or 'NONE'}")
    print("Corpus terms are UNVERIFIED for every generator above, so none is ADMITTED. That "
          "is condition 1 of the synthetic-data rule\nunanswered by the whole field at once, "
          "not an oversight in one row.")

    if len(ok) < 3:
        raise SystemExit(
            f"FAIL: {len(ok)} licence-clear backends. A panel needs at least 3 to tolerate "
            f"one outage. This is the intended initial state -- verify the roster above."
        )
