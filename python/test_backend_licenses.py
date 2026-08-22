# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Negative controls for the licence gate, written from the defects they now catch.

WHAT WAS WRONG, TWICE.

First, `Backend` carried two licences and a generator has three. Control weights are a separate
release under separate terms, so a base-licence check admits FLUX.1 [schnell] (Apache-2.0) and
HiDream-I1 (MIT), neither of which has a readable way to be conditioned. Every corpus use here
renders an ANNY pose and requires the generated image to keep that geometry, so the control is
the term that decides and it was the term not being read.

Second, `juggernaut-z-image` is CC-BY-NC-4.0 and `z-image-turbo` is Apache-2.0. RunDiffusion's
finetune and Tongyi-MAI's base differ by a prefix and their verdicts invert. A search for
`z-image` returns one of each, and the distinction was carried by a comment nobody executed --
`GENERATORS` was a table no command read, which is the "recall is not a licence check" failure
the module docstring exists to fix, reappearing one list further down the same file.

WHY THEY SURVIVED. Both tables looked right. `gate()` was correct on every row it was ever run
against, because it was only ever run against `ROSTER`, where no entry conditions anything.

THE CONTROLS. Each breaks one thing and asserts the gate says so. Control 2 runs the RETRACTED
two-axis gate and asserts it MISSES the defect, because a fix with no failing negative control
has certified the defect rather than removed it.

Run:  python3 test_backend_licenses.py
"""

import sys

from backend_licenses import (GENERATORS, NOT_APPLICABLE, Backend, Role, Status, classify,
                              gate, survey)

BY_NAME = {b.name: b for b in GENERATORS}


def _refusal(roster):
    """(name -> reason) for everything the gate turns away."""
    _, refused = gate(roster)
    return dict(refused)


def control_1_the_collision():
    """Two models one prefix apart must land on opposite verdicts."""
    nc = classify(BY_NAME["juggernaut-z-image"].weights_license)
    ok = classify(BY_NAME["z-image-turbo"].weights_license)
    assert nc is Status.DENIED, f"RunDiffusion's NC finetune classified {nc.name}"
    assert ok is Status.OK, f"Tongyi-MAI's Apache-2.0 base classified {ok.name}"
    assert nc is not ok
    return f"juggernaut-z-image={nc.name}, z-image-turbo={ok.name} -- inverted, as required"


def control_2_retracted_two_axis_gate():
    """The gate before the control axis. It must MISS what the three-axis gate catches.

    It does not admit `flux-schnell` -- the corpus is UNVERIFIED, so it refuses it. It refuses
    it for the WRONG REASON, which is the whole defect: a reader who later verifies the corpus
    would admit a model that cannot be conditioned. So the assertion is about the reason.
    """
    def two_axis(b):
        w, c = classify(b.weights_license), classify(b.corpus_license)
        return max((w, c), key=lambda s: [Status.OK, Status.FLAGGED,
                                          Status.UNVERIFIED, Status.DENIED].index(s))

    for name in ("flux-schnell", "hidream-i1"):
        b = BY_NAME[name]
        assert classify(b.weights_license) is Status.OK, f"{name} base is not clean"
        assert two_axis(b) is not Status.DENIED, (
            f"RETRACTED gate already denied {name}; this control certifies nothing")
        reason = _refusal([b])[name]
        assert "control" in reason, f"three-axis refusal of {name} does not name the control: {reason}"
    return "flux-schnell and hidream-i1: clean base, retracted gate misses it, control catches it"


def control_3_not_applicable_is_not_a_pass():
    """A row nobody surveyed must not print as a cleared control, or count as conditionable."""
    unsurveyed = [n for n, _, _, tok in survey() if tok == NOT_APPLICABLE]
    assert unsurveyed, "no unsurveyed rows left -- this control no longer tests anything"
    # Force a clean base onto an unsurveyed row. It must still not be conditionable.
    victim = Backend(BY_NAME[unsurveyed[0]].name, "Apache-2.0", "Apache-2.0", checked=True)
    assert classify(victim.control_license) is Status.OK, "NOT-APPLICABLE should not fail a row"
    name, base, control, token = survey([victim])[0]
    assert base is Status.OK and token == NOT_APPLICABLE, "setup wrong"
    assert token != Status.OK.name, "an unsurveyed control renders as a verdict"
    return f"{len(unsurveyed)} unsurveyed rows; a clean base does not make one conditionable"


def control_4_missing_checkpoint_is_not_a_licence_pass():
    """SANA is Apache-2.0 on base and control architecture, and must still be refused."""
    b = BY_NAME["sana"]
    assert classify(b.weights_license) is Status.OK, "SANA base is not Apache-2.0 here"
    reason = _refusal([b])["sana"]
    assert "NO-DEPTH-CHECKPOINT" in reason, f"refusal does not name the missing control: {reason}"
    return f"sana refused on the control despite a clean base: {reason}"


def control_5_per_control_not_per_owner():
    """Kolors ships -Depth as Apache-2.0 and -Pose with no tag. An owner verdict would be wrong."""
    assert classify("NO-LICENCE-TAG") is Status.DENIED, "an untagged control classified as clear"
    assert classify(BY_NAME["kolors"].control_license) is Status.OK
    return "NO-LICENCE-TAG=DENIED while the same owner's depth control=OK"


def control_6_same_base_opposite_verdict():
    """qwen-image and qwen-image-edit share a base licence and must not share a verdict."""
    a, b = BY_NAME["qwen-image"], BY_NAME["qwen-image-edit"]
    assert a.weights_license == b.weights_license, "the premise of this control is gone"
    ra, rb = _refusal([a]).get("qwen-image", ""), _refusal([b])["qwen-image-edit"]
    assert "control" not in ra, f"qwen-image refused on its control: {ra}"
    assert "control" in rb, f"qwen-image-edit not refused on its control: {rb}"
    return f"both {a.weights_license}; edit refused on the control, base is not"


def control_7_role_reaches_the_gate():
    """OpenRAIL-M turns on role. Two rows differing ONLY in role must not share a verdict.

    This is the retracted defect stated as an assertion. `classify` always accepted a role and
    `gate` never passed one, so both rows below came out DENIED -- the right verdict for the
    generator and the wrong reason for the passthrough. A rule reachable only from a caller
    that does not exist is not enforced.
    """
    lic = "OpenRAIL-M"
    thru = Backend("control-passthrough", lic, "Apache-2.0", checked=True, role=Role.PASSTHROUGH)
    gen = Backend("control-generator", lic, "Apache-2.0", checked=True, role=Role.GENERATOR)
    assert classify(lic, Role.PASSTHROUGH) is Status.FLAGGED
    assert classify(lic, Role.GENERATOR) is Status.DENIED
    r_thru, r_gen = _refusal([thru])["control-passthrough"], _refusal([gen])["control-generator"]
    assert r_thru != r_gen, f"role does not reach gate(): both refused as {r_gen!r}"
    assert "FLAGGED" in r_thru and "DENIED" in r_gen, f"{r_thru!r} / {r_gen!r}"
    # And a row that states no role gets the strict answer, not a convenient default.
    silent = Backend("control-silent", lic, "Apache-2.0", checked=True)
    assert "DENIED" in _refusal([silent])["control-silent"], "an unstated role read as permissive"
    return f"passthrough={r_thru}; generator={r_gen}; unstated=DENIED"


def control_8_every_generator_states_its_role():
    """An OpenRAIL generator row that forgets `role` is denied by accident, not by the rule."""
    silent = [b.name for b in GENERATORS if b.role is not Role.GENERATOR]
    assert not silent, f"generator rows with no role declared: {silent}"
    return f"all {len(GENERATORS)} generator rows declare role=GENERATOR"


CONTROLS = [
    ("1 the z-image collision", control_1_the_collision),
    ("2 retracted two-axis gate misses it", control_2_retracted_two_axis_gate),
    ("3 NOT-APPLICABLE is not a pass", control_3_not_applicable_is_not_a_pass),
    ("4 missing checkpoint is not a licence pass", control_4_missing_checkpoint_is_not_a_licence_pass),
    ("5 per-control, not per-owner", control_5_per_control_not_per_owner),
    ("6 same base, opposite verdict", control_6_same_base_opposite_verdict),
    ("7 role reaches the gate", control_7_role_reaches_the_gate),
    ("8 every generator states its role", control_8_every_generator_states_its_role),
]


def main():
    fails = []
    for name, fn in CONTROLS:
        try:
            detail = fn()
            print(f"  ok   {name}\n         {detail}")
        except AssertionError as e:
            fails.append(name)
            print(f"  FAIL {name}\n         {e}")
    print()
    if fails:
        print(f"{len(fails)} of {len(CONTROLS)} controls failed.")
        return 1
    print(f"{len(CONTROLS)} controls, each failing for its own reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
