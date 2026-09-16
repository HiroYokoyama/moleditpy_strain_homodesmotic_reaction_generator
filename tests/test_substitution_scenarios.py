"""
What substituting a reference molecule is actually for, end to end.

Each test is a workflow a user would run, and the chemistry is re-counted
here from the equation terms rather than read back off the result. The point
is to catch a balance or a classification that the plugin believes but that
does not hold: ``core`` computing its own verdict and a test asserting that
verdict would agree with each other no matter how wrong both were.
"""

from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator import core  # noqa: E402
from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
)
from strain_homodesmotic_reaction_generator.data import UserSpecies  # noqa: E402

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")

CYCLOPROPANE = "C1CC1"
CYCLOBUTANE = "C1CCC1"
CYCLOPENTANE = "C1CCCC1"
OXETANE = "C1COC1"

#: The environment every unsubstituted ring carbon pair matches.
RING_ENVIRONMENT = "secondary carbon - secondary carbon"
RING_DEFAULT_REFERENCE = "CCCC"


# ---------------------------------------------------------------------------
# An independent recount, deliberately not reusing core's helpers
# ---------------------------------------------------------------------------


def _explicit(smiles: str):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _carbon_label(atom) -> str:
    hydrogens = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    return f"{atom.GetSymbol()}H{hydrogens}"


def _atoms(smiles: str) -> Counter:
    return Counter(atom.GetSymbol() for atom in _explicit(smiles).GetAtoms())


def _groups(smiles: str) -> Counter:
    """Heavy atoms labelled by element and hydrogen count."""
    return Counter(
        _carbon_label(atom)
        for atom in _explicit(smiles).GetAtoms()
        if atom.GetAtomicNum() != 1
    )


def _heavy_bonds(smiles: str) -> Counter:
    """Heavy-atom bonds labelled by the two groups they join."""
    counts: Counter = Counter()
    for bond in _explicit(smiles).GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
            continue
        pair = "-".join(sorted((_carbon_label(begin), _carbon_label(end))))
        counts[f"{pair} {bond.GetBondType()}"] += 1
    return counts


def _sides(result) -> tuple[list, list]:
    """(left, right) as (count, smiles) pairs, target included on the left."""
    left = [(1, result.target_smiles)]
    left += [
        (term.count, term.smiles)
        for term in result.left_balance_terms
        if term.count > 0
    ]
    right = [
        (match.count, match.reference_smiles)
        for match in result.matches
        if match.count > 0
    ]
    right += [
        (term.count, term.smiles)
        for term in result.right_balance_terms
        if term.count > 0
    ]
    return left, right


def _totals(side, counter) -> Counter:
    total: Counter = Counter()
    for count, smiles in side:
        for key, value in counter(smiles).items():
            total[key] += count * value
    return total


def _check(result) -> dict:
    """Recount the equation and say what it conserves."""
    left, right = _sides(result)
    return {
        "atoms": _totals(left, _atoms) == _totals(right, _atoms),
        "groups": _totals(left, _groups) == _totals(right, _groups),
        "bonds": _totals(left, _heavy_bonds) == _totals(right, _heavy_bonds),
        "left": left,
        "right": right,
    }


def _molecules(side) -> Counter:
    return Counter({smiles: count for count, smiles in side})


# ---------------------------------------------------------------------------
# The recount agrees with the plugin's own verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,overrides",
    [
        (CYCLOPROPANE, {}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CCC"}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CCCCC"}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CC"}),
        (CYCLOBUTANE, {}),
        (CYCLOBUTANE, {RING_ENVIRONMENT: "CCCCC"}),
        (CYCLOPENTANE, {}),
        (CYCLOPENTANE, {RING_ENVIRONMENT: "CCC"}),
    ],
)
def test_a_substituted_equation_still_balances(target, overrides):
    """Whatever reference is chosen, the atoms have to come out even."""
    result = analyze_molecule(Chem.MolFromSmiles(target), overrides)
    assert _check(result)["atoms"], result.equation_text.splitlines()[3]


@pytest.mark.parametrize(
    "target,overrides",
    [
        (CYCLOPROPANE, {}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CCC"}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CCCCC"}),
        (CYCLOPROPANE, {RING_ENVIRONMENT: "CC"}),
        (CYCLOBUTANE, {}),
        (CYCLOPENTANE, {}),
        (CYCLOPENTANE, {RING_ENVIRONMENT: "CCC"}),
    ],
)
def test_the_reported_type_matches_an_independent_recount(target, overrides):
    """Hyperhomodesmotic must mean bonds conserved, Homodesmotic only groups."""
    result = analyze_molecule(Chem.MolFromSmiles(target), overrides)
    checked = _check(result)
    if result.reaction_type == "Hyperhomodesmotic":
        assert checked["bonds"] and checked["groups"] and checked["atoms"]
    elif result.reaction_type == "Homodesmotic":
        assert checked["groups"] and checked["atoms"]
        assert not checked["bonds"]
    else:
        assert not checked["groups"]


# ---------------------------------------------------------------------------
# Reproducing a scheme from the literature
# ---------------------------------------------------------------------------


def test_the_default_cyclopropane_scheme_is_the_textbook_one():
    result = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    left, right = _sides(result)
    assert _molecules(left) == Counter({CYCLOPROPANE: 1, "CCC": 3})
    assert _molecules(right) == Counter({"CCCC": 3})
    assert result.reaction_type == "Hyperhomodesmotic"


def test_substituting_propane_reproduces_the_published_ring_opening():
    """cyclopropane + 3 ethane -> 3 propane, the form most papers quote.

    The default answers with butane references; a reader reproducing a paper
    needs the propane form, and this is the substitution that gets it.
    """
    result = analyze_molecule(
        Chem.MolFromSmiles(CYCLOPROPANE), {RING_ENVIRONMENT: "CCC"}
    )
    left, right = _sides(result)
    assert _molecules(left) == Counter({CYCLOPROPANE: 1, "CC": 3})
    assert _molecules(right) == Counter({"CCC": 3})
    assert _check(result)["groups"]


def test_the_published_form_is_weaker_than_the_default_and_says_so():
    """Trading butane for propane loses bond conservation; that is the cost."""
    default = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    published = analyze_molecule(
        Chem.MolFromSmiles(CYCLOPROPANE), {RING_ENVIRONMENT: "CCC"}
    )
    assert default.reaction_type == "Hyperhomodesmotic"
    assert published.reaction_type == "Homodesmotic"
    assert _check(default)["bonds"]
    assert not _check(published)["bonds"]


# ---------------------------------------------------------------------------
# Working from the reference energies you already have
# ---------------------------------------------------------------------------


def test_a_longer_reference_still_reaches_hyperhomodesmotic():
    """Someone holding pentane energies but no butane is not stuck."""
    result = analyze_molecule(
        Chem.MolFromSmiles(CYCLOPROPANE), {RING_ENVIRONMENT: "CCCCC"}
    )
    _left, right = _sides(result)
    assert _molecules(right) == Counter({"CCCCC": 3})
    assert result.reaction_type == "Hyperhomodesmotic"
    assert _check(result)["bonds"]


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_forcing_a_species_you_have_energies_for_puts_it_in_the_equation():
    result = analyze_molecule(
        Chem.MolFromSmiles(CYCLOPENTANE),
        user_species=[UserSpecies("CCCCCC", "hexane", required=True)],
    )
    left, _right = _sides(result)
    assert dict(_molecules(left)).get("CCCCCC") == 1
    assert result.unmet_required == ()
    assert _check(result)["atoms"]


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_a_species_that_cannot_help_is_named_and_the_draft_survives():
    """Isobutane brings a methine carbon nothing on the other side has."""
    result = analyze_molecule(
        Chem.MolFromSmiles(CYCLOBUTANE),
        user_species=[UserSpecies("CC(C)C", "isobutane", required=True)],
    )
    assert result.unmet_required == ("CC(C)C",)
    assert result.reaction_type == "Hyperhomodesmotic"
    assert _check(result)["bonds"]


# ---------------------------------------------------------------------------
# Comparing candidate schemes, which is what the substitution is for
# ---------------------------------------------------------------------------


def test_candidate_schemes_can_be_ranked_by_what_they_conserve():
    """The workflow the feature exists for: try several, keep the best."""
    candidates = {
        "butane (default)": {},
        "pentane": {RING_ENVIRONMENT: "CCCCC"},
        "propane": {RING_ENVIRONMENT: "CCC"},
        "ethane": {RING_ENVIRONMENT: "CC"},
    }
    verdicts = {
        name: analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE), overrides)
        for name, overrides in candidates.items()
    }
    for name, result in verdicts.items():
        assert _check(result)["atoms"], name
        assert _check(result)["groups"], name

    strong = {
        name
        for name, result in verdicts.items()
        if result.reaction_type == "Hyperhomodesmotic"
    }
    assert strong == {"butane (default)", "pentane"}


def test_a_substitution_is_recorded_so_two_runs_can_be_told_apart():
    default = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    swapped = analyze_molecule(
        Chem.MolFromSmiles(CYCLOPROPANE), {RING_ENVIRONMENT: "CCCCC"}
    )
    assert default.reference_overrides == ()
    assert swapped.reference_overrides == ((RING_ENVIRONMENT, "CCCCC"),)
    assert "CCCCC" in swapped.equation_text
    assert default.equation_text != swapped.equation_text


# ---------------------------------------------------------------------------
# Heteroatoms
# ---------------------------------------------------------------------------


def test_an_ether_reference_can_be_substituted_without_breaking_the_ring_terms():
    """Oxetane carries two environments; overriding one must not disturb the other."""
    environment = "secondary carbon - ether oxygen - secondary carbon"
    result = analyze_molecule(Chem.MolFromSmiles(OXETANE), {environment: "CCOCCC"})
    checked = _check(result)
    assert checked["atoms"]
    assert checked["groups"]
    assert dict(_molecules(checked["right"])).get("CCCC") == 2


def test_the_oxygen_is_conserved_through_a_substitution():
    environment = "secondary carbon - ether oxygen - secondary carbon"
    for overrides in ({}, {environment: "CCOCCC"}):
        result = analyze_molecule(Chem.MolFromSmiles(OXETANE), overrides)
        left, right = _sides(result)
        assert _totals(left, _atoms)["O"] == _totals(right, _atoms)["O"] == 1
