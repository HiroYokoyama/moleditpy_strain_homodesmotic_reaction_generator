"""Charge constraints, required SciPy and solver integrity."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from strain_homodesmotic_reaction_generator import core, ui
from strain_homodesmotic_reaction_generator.data import BalanceSpecies, UserSpecies


def test_charge_mismatch_is_prevented_by_the_solver():
    users = [
        UserSpecies("[Na+]", required=True, side="left"),
        UserSpecies("[Na]", required=True, side="right"),
    ]
    result = core.analyze_molecule(core.Chem.MolFromSmiles("C1CC1"), user_species=users)
    assert result.reaction_type != "Unbalanced"
    assert result.unmet_required
    left = [(1, result.target_smiles)] + [
        (t.count, t.smiles) for t in result.left_balance_terms
    ]
    right = [(t.count, t.smiles) for t in result.right_balance_terms]
    right += [(m.count, m.reference_smiles) for m in result.matches]
    assert core._side_net_charge(left) == core._side_net_charge(right)


def test_atomic_shortfall_is_solved_even_without_a_group_shortfall():
    left, right, success, _ = core.build_hyperhomodesmotic_balance_terms(
        Counter(),
        Counter({"H": 2}),
        extra_species=[BalanceSpecies("hydrogen", "[H][H]", "")],
    )
    assert success
    assert left or right
    balance = Counter()
    for sign, terms in [(1, left), (-1, right)]:
        for term in terms:
            for element, count in core.species_atom_counts(term.smiles).items():
                balance[element] += sign * term.count * count
    assert balance == Counter({"H": 2})


def test_infeasible_solver_does_not_enter_an_elemental_fallback(monkeypatch):
    monkeypatch.setattr(core, "_solve_with_required", lambda *a, **kw: None)
    monkeypatch.setattr(
        core,
        "build_balance_terms",
        lambda *a, **kw: pytest.fail("fallback must not run"),
    )
    result = core.analyze_molecule(
        core.Chem.MolFromSmiles("C1CC1"),
        user_species=[UserSpecies("C", required=True, side="right")],
    )
    assert "C" in result.unmet_required
    assert result.solver_failed
    assert result.reaction_type == "Unbalanced"


@pytest.mark.parametrize(
    "left,right",
    [
        ([(1, "bad-smiles")], [(1, "bad-smiles")]),
        ([(1.5, "C")], [(1.5, "C")]),
        ([(-1, "C")], [(-1, "C")]),
    ],
)
def test_invalid_species_and_coefficients_cannot_receive_a_verdict(left, right):
    with pytest.raises(ValueError):
        core.classify_sides(left, right)


def test_empty_equations_are_unbalanced():
    assert core.classify_sides([], []) == "Unbalanced"


@pytest.mark.parametrize(
    "values", [[np.nan, 0], [np.inf, 0], [0.6, 0], [0, 0], [-1, -2]]
)
def test_solver_success_is_rechecked_against_integer_constraints(monkeypatch, values):
    def solve(**kwargs):
        assert kwargs["options"]["time_limit"] > 0
        return SimpleNamespace(success=True, x=np.asarray(values))

    monkeypatch.setattr(core, "milp", solve)
    constraint = core.LinearConstraint(np.array([[1.0, -1.0]]), 1, 1)
    assert (
        core._solve_with_required(np.ones(2), [constraint], np.ones(2), 1, []) is None
    )


def test_worker_uses_a_snapshot_of_the_host_molecule():
    original = core.Chem.RWMol(core.Chem.MolFromSmiles("CC"))
    worker = ui.AnalysisWorker(original)
    original.RemoveAtom(1)
    assert worker.mol.GetNumAtoms() == 2
    assert original.GetNumAtoms() == 1


def test_dative_bonds_are_not_treated_as_reaction_arrows():
    check = core.check_equation("N->[Cu+] -> N->[Cu+]")
    assert not check.errors
    assert check.reaction_type == "Hyperhomodesmotic"
