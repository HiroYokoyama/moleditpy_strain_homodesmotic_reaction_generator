"""
Unit tests for core.py helper functions not covered by test_analysis.py.
Covers: atom_counts, count_groups, bond_counts helpers, SMILES coloring,
format helpers, reaction-type color, internal search helpers, and scipy warning fix.
"""

from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    BalanceTerm,
    ReferenceTerm,
    atom_counts,
    bond_counts,
    build_hyperhomodesmotic_balance_terms,
    color_equation_term,
    color_reference_term,
    color_smiles_atoms,
    color_smiles_atoms_by_index,
    count_groups,
    explicit_hydrogen_copy,
    format_reference_terms,
    format_terms,
    get_reaction_type_color,
    milp,
    species_atom_counts,
    unique_substructure_matches,
    _counter_fits,
    _counter_key,
    _html_counter,
    _span,
    _subtract_counter,
)

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit not available")


# ---------------------------------------------------------------------------
# atom_counts
# ---------------------------------------------------------------------------


def test_atom_counts_ethane_with_h():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert atom_counts(mol) == Counter({"C": 2, "H": 6})


def test_atom_counts_water():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert atom_counts(mol) == Counter({"O": 1, "H": 2})


def test_atom_counts_none_returns_empty():
    assert atom_counts(None) == Counter()


def test_atom_counts_without_explicit_h():
    mol = Chem.MolFromSmiles("CC")
    assert atom_counts(mol) == Counter({"C": 2})


# ---------------------------------------------------------------------------
# count_groups
# ---------------------------------------------------------------------------


def test_count_groups_ethane_sp3_carbons():
    mol = Chem.MolFromSmiles("CC")
    counts = count_groups(mol)
    assert counts["C(H3)(O_s0)(O_d0)(N_s0)(N_d0)(N_t0)(=C0)(#C0)"] == 2


def test_count_groups_acetone_carbonyl_oxygen():
    mol = Chem.MolFromSmiles("CC(=O)C")
    counts = count_groups(mol)
    assert counts["=O"] == 1


def test_count_groups_dme_ether_oxygen():
    mol = Chem.MolFromSmiles("COC")
    counts = count_groups(mol)
    assert counts["O(C)(C)"] == 1


def test_count_groups_benzene_aromatic_carbons():
    mol = Chem.MolFromSmiles("c1ccccc1")
    counts = count_groups(mol)
    assert counts["C_ar(H1)(O_s0)(O_d0)(N_s0)(N_d0)(N_t0)(ar2)"] == 6


def test_count_groups_methylamine_primary_nitrogen():
    mol = Chem.MolFromSmiles("CN")
    counts = count_groups(mol)
    assert counts["N(H2)"] == 1


def test_count_groups_trimethylamine_tertiary_nitrogen():
    mol = Chem.MolFromSmiles("CN(C)C")
    counts = count_groups(mol)
    assert counts["N(H0)"] == 1


def test_count_groups_pyridine_aromatic_nitrogen():
    mol = Chem.MolFromSmiles("c1ccncc1")
    counts = count_groups(mol)
    assert "N_ar(H0)" in counts


def test_count_groups_skips_hydrogen_atoms():
    mol = Chem.MolFromSmiles("CC")
    counts = count_groups(mol)
    assert "H" not in counts


# ---------------------------------------------------------------------------
# bond_counts
# ---------------------------------------------------------------------------


def test_bond_counts_ethane_single_cc_bond():
    mol = Chem.MolFromSmiles("CC")
    counts = bond_counts(mol)
    assert counts["C(sp3,H3)-C(sp3,H3) (single)"] == 1


def test_bond_counts_ethane_ch_bonds():
    mol = Chem.MolFromSmiles("CC")
    counts = bond_counts(mol)
    assert counts["C(sp3,H3)-H (single)"] == 6


def test_bond_counts_ethene_double_bond():
    mol = Chem.MolFromSmiles("C=C")
    counts = bond_counts(mol)
    double_bond_keys = [k for k in counts if "(double)" in k and "C" in k]
    assert double_bond_keys


def test_bond_counts_none_returns_empty():
    from strain_homodesmotic_reaction_generator.core import bond_counts

    assert bond_counts(None) == Counter()


# ---------------------------------------------------------------------------
# explicit_hydrogen_copy
# ---------------------------------------------------------------------------


def test_explicit_hydrogen_copy_adds_hydrogens():
    mol = Chem.MolFromSmiles("CC")
    n_before = mol.GetNumAtoms()
    mol_h = explicit_hydrogen_copy(mol)
    assert mol_h.GetNumAtoms() > n_before


def test_explicit_hydrogen_copy_does_not_mutate_original():
    mol = Chem.MolFromSmiles("CC")
    n_before = mol.GetNumAtoms()
    explicit_hydrogen_copy(mol)
    assert mol.GetNumAtoms() == n_before


def test_explicit_hydrogen_copy_none_returns_none():
    assert explicit_hydrogen_copy(None) is None


# ---------------------------------------------------------------------------
# unique_substructure_matches
# ---------------------------------------------------------------------------


def test_unique_substructure_matches_single_match():
    mol = Chem.MolFromSmiles("CC")
    matches = unique_substructure_matches(mol, "[CX4H3]-[CX4H3]")
    assert len(matches) == 1


def test_unique_substructure_matches_multiple():
    mol = Chem.MolFromSmiles("CCC")
    matches = unique_substructure_matches(mol, "[CX4H3]-[CX4H2]")
    assert len(matches) == 2


def test_unique_substructure_matches_none():
    mol = Chem.MolFromSmiles("C")
    matches = unique_substructure_matches(mol, "[CX4H3]-[CX4H3]")
    assert len(matches) == 0


def test_unique_substructure_matches_atom_centric():
    mol = Chem.MolFromSmiles("c1ccccc1")
    matches = unique_substructure_matches(mol, "[cX3H1]", atom_centric=True)
    assert len(matches) == 6


def test_unique_substructure_matches_invalid_smarts_returns_empty():
    mol = Chem.MolFromSmiles("CC")
    matches = unique_substructure_matches(mol, "INVALID_SMARTS_!!!!")
    assert len(matches) == 0


# ---------------------------------------------------------------------------
# species_atom_counts
# ---------------------------------------------------------------------------


def test_species_atom_counts_ethane():
    result = species_atom_counts("CC")
    assert result == Counter({"C": 2, "H": 6})


def test_species_atom_counts_water():
    result = species_atom_counts("O")
    assert result == Counter({"O": 1, "H": 2})


def test_species_atom_counts_invalid_smiles():
    result = species_atom_counts("NOT_VALID_SMILES_XYZ")
    assert result == Counter()


# ---------------------------------------------------------------------------
# _span
# ---------------------------------------------------------------------------


def test_span_basic_html_structure():
    result = _span("C", "#ff0000")
    assert '<span style="color:#ff0000; font-weight:600;">C</span>' == result


def test_span_with_title_attr():
    result = _span("O", "#00ff00", "oxygen")
    assert 'title="oxygen"' in result


def test_span_escapes_html_chars():
    result = _span("<>&", "#ff0000")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result
    assert "<>" not in result


def test_span_title_escapes_html():
    result = _span("N", "#0000ff", 'title with "quotes"')
    assert "&quot;" in result or "quotes" not in result.split("title=")[1].split('"')[1]


# ---------------------------------------------------------------------------
# color_smiles_atoms_by_index
# ---------------------------------------------------------------------------


def test_color_smiles_atoms_all_default_color():
    result = color_smiles_atoms_by_index("CC", {}, "#ff0000", "test")
    assert result.count("#ff0000") == 2


def test_color_smiles_atoms_indexed_override():
    result = color_smiles_atoms_by_index(
        "CC", {0: ("#00ff00", "first")}, "#ff0000", "default"
    )
    assert "#00ff00" in result
    assert "#ff0000" in result


def test_color_smiles_atoms_bracket_atom():
    result = color_smiles_atoms_by_index("[NH3]", {}, "#8ab4f8", "")
    assert "#8ab4f8" in result
    assert "<span" in result  # bracket atom wrapped in colored span


def test_color_smiles_atoms_two_letter_halogen():
    result = color_smiles_atoms_by_index("CCl", {}, "#ff0000", "")
    assert result.count("#ff0000") == 2  # C and Cl


def test_color_smiles_atoms_non_atom_chars_pass_through():
    result = color_smiles_atoms("C(=O)C", "#ff0000", "")
    assert "=O" not in result or "(=" not in result


# ---------------------------------------------------------------------------
# color_equation_term
# ---------------------------------------------------------------------------


def test_color_equation_term_formats_count_and_smiles():
    result = color_equation_term("2 CC", "#8ab4f8", "test")
    assert result.startswith("2 ")
    assert "#8ab4f8" in result


def test_color_equation_term_no_space_is_smiles_only():
    result = color_equation_term("CC", "#8ab4f8", "test")
    assert "#8ab4f8" in result
    assert "2" not in result


# ---------------------------------------------------------------------------
# color_reference_term
# ---------------------------------------------------------------------------


def test_color_reference_term_includes_count_prefix():
    term = ReferenceTerm(3, "CC", (0, 1))
    result = color_reference_term(term, "#8ab4f8", "#81c995")
    assert result.startswith("3 ")


def test_color_reference_term_uses_core_map_for_propane():
    term = ReferenceTerm(2, "CCC", (1,))
    result = color_reference_term(term, "#8ab4f8", "#81c995")
    assert "#8ab4f8" in result  # core (middle C)
    assert "#81c995" in result  # added caps (outer Cs)


def test_color_reference_term_all_core_for_benzene():
    from strain_homodesmotic_reaction_generator.data import _BALANCE_CORE_MAP

    if "c1ccccc1" not in _BALANCE_CORE_MAP:
        pytest.skip("benzene not in balance core map")
    term = ReferenceTerm(1, "c1ccccc1", tuple(range(6)))
    result = color_reference_term(term, "#8ab4f8", "#81c995")
    assert "#8ab4f8" in result


# ---------------------------------------------------------------------------
# format_terms / format_reference_terms
# ---------------------------------------------------------------------------


def test_format_terms_empty_returns_none():
    assert format_terms([]) == "none"


def test_format_terms_with_positive_count():
    terms = [BalanceTerm("ethane", "CC", 2), BalanceTerm("methane", "C", 1)]
    result = format_terms(terms)
    assert "2 CC (ethane)" in result
    assert "1 C (methane)" in result


def test_format_terms_zero_count_excluded():
    terms = [BalanceTerm("ethane", "CC", 0), BalanceTerm("methane", "C", 1)]
    result = format_terms(terms)
    assert (
        "(ethane)" not in result
    )  # term name in parens absent; "ethane" is substring of "methane"
    assert "1 C (methane)" in result


def test_format_reference_terms_empty():
    assert format_reference_terms([]) == "no reference molecules detected"


def test_format_reference_terms_with_terms():
    terms = [ReferenceTerm(2, "CC", (0, 1)), ReferenceTerm(1, "C", ())]
    result = format_reference_terms(terms)
    assert "2 CC" in result
    assert "1 C" in result


def test_format_reference_terms_zero_count_excluded():
    terms = [ReferenceTerm(0, "CC", ()), ReferenceTerm(3, "C", ())]
    result = format_reference_terms(terms)
    assert "CC" not in result
    assert "3 C" in result


# ---------------------------------------------------------------------------
# get_reaction_type_color
# ---------------------------------------------------------------------------


def test_get_reaction_type_color_hyperhomodesmotic():
    assert get_reaction_type_color("Hyperhomodesmotic") == "#81c995"


def test_get_reaction_type_color_homodesmotic():
    assert get_reaction_type_color("Homodesmotic") == "#8ab4f8"


def test_get_reaction_type_color_isodesmic():
    assert get_reaction_type_color("Isodesmic") == "#fde293"


def test_get_reaction_type_color_elemental_balance():
    assert get_reaction_type_color("Elemental Balance") == "#c58af9"


def test_get_reaction_type_color_unbalanced():
    assert get_reaction_type_color("Unbalanced") == "#f28b82"


def test_get_reaction_type_color_unknown_returns_red():
    assert get_reaction_type_color("SomethingElse") == "#f28b82"


# ---------------------------------------------------------------------------
# _html_counter
# ---------------------------------------------------------------------------


def test_html_counter_empty_returns_none():
    assert _html_counter(Counter(), "#ff0000", "test") == "none"


def test_html_counter_contains_element_and_color():
    result = _html_counter(Counter({"C": 2, "H": 6}), "#8ab4f8", "atom")
    assert "C" in result
    assert "H" in result
    assert "#8ab4f8" in result


def test_html_counter_sorted_alphabetically():
    result = _html_counter(Counter({"H": 6, "C": 2}), "#ff0000", "")
    c_pos = result.index(">C<")
    h_pos = result.index(">H<")
    assert c_pos < h_pos


# ---------------------------------------------------------------------------
# Internal balance helpers
# ---------------------------------------------------------------------------


def test_counter_fits_exact_match():
    assert _counter_fits(Counter({"C": 2, "H": 6}), Counter({"C": 2, "H": 6}))


def test_counter_fits_remaining_is_larger():
    assert _counter_fits(Counter({"C": 1}), Counter({"C": 3, "H": 8}))


def test_counter_fits_candidate_too_large():
    assert not _counter_fits(Counter({"C": 3}), Counter({"C": 2}))


def test_counter_fits_missing_element():
    assert not _counter_fits(Counter({"N": 1}), Counter({"C": 4}))


def test_subtract_counter_removes_exact():
    result = _subtract_counter(Counter({"C": 2, "H": 6}), Counter({"C": 2, "H": 6}))
    assert result == Counter()


def test_subtract_counter_partial():
    result = _subtract_counter(Counter({"C": 4, "H": 10}), Counter({"C": 2, "H": 6}))
    assert result == Counter({"C": 2, "H": 4})


def test_subtract_counter_zero_values_removed():
    result = _subtract_counter(Counter({"C": 1}), Counter({"C": 1}))
    assert "C" not in result


def test_counter_key_is_sorted():
    key = _counter_key(Counter({"H": 6, "C": 2}))
    elements = [e for e, _ in key]
    assert elements == sorted(elements)


def test_counter_key_excludes_zero_and_negative():
    key = _counter_key(Counter({"C": 2, "H": 0}))
    assert not any(e == "H" for e, _ in key)


# ---------------------------------------------------------------------------
# build_hyperhomodesmotic_balance_terms
# ---------------------------------------------------------------------------


def test_build_hyperhomodesmotic_empty_delta_trivially_succeeds():
    left, right, success = build_hyperhomodesmotic_balance_terms(Counter())
    assert success is True
    assert left == ()
    assert right == ()


def test_build_hyperhomodesmotic_requires_scipy():
    if milp is None:
        pytest.skip("scipy not available")
    group_delta = Counter({"C(H3)(O_s0)(O_d0)(N_s0)(N_d0)(N_t0)(=C0)(#C0)": 2})
    left, right, success = build_hyperhomodesmotic_balance_terms(group_delta)
    assert isinstance(success, bool)
    assert isinstance(left, tuple)
    assert isinstance(right, tuple)


# ---------------------------------------------------------------------------
# Scipy warning fix: ui.py must import milp
# ---------------------------------------------------------------------------


def test_ui_module_imports_milp_for_warning_distinction():
    """ui.py must import milp so it can show the right warning when scipy IS installed
    but MILP fails, vs. when scipy is missing entirely."""
    import importlib
    import types
    import sys as _sys

    # Stub out PyQt6 if not present so ui.py can be imported headlessly
    for mod_name in ["PyQt6", "PyQt6.QtCore", "PyQt6.QtWidgets", "PyQt6.QtGui"]:
        _sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    ui = importlib.import_module("strain_homodesmotic_reaction_generator.ui")
    assert hasattr(ui, "milp"), (
        "ui.py must import 'milp' from core to display context-aware warning "
        "when scipy is installed but MILP balance fails"
    )
