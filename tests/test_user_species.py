"""
User-defined reference molecules and balance species.

Covers the core plumbing (overrides, an extended species library, the
``required`` constraint) and the dialog wiring that feeds it, including the
Add Species dialog and the re-orderable results table.
"""

from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator import ui  # noqa: E402
from strain_homodesmotic_reaction_generator import core  # noqa: E402
from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
    balance_pool,
    build_balance_terms,
    build_hyperhomodesmotic_balance_terms,
    describe_user_input,
    is_valid_smiles,
    normalize_user_species,
)
from strain_homodesmotic_reaction_generator.data import UserSpecies  # noqa: E402

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")

CYCLOPROPANE = "C1CC1"
#: The environment cyclopropane matches, and the reference it gets by default.
CP_ENVIRONMENT = "secondary carbon - secondary carbon"
CP_DEFAULT_REFERENCE = "CCCC"


class _FakeContext:
    def __init__(self, smiles=CYCLOPROPANE):
        self._molecule = Chem.MolFromSmiles(smiles) if smiles else None
        self.status_messages = []
        self.windows = {}
        self.loaded_smiles = []

    @property
    def current_molecule(self):
        return self._molecule

    def get_main_window(self):
        return None

    def show_status_message(self, message, timeout=3000):
        self.status_messages.append((message, timeout))

    def load_from_smiles(self, smiles):
        self.loaded_smiles.append(smiles)

    def register_window(self, key, window):
        self.windows[key] = window

    def get_window(self, key):
        return self.windows.get(key)


def _dialog(smiles=CYCLOPROPANE):
    return ui.HomodesmoticAnalyzerDialog(_FakeContext(smiles))


def _equation(result):
    return result.equation_text.splitlines()[3]


# ---------------------------------------------------------------------------
# Validation and normalisation
# ---------------------------------------------------------------------------


def test_is_valid_smiles_accepts_and_rejects():
    assert is_valid_smiles("CCO") is True
    assert is_valid_smiles("  CCO  ") is True
    assert is_valid_smiles("not-a-molecule") is False
    assert is_valid_smiles("") is False
    assert is_valid_smiles("   ") is False


def test_normalize_user_species_drops_unparseable_entries():
    kept, rejected = normalize_user_species(
        [UserSpecies("CCO"), UserSpecies("XYZ!"), UserSpecies("   ")]
    )
    assert [entry.smiles for entry in kept] == ["CCO"]
    assert rejected == ("XYZ!",)


def test_normalize_user_species_collapses_duplicates_keeping_required():
    kept, _rejected = normalize_user_species(
        [UserSpecies("CCO", "first"), UserSpecies("CCO", "second", required=True)]
    )
    assert len(kept) == 1
    assert kept[0].name == "first"
    assert kept[0].required is True


def test_normalize_user_species_strips_surrounding_whitespace():
    kept, _rejected = normalize_user_species([UserSpecies("  CCO ", "  label  ")])
    assert kept[0].smiles == "CCO"
    assert kept[0].name == "label"


def test_balance_pool_reuses_a_library_entry_but_still_flags_it_as_the_users():
    pool, user_smiles = balance_pool([UserSpecies("CC").as_balance_species()])
    assert [species.smiles for species in pool].count("CC") == 1
    assert "CC" in user_smiles


def test_balance_pool_appends_a_genuinely_new_species():
    pool, user_smiles = balance_pool([UserSpecies("CCCCCCC").as_balance_species()])
    assert pool[-1].smiles == "CCCCCCC"
    assert user_smiles == frozenset({"CCCCCCC"})


# ---------------------------------------------------------------------------
# Reference overrides
# ---------------------------------------------------------------------------


def test_override_replaces_the_proposed_reference_molecule():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    smiles = [match.reference_smiles for match in result.matches if match.count > 0]
    assert smiles == ["CCCCC"]
    assert all(match.user_defined for match in result.matches if match.count > 0)


def test_override_keeps_the_reaction_balanced_and_reclassifies_it():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    assert result.reaction_type in {"Hyperhomodesmotic", "Homodesmotic"}
    assert not result.unresolved_left_atoms
    assert not result.unresolved_right_atoms


def test_a_reference_that_cancels_itself_away_is_reported():
    """Benzene for an alkane environment balances, but measures nothing.

    The solver clears the mismatch by putting the same benzene on the left,
    and the two cancel. What is left is a perfectly valid reaction that says
    nothing about the ring, so the result has to say so rather than let the
    clean reaction type imply the substitution worked.
    """
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "c1ccccc1"})
    assert result.cancelled_references == (CP_ENVIRONMENT,)
    assert not [match for match in result.matches if match.count > 0]
    assert "cancelled out" in result.equation_text


def test_a_workable_override_does_not_report_a_cancellation():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    assert result.cancelled_references == ()


def test_a_cancelled_reference_raises_the_warning_label():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "c1ccccc1"))
    assert dialog.warning_label.isVisible() is True
    assert "cancelled out" in dialog.warning_label.text()


def test_unparseable_override_falls_back_to_the_default_reference():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "not-a-molecule"})
    smiles = [match.reference_smiles for match in result.matches if match.count > 0]
    assert smiles == [CP_DEFAULT_REFERENCE]
    assert result.reference_overrides == ()


def test_override_for_an_undetected_environment_changes_nothing():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    baseline = analyze_molecule(mol)
    result = analyze_molecule(mol, {"primary carbon - primary carbon": "CCCCC"})
    assert _equation(result) == _equation(baseline)


def test_override_is_recorded_verbatim_on_the_result():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "  CCCCC  "})
    assert result.reference_overrides == ((CP_ENVIRONMENT, "CCCCC"),)


# ---------------------------------------------------------------------------
# User balance species
# ---------------------------------------------------------------------------


def test_an_optional_user_species_is_recorded_without_forcing_its_use():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, user_species=[UserSpecies("CCCCCCC", "heptane")])
    assert result.user_species[0].smiles == "CCCCCCC"
    assert result.unmet_required == ()


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_a_required_species_actually_appears_in_the_equation():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies("CCCCCCC", "my heptane", required=True)]
    )
    used = {
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count > 0
    }
    assert "CCCCCCC" in used
    assert result.unmet_required == ()


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_a_required_species_survives_the_both_sides_cancellation():
    """Satisfying "required" with one copy on each side cancels to nothing."""
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, user_species=[UserSpecies("CCCCCCC", required=True)])
    left = sum(
        term.count for term in result.left_balance_terms if term.smiles == "CCCCCCC"
    )
    right = sum(
        term.count for term in result.right_balance_terms if term.smiles == "CCCCCCC"
    )
    assert (left > 0) != (right > 0)


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_a_required_species_keeps_the_reaction_balanced():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies("C1CCCCC1", required=True)]
    )
    assert result.reaction_type in {"Hyperhomodesmotic", "Homodesmotic"}
    assert not result.unresolved_left_atoms
    assert not result.unresolved_right_atoms
    assert result.lhs_bonds == result.rhs_bonds


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_an_impossible_requirement_is_dropped_and_reported():
    """A balance the user cannot have still beats no balance at all."""
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies("[SiH4]", "silane", required=True)]
    )
    assert result.unmet_required == ("[SiH4]",)
    assert result.reaction_type == "Hyperhomodesmotic"


def test_user_terms_are_flagged_so_the_report_can_mark_them():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies("CCCCCCC", "my heptane", required=True)]
    )
    flagged = [
        term
        for term in result.left_balance_terms + result.right_balance_terms
        if term.smiles == "CCCCCCC"
    ]
    assert flagged and all(term.user_defined for term in flagged)


def test_too_many_required_species_are_reported_rather_than_searched():
    required = [f"C{'C' * n}" for n in range(core._MAX_REQUIRED_SEARCH + 1)]
    left, right, ok, unmet = build_hyperhomodesmotic_balance_terms(
        Counter({"C(H3)(O_s0)(O_d0)(N_s0)(N_d0)(N_t0)(=C0)(#C0)": 2}),
        Counter({"C": 2, "H": 6}),
        tuple(UserSpecies(smiles).as_balance_species() for smiles in required),
        tuple(required),
    )
    assert set(required).issubset(set(unmet)) or ok is False


# ---------------------------------------------------------------------------
# Fallback (no MILP) path
# ---------------------------------------------------------------------------


def test_build_balance_terms_can_draw_on_a_user_species():
    terms, unresolved, unmet = build_balance_terms(
        Counter({"C": 7, "H": 16}),
        (UserSpecies("CCCCCCC", "heptane").as_balance_species(),),
        ("CCCCCCC",),
    )
    assert not unresolved
    assert unmet == ()
    assert [term.smiles for term in terms] == ["CCCCCCC"]
    assert terms[0].user_defined is True


def test_build_balance_terms_reports_a_required_species_that_cannot_fit():
    terms, unresolved, unmet = build_balance_terms(
        Counter({"C": 2, "H": 6}),
        (UserSpecies("CCCCCCC").as_balance_species(),),
        ("CCCCCCC",),
    )
    assert unmet == ("CCCCCCC",)
    assert not unresolved
    assert [term.smiles for term in terms] == ["CC"]


def test_build_balance_terms_reports_a_required_species_outside_the_pool():
    _terms, _unresolved, unmet = build_balance_terms(
        Counter({"C": 2, "H": 6}), (), ("CCCCCCC",)
    )
    assert unmet == ("CCCCCCC",)


def test_build_balance_terms_with_nothing_needed_still_seeds_the_requirement():
    terms, unresolved, unmet = build_balance_terms(Counter(), (), ())
    assert terms == ()
    assert not unresolved
    assert unmet == ()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_describe_user_input_quotes_entries_as_typed():
    lines = describe_user_input(
        (("an environment", "CCCCC"),),
        (UserSpecies("CCCCCCC", "my heptane", required=True),),
        ("[SiH4]",),
    )
    assert lines == (
        "Reference override - an environment: CCCCC",
        "User species - CCCCCCC (my heptane) [required]",
        "Required species could not be placed: [SiH4]",
    )


def test_describe_user_input_is_empty_when_nothing_was_supplied():
    assert describe_user_input((), (), ()) == ()


def test_text_report_carries_a_user_input_section():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    assert "User-specified input:" in result.equation_text
    assert f"Reference override - {CP_ENVIRONMENT}: CCCCC" in result.equation_text


def test_text_report_says_so_when_everything_is_a_default():
    result = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    assert "none (all species are the built-in defaults)" in result.equation_text


def test_html_report_marks_user_species_with_the_user_colour():
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    assert core.USER_COLOR in result.equation_html
    assert "User-specified input" in result.equation_html


def test_html_export_includes_a_user_input_card(tmp_path):
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, user_species=[UserSpecies("CCCCCCC", "my heptane")])
    target = tmp_path / "report.html"
    core.export_analysis(target, result)
    text = target.read_text(encoding="utf-8")
    assert "User-specified input" in text
    assert "CCCCCCC (my heptane)" in text


def test_csv_export_lists_user_input(tmp_path):
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(mol, {CP_ENVIRONMENT: "CCCCC"})
    target = tmp_path / "report.csv"
    core.export_analysis(target, result)
    text = target.read_text(encoding="utf-8")
    assert "User-specified input" in text
    assert "CCCCC" in text


def test_csv_export_says_so_when_nothing_was_supplied(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    target = tmp_path / "report.csv"
    core.export_analysis(target, result)
    assert "none (all species are the built-in defaults)" in target.read_text(
        encoding="utf-8"
    )


def test_html_export_omits_the_card_when_nothing_was_supplied(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles(CYCLOPROPANE))
    target = tmp_path / "report.html"
    core.export_analysis(target, result)
    assert "User-specified input</h3>" not in target.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Add Species dialog
# ---------------------------------------------------------------------------


def test_add_dialog_offers_the_reference_type_only_when_environments_exist():
    with_env = ui.AddSpeciesDialog(None, ("an environment",))
    without_env = ui.AddSpeciesDialog(None, ())
    assert with_env.type_combo.count() == 2
    assert without_env.type_combo.count() == 1


def test_add_dialog_shows_the_environment_combo_only_for_an_override():
    dialog = ui.AddSpeciesDialog(None, ("an environment",))
    assert dialog.environment_combo.isVisible() is False
    assert dialog.required_check.isVisible() is True

    dialog.type_combo.setCurrentText(ui._ADD_REFERENCE)
    assert dialog.environment_combo.isVisible() is True
    assert dialog.required_check.isVisible() is False


def test_add_dialog_rejects_an_unparseable_smiles_without_closing():
    dialog = ui.AddSpeciesDialog(None, ())
    dialog.smiles_edit.setText("not-a-molecule")
    dialog.confirm()
    assert dialog.entry is None
    assert dialog.error_label.isVisible() is True


def test_add_dialog_rejects_an_empty_smiles():
    dialog = ui.AddSpeciesDialog(None, ())
    dialog.confirm()
    assert dialog.entry is None
    assert "Enter a SMILES" in dialog.error_label.text()


def test_add_dialog_builds_a_balance_entry():
    dialog = ui.AddSpeciesDialog(None, ())
    dialog.smiles_edit.setText("CCCCCCC")
    dialog.name_edit.setText("my heptane")
    dialog.required_check.setChecked(True)
    dialog.confirm()
    kind, species = dialog.entry
    assert kind == "balance"
    assert species == UserSpecies("CCCCCCC", "my heptane", True)


def test_add_dialog_builds_a_reference_entry():
    dialog = ui.AddSpeciesDialog(None, ("an environment",))
    dialog.type_combo.setCurrentText(ui._ADD_REFERENCE)
    dialog.smiles_edit.setText("CCCCC")
    dialog.confirm()
    assert dialog.entry == ("reference", "an environment", "CCCCC")


# ---------------------------------------------------------------------------
# Dialog wiring
# ---------------------------------------------------------------------------


def test_dialog_starts_with_no_user_input():
    dialog = _dialog()
    assert dialog.user_species == []
    assert dialog.reference_overrides == {}
    assert dialog.user_table.rowCount() == 0


def test_adding_a_balance_entry_reanalyzes_with_it():
    dialog = _dialog()
    dialog.apply_new_entry(
        ("balance", UserSpecies("CCCCCCC", "my heptane", required=True))
    )
    assert dialog.user_table.rowCount() == 1
    assert dialog.last_result.user_species[0].smiles == "CCCCCCC"


def test_adding_a_reference_entry_reanalyzes_with_it():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    assert dialog.reference_overrides == {CP_ENVIRONMENT: "CCCCC"}
    assert dialog.last_result.reference_overrides == ((CP_ENVIRONMENT, "CCCCC"),)


def test_adding_the_same_smiles_twice_replaces_rather_than_duplicates():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "first")))
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "second", True)))
    assert len(dialog.user_species) == 1
    assert dialog.user_species[0].name == "second"
    assert dialog.user_species[0].required is True


def test_environment_names_offered_are_the_detected_ones():
    dialog = _dialog()
    assert dialog._environment_names() == (CP_ENVIRONMENT,)


def test_environment_names_fall_back_to_every_rule_with_no_molecule():
    dialog = _dialog(smiles=None)
    assert len(dialog._environment_names()) > 1


def test_no_results_cell_is_editable():
    """The results table reports the solver's answer; edits belong below it."""
    import PyQt6.QtCore as qtc

    dialog = _dialog()
    editable = qtc.Qt.ItemFlag.ItemIsEditable
    assert dialog.table.rowCount() > 0
    for row in range(dialog.table.rowCount()):
        for col in range(3):
            assert not dialog.table.item(row, col).flags() & editable


def test_a_reference_row_points_at_the_override_route():
    dialog = _dialog()
    row = next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["kind"] == ui._KIND_REFERENCE
    )
    assert "Reference molecule override" in dialog.table.item(row, 2).toolTip()


def test_a_balance_row_points_at_the_species_box():
    dialog = _dialog()
    row = next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["kind"] != ui._KIND_REFERENCE
    )
    assert "Add your own below" in dialog.table.item(row, 2).toolTip()


def test_editing_a_user_table_smiles_updates_the_species():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.user_table.item(0, 2).setText("CCCCCC")
    assert [species.smiles for species in dialog.user_species] == ["CCCCCC"]


def test_editing_a_user_table_name_updates_the_label():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.user_table.item(0, 1).setText("renamed")
    assert dialog.user_species[0].name == "renamed"


def test_editing_a_user_table_override_smiles_updates_the_override():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.user_table.item(0, 2).setText("CCCCCC")
    assert dialog.reference_overrides == {CP_ENVIRONMENT: "CCCCCC"}


def test_editing_a_user_table_smiles_to_rubbish_warns_and_reverts():
    import PyQt6.QtWidgets as qtw

    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    qtw.QMessageBox.warning_calls.clear()
    dialog.user_table.item(0, 2).setText("not-a-molecule")
    assert [species.smiles for species in dialog.user_species] == ["CCCCCCC"]
    assert qtw.QMessageBox.warning_calls


def test_removing_a_selected_species_reanalyzes_without_it():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.user_table.selectRows([0])
    dialog.remove_selected_species()
    assert dialog.user_species == []
    assert dialog.user_table.rowCount() == 0


def test_removing_a_selected_override_reanalyzes_without_it():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.user_table.selectRows([0])
    dialog.remove_selected_species()
    assert dialog.reference_overrides == {}


def test_removing_with_nothing_selected_says_so():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC")))
    dialog.remove_selected_species()
    assert dialog.user_species
    assert "Select a row to remove." in [
        message for message, _timeout in dialog.context.status_messages
    ]


def test_reset_clears_both_kinds_of_entry():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC")))
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.reset_species()
    assert dialog.user_species == []
    assert dialog.reference_overrides == {}
    assert dialog.user_table.rowCount() == 0


def test_reset_with_nothing_to_clear_is_a_no_op():
    dialog = _dialog()
    before = dialog.last_result
    dialog.reset_species()
    assert dialog.last_result is before


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_an_impossible_requirement_raises_the_warning_label():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("[SiH4]", "silane", required=True)))
    assert dialog.warning_label.isVisible() is True
    assert "[SiH4]" in dialog.warning_label.text()


def test_the_warning_label_clears_once_the_requirement_is_removed():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("[SiH4]", "silane", required=True)))
    dialog.reset_species()
    assert dialog.warning_label.isVisible() is False


# ---------------------------------------------------------------------------
# Re-ordering the results table
# ---------------------------------------------------------------------------


def test_rows_are_grouped_reference_then_left_then_right_by_default():
    dialog = _dialog()
    kinds = [data["kind"] for data in dialog._table_rows]
    assert kinds == sorted(kinds)


def test_clicking_a_header_sorts_then_reverses_then_restores_the_grouping():
    dialog = _dialog()
    grouped = [data["label"] for data in dialog._table_rows]

    dialog._on_header_clicked(2)
    ascending = [data["smiles"] for data in dialog._table_rows]
    assert ascending == sorted(ascending)

    dialog._on_header_clicked(2)
    descending = [data["smiles"] for data in dialog._table_rows]
    assert descending == sorted(descending, reverse=True)

    dialog._on_header_clicked(2)
    assert [data["label"] for data in dialog._table_rows] == grouped


def test_sorting_by_count_orders_by_count():
    dialog = _dialog()
    dialog._on_header_clicked(1)
    counts = [data["count"] for data in dialog._table_rows]
    assert counts == sorted(counts)


def test_sorting_by_environment_orders_by_label():
    dialog = _dialog()
    dialog._on_header_clicked(0)
    labels = [data["label"] for data in dialog._table_rows]
    assert labels == sorted(labels)


def test_clicking_the_action_header_does_nothing():
    dialog = _dialog()
    before = [data["label"] for data in dialog._table_rows]
    dialog._on_header_clicked(3)
    assert dialog._sort_state is None
    assert [data["label"] for data in dialog._table_rows] == before


def test_switching_sort_column_starts_ascending_again():
    dialog = _dialog()
    dialog._on_header_clicked(2)
    dialog._on_header_clicked(2)
    dialog._on_header_clicked(1)
    assert dialog._sort_state == (1, False)


def test_a_sorted_table_keeps_its_load_buttons_aligned_with_its_rows():
    dialog = _dialog()
    dialog._on_header_clicked(2)
    for row, data in enumerate(dialog._table_rows):
        assert dialog.table.item(row, 2).text() == data["smiles"]
        assert dialog.table.cellWidget(row, 3) is not None
