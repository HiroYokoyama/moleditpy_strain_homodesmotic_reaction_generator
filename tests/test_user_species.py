"""
User-defined reference molecules and balance species.

Covers the core plumbing (overrides, an extended species library, the
``required`` constraint) and the dialog wiring that feeds it, including the
Add Species dialog and the re-orderable results table.
"""

from collections import Counter
from pathlib import Path
import sys
import time

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


def _apply(dialog, *entries):
    """Stage entries the way the dialogs do, then run the analysis."""
    for entry in entries:
        dialog.apply_new_entry(entry)
    dialog.refresh_analysis()
    return dialog


def _row_index(dialog, smiles):
    """Index of the one row carrying this SMILES."""
    return next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["smiles"] == smiles
    )


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
    dialog = _apply(_dialog(), ("reference", CP_ENVIRONMENT, "c1ccccc1"))
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


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
@pytest.mark.parametrize(
    "order",
    [("CCCCCCC", "[SiH4]"), ("[SiH4]", "CCCCCCC")],
    ids=["possible-first", "impossible-first"],
)
def test_only_the_impossible_requirement_is_dropped(order):
    """One unsatisfiable requirement must not take a workable one down with it."""
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies(smiles, "", True) for smiles in order]
    )
    assert result.unmet_required == ("[SiH4]",)
    used = {
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count > 0
    }
    assert "CCCCCCC" in used


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


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_many_required_species_stay_one_solve_not_two_to_the_n():
    """Side choice is a binary per species, so seven cost about what one does."""
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    species = [UserSpecies("C" * (n + 1), f"s{n}", True) for n in range(7)]

    start = time.perf_counter()
    result = analyze_molecule(mol, user_species=species)
    elapsed = time.perf_counter() - start

    assert elapsed < 10
    used = {
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count > 0
    }
    honoured = [
        entry.smiles for entry in species if entry.smiles not in result.unmet_required
    ]
    assert honoured
    assert all(smiles in used for smiles in honoured)


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_a_reference_is_never_cancelled_away_to_make_the_balance_cheaper():
    """The cheapest balance can erase the references; that answer is useless."""
    mol = Chem.MolFromSmiles(CYCLOPROPANE)
    result = analyze_molecule(
        mol, user_species=[UserSpecies("CCCCCCC", "heptane", required=True)]
    )
    assert result.cancelled_references == ()
    assert [match.count for match in result.matches if match.count > 0] == [3]


#: A shortfall only ethane on the left can cover with one molecule.
_ETHANE_GROUP = "C(H3)(O_s0)(O_d0)(N_s0)(N_d0)(N_t0)(=C0)(#C0)"


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_naming_a_reference_keeps_it_off_the_left():
    group_delta = Counter({_ETHANE_GROUP: 2})
    atom_delta = Counter({"C": 2, "H": 6})

    free, _r, _ok, _u = build_hyperhomodesmotic_balance_terms(
        group_delta, atom_delta, (), (), ()
    )
    guarded, _r2, _ok2, _u2 = build_hyperhomodesmotic_balance_terms(
        group_delta, atom_delta, (), (), ("CC",)
    )
    assert [term.smiles for term in free] == ["CC"]
    assert "CC" not in [term.smiles for term in guarded]


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_protecting_the_references_is_relaxed_rather_than_failing(monkeypatch):
    """With the ban unsatisfiable, a cancelled answer still beats no answer."""
    real = core._solve_with_required
    attempts = []

    def only_without_the_ban(*args, **kwargs):
        banned = args[5] if len(args) > 5 else kwargs.get("banned_left_rows", ())
        attempts.append(tuple(banned))
        if banned:
            return None
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "_solve_with_required", only_without_the_ban)

    left, right, ok, _unmet = build_hyperhomodesmotic_balance_terms(
        Counter({_ETHANE_GROUP: 2}), Counter({"C": 2, "H": 6}), (), (), ("CC",)
    )
    assert ok is True
    assert [term.smiles for term in left] == ["CC"]
    assert right == ()
    assert attempts[0] and not attempts[-1]


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
    assert all(row["source"] == "default" for row in dialog._table_rows)


def test_adding_a_balance_entry_stages_it_then_analyze_applies_it():
    dialog = _dialog()
    entry = ("balance", UserSpecies("CCCCCCC", "my heptane", required=True))

    dialog.apply_new_entry(entry)
    assert dialog.last_result.user_species == ()
    assert any(row["source"] == ui._SOURCE_PENDING for row in dialog._table_rows)

    dialog.refresh_analysis()
    assert dialog.last_result.user_species[0].smiles == "CCCCCCC"
    assert any(row["source"] == ui._SOURCE_YOURS for row in dialog._table_rows)


def test_adding_a_reference_entry_stages_it_then_analyze_applies_it():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    assert dialog.reference_overrides == {CP_ENVIRONMENT: "CCCCC"}
    assert dialog.last_result.reference_overrides == ()

    dialog.refresh_analysis()
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


def test_no_cell_is_editable():
    """The table reports the draft; changes go through Add and Edit."""
    import PyQt6.QtCore as qtc

    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    editable = qtc.Qt.ItemFlag.ItemIsEditable
    assert dialog.table.rowCount() > 0
    for row in range(dialog.table.rowCount()):
        for col in range(ui._COL_ACTION):
            assert not dialog.table.item(row, col).flags() & editable


def test_there_is_only_one_table():
    # The Qt stub auto-mocks unknown attributes, so hasattr proves nothing.
    dialog = _dialog()
    assert "user_table" not in dialog.__dict__


def test_edit_dialog_starts_filled_in_from_a_balance_entry():
    dialog = ui.AddSpeciesDialog(
        None, (), ("balance", UserSpecies("CCCCCCC", "my heptane", True))
    )
    assert dialog.smiles_edit.text() == "CCCCCCC"
    assert dialog.name_edit.text() == "my heptane"
    assert dialog.required_check.isChecked() is True
    assert dialog.type_combo.currentText() == ui._ADD_BALANCE


def test_edit_dialog_starts_filled_in_from_a_reference_entry():
    dialog = ui.AddSpeciesDialog(
        None, ("an environment",), ("reference", "an environment", "CCCCC")
    )
    assert dialog.type_combo.currentText() == ui._ADD_REFERENCE
    assert dialog.environment_combo.currentText() == "an environment"
    assert dialog.smiles_edit.text() == "CCCCC"


def test_edit_dialog_offers_an_environment_that_is_no_longer_detected():
    """Editing must not silently retarget an override at some other environment."""
    dialog = ui.AddSpeciesDialog(
        None, ("another environment",), ("reference", "gone", "CCCCC")
    )
    assert dialog.environment_combo.currentText() == "gone"


@pytest.mark.parametrize(
    "existing", [None, ("balance", UserSpecies("CC"))], ids=["add", "edit"]
)
def test_enter_confirms_the_dialog(existing):
    dialog = ui.AddSpeciesDialog(None, (), existing)
    assert dialog.add_button.isDefault() is True
    assert dialog.cancel_button.autoDefault() is False


@pytest.mark.parametrize(
    "existing", [None, ("balance", UserSpecies("CC"))], ids=["add", "edit"]
)
def test_the_smiles_field_starts_focused(existing):
    dialog = ui.AddSpeciesDialog(None, (), existing)
    assert dialog.smiles_edit.hasFocus() is True


def test_edit_dialog_says_save_and_locks_the_type():
    dialog = ui.AddSpeciesDialog(None, (), ("balance", UserSpecies("CC")))
    assert dialog.add_button._text == "Save"
    assert dialog.type_combo.isEnabled() is False


def test_editing_a_species_smiles_replaces_the_old_entry():
    dialog = _dialog()
    old = ("balance", UserSpecies("CCCCCCC", "heptane"))
    dialog.apply_new_entry(old)
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCC", "hexane")), replacing=old)
    assert [species.smiles for species in dialog.user_species] == ["CCCCCC"]
    assert not any(row["smiles"] == "CCCCCCC" for row in dialog._table_rows)


def test_editing_a_species_name_only_keeps_one_entry():
    dialog = _dialog()
    old = ("balance", UserSpecies("CCCCCCC", "heptane"))
    dialog.apply_new_entry(old)
    dialog.apply_new_entry(
        ("balance", UserSpecies("CCCCCCC", "renamed")), replacing=old
    )
    assert len(dialog.user_species) == 1
    assert dialog.user_species[0].name == "renamed"


def test_editing_an_override_onto_another_environment_drops_the_old_one():
    dialog = _dialog()
    old = ("reference", CP_ENVIRONMENT, "CCCCC")
    dialog.apply_new_entry(old)
    dialog.apply_new_entry(("reference", "somewhere else", "CCCCCC"), replacing=old)
    assert dialog.reference_overrides == {"somewhere else": "CCCCCC"}


def test_an_edited_species_is_still_marked_as_the_users():
    """Editing onto a SMILES the library already has must not lose the mark."""
    dialog = _dialog()
    old = ("balance", UserSpecies("CCCCCCC", "heptane", required=True))
    dialog.apply_new_entry(old)
    dialog.apply_new_entry(
        ("balance", UserSpecies("CCCCCC", "hexane", required=True)), replacing=old
    )
    dialog.refresh_analysis()
    terms = [
        term
        for term in dialog.last_result.left_balance_terms
        + dialog.last_result.right_balance_terms
        if term.smiles == "CCCCCC" and term.count > 0
    ]
    assert terms and all(term.user_defined for term in terms)


def test_selecting_a_user_row_yields_that_entry():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.table.selectRows([_row_index(dialog, "CCCCCCC")])
    kind, species = dialog._selected_row()["edit_entry"]
    assert kind == "balance"
    assert species.smiles == "CCCCCCC"


def test_selecting_a_reference_row_yields_a_reference_entry():
    dialog = _dialog()
    dialog.table.selectRows([_row_index(dialog, CP_DEFAULT_REFERENCE)])
    assert dialog._selected_row()["edit_entry"] == (
        "reference",
        CP_ENVIRONMENT,
        CP_DEFAULT_REFERENCE,
    )


def test_a_default_reference_can_be_edited_but_not_removed():
    dialog = _dialog()
    row = dialog._table_rows[_row_index(dialog, CP_DEFAULT_REFERENCE)]
    assert row["edit_entry"] is not None
    assert row["removable"] is False


def test_a_solver_chosen_balance_species_can_be_neither():
    dialog = _dialog()
    row = next(
        data
        for data in dialog._table_rows
        if data["role"] in (ui._ROLE_LEFT, ui._ROLE_RIGHT)
    )
    assert row["edit_entry"] is None
    assert row["removable"] is False


def test_editing_a_solver_row_says_why_it_cannot():
    dialog = _dialog()
    row = next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["role"] in (ui._ROLE_LEFT, ui._ROLE_RIGHT)
    )
    dialog.table.selectRows([row])
    dialog.edit_selected_species()
    assert any(
        "can be edited" in message
        for message, _timeout in dialog.context.status_messages
    )


def test_removing_a_solver_row_says_why_it_cannot():
    dialog = _dialog()
    row = next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["role"] in (ui._ROLE_LEFT, ui._ROLE_RIGHT)
    )
    dialog.table.selectRows([row])
    dialog.remove_selected_species()
    assert any(
        "can be removed" in message
        for message, _timeout in dialog.context.status_messages
    )


def test_double_clicking_a_row_opens_its_edit_dialog(monkeypatch):
    opened = []

    class _Dialog:
        entry = None

        def __init__(self, parent, environments, existing=None):
            opened.append(existing)

        def exec(self):
            return 0

    monkeypatch.setattr(ui, "AddSpeciesDialog", _Dialog)
    dialog = _dialog()
    row = _row_index(dialog, CP_DEFAULT_REFERENCE)
    dialog.table.doubleClick(row)
    assert opened == [("reference", CP_ENVIRONMENT, CP_DEFAULT_REFERENCE)]


def test_double_clicking_selects_the_row_it_was_given():
    dialog = _dialog()
    row = _row_index(dialog, CP_DEFAULT_REFERENCE)
    dialog.table.doubleClick(row)
    assert dialog._selected_row()["smiles"] == CP_DEFAULT_REFERENCE


def test_editing_needs_exactly_one_selected_row():
    dialog = _dialog()
    dialog.table.selectRows([0, 1])
    dialog.edit_selected_species()
    assert "Select one row to edit." in [
        message for message, _timeout in dialog.context.status_messages
    ]


def test_editing_with_nothing_selected_says_so():
    dialog = _dialog()
    dialog.edit_selected_species()
    assert "Select one row to edit." in [
        message for message, _timeout in dialog.context.status_messages
    ]


def test_removing_a_selected_species_reanalyzes_without_it():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.table.selectRows([_row_index(dialog, "CCCCCCC")])
    dialog.remove_selected_species()
    assert dialog.user_species == []
    assert not any(row["smiles"] == "CCCCCCC" for row in dialog._table_rows)


def test_removing_a_selected_override_reverts_to_the_default_reference():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.table.selectRows([_row_index(dialog, "CCCCC")])
    dialog.remove_selected_species()
    assert dialog.reference_overrides == {}
    assert _row_index(dialog, CP_DEFAULT_REFERENCE) is not None


def test_removing_with_nothing_selected_says_so():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC")))
    dialog.remove_selected_species()
    assert dialog.user_species
    assert "Select one row to remove." in [
        message for message, _timeout in dialog.context.status_messages
    ]


def test_reset_clears_both_kinds_of_entry():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC")))
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.reset_species()
    assert dialog.user_species == []
    assert dialog.reference_overrides == {}
    assert all(row["source"] == "default" for row in dialog._table_rows)


def test_reset_with_nothing_to_clear_is_a_no_op():
    dialog = _dialog()
    before = dialog.last_result
    dialog.reset_species()
    assert dialog.last_result is before


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_an_impossible_requirement_raises_the_warning_label():
    dialog = _apply(
        _dialog(), ("balance", UserSpecies("[SiH4]", "silane", required=True))
    )
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


def test_rows_are_grouped_reference_then_balance_then_unused_by_default():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("[SiH4]", "silane")))
    roles = [data["role"] for data in dialog._table_rows]
    assert roles == sorted(roles)
    assert roles[-1] == ui._ROLE_UNUSED


def test_clicking_a_header_sorts_then_reverses_then_restores_the_grouping():
    dialog = _dialog()
    grouped = [data["name"] for data in dialog._table_rows]

    dialog._on_header_clicked(ui._COL_SMILES)
    ascending = [data["smiles"] for data in dialog._table_rows]
    assert ascending == sorted(ascending)

    dialog._on_header_clicked(ui._COL_SMILES)
    descending = [data["smiles"] for data in dialog._table_rows]
    assert descending == sorted(descending, reverse=True)

    dialog._on_header_clicked(ui._COL_SMILES)
    assert [data["name"] for data in dialog._table_rows] == grouped


def test_sorting_by_count_orders_by_count():
    dialog = _dialog()
    dialog._on_header_clicked(ui._COL_COUNT)
    counts = [data["count"] for data in dialog._table_rows]
    assert counts == sorted(counts)


def test_sorting_by_count_does_not_trip_over_a_row_with_no_count():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("[SiH4]", "silane")))
    dialog._on_header_clicked(ui._COL_COUNT)
    assert dialog._table_rows[0]["count"] is None


def test_sorting_by_name_orders_by_name():
    dialog = _dialog()
    dialog._on_header_clicked(ui._COL_NAME)
    names = [data["name"] for data in dialog._table_rows]
    assert names == sorted(names)


def test_sorting_by_source_orders_by_source():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog._on_header_clicked(ui._COL_SOURCE)
    sources = [data["source"] for data in dialog._table_rows]
    assert sources == sorted(sources)


def test_clicking_the_action_header_does_nothing():
    dialog = _dialog()
    before = [data["name"] for data in dialog._table_rows]
    dialog._on_header_clicked(ui._COL_ACTION)
    assert dialog._sort_state is None
    assert [data["name"] for data in dialog._table_rows] == before


def test_switching_sort_column_starts_ascending_again():
    dialog = _dialog()
    dialog._on_header_clicked(ui._COL_SMILES)
    dialog._on_header_clicked(ui._COL_SMILES)
    dialog._on_header_clicked(ui._COL_COUNT)
    assert dialog._sort_state == (ui._COL_COUNT, False)


def test_a_sorted_table_keeps_its_load_buttons_aligned_with_its_rows():
    dialog = _dialog()
    dialog._on_header_clicked(ui._COL_SMILES)
    for row, data in enumerate(dialog._table_rows):
        assert dialog.table.item(row, ui._COL_SMILES).text() == data["smiles"]
        assert dialog.table.cellWidget(row, ui._COL_ACTION) is not None


# ---------------------------------------------------------------------------
# Staged changes
# ---------------------------------------------------------------------------


def test_nothing_is_pending_to_begin_with():
    dialog = _dialog()
    assert dialog.pending_label.isVisible() is False


def test_a_staged_change_raises_the_pending_banner():
    """A full re-analysis is seconds of modal wait; edits must not each cost one."""
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    assert dialog.pending_label.isVisible() is True
    assert "1 change pending" in dialog.pending_label.text()


def test_the_banner_counts_several_staged_changes():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC")))
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    assert "2 changes pending" in dialog.pending_label.text()


def test_analyzing_clears_the_pending_banner():
    dialog = _apply(_dialog(), ("balance", UserSpecies("CCCCCCC", "heptane")))
    assert dialog.pending_label.isVisible() is False


def test_the_analyze_button_is_highlighted_while_edits_are_waiting():
    dialog = _dialog()
    assert dialog.analyze_button.styleSheet() == ""

    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    assert ui._USER_COLOR in dialog.analyze_button.styleSheet()

    dialog.refresh_analysis()
    assert dialog.analyze_button.styleSheet() == ""


def test_a_staged_override_shows_on_the_reference_row_as_pending():
    dialog = _dialog()
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    row = dialog._table_rows[_row_index(dialog, "CCCCC")]
    assert row["role"] == ui._ROLE_REFERENCE
    assert row["source"] == ui._SOURCE_PENDING


def test_removing_an_applied_override_is_pending_until_analyzed():
    dialog = _apply(_dialog(), ("reference", CP_ENVIRONMENT, "CCCCC"))
    dialog.table.selectRows([_row_index(dialog, "CCCCC")])
    dialog.remove_selected_species()
    row = dialog._table_rows[_row_index(dialog, CP_DEFAULT_REFERENCE)]
    assert row["source"] == ui._SOURCE_REMOVED

    dialog.refresh_analysis()
    row = dialog._table_rows[_row_index(dialog, CP_DEFAULT_REFERENCE)]
    assert row["source"] == "default"


def test_removing_an_applied_species_is_pending_until_analyzed():
    dialog = _apply(
        _dialog(), ("balance", UserSpecies("CCCCCCC", "heptane", required=True))
    )
    dialog.table.selectRows([_row_index(dialog, "CCCCCCC")])
    dialog.remove_selected_species()
    assert any(row["source"] == ui._SOURCE_REMOVED for row in dialog._table_rows)

    dialog.refresh_analysis()
    assert not any(row["smiles"] == "CCCCCCC" for row in dialog._table_rows)


def test_reset_is_pending_until_analyzed():
    dialog = _apply(_dialog(), ("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.reset_species()
    assert dialog.pending_label.isVisible() is True
    dialog.refresh_analysis()
    assert dialog.pending_label.isVisible() is False


def test_the_analyze_button_is_not_named_after_the_molecule():
    """ "Analyze Current Molecule" read as though it would discard your entries."""
    dialog = _dialog()
    assert dialog.analyze_button._text == "Analyze"


# ---------------------------------------------------------------------------
# Entries the equation does not contain
# ---------------------------------------------------------------------------


def test_an_unused_species_still_gets_a_row():
    """Otherwise the entry is invisible and cannot be edited or removed."""
    dialog = _apply(_dialog(), ("balance", UserSpecies("[SiH4]", "silane")))
    row = dialog._table_rows[_row_index(dialog, "[SiH4]")]
    assert row["role"] == ui._ROLE_UNUSED
    assert row["source"] == ui._SOURCE_NOT_USED
    assert row["count"] is None
    assert row["removable"] is True


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
def test_an_unplaceable_required_species_says_so_in_its_row():
    dialog = _apply(
        _dialog(), ("balance", UserSpecies("[SiH4]", "silane", required=True))
    )
    row = dialog._table_rows[_row_index(dialog, "[SiH4]")]
    assert row["source"] == ui._SOURCE_UNPLACEABLE


def test_a_cancelled_override_keeps_a_row_saying_so():
    dialog = _apply(_dialog(), ("reference", CP_ENVIRONMENT, "c1ccccc1"))
    row = dialog._table_rows[_row_index(dialog, "c1ccccc1")]
    assert row["source"] == ui._SOURCE_CANCELLED
    assert row["removable"] is True


def test_an_override_for_an_undetected_environment_says_so():
    dialog = _apply(
        _dialog(), ("reference", "primary carbon - primary carbon", "CCCCC")
    )
    row = dialog._table_rows[_row_index(dialog, "CCCCC")]
    assert row["source"] == ui._SOURCE_NO_ENVIRONMENT


def test_an_unused_species_can_be_removed_from_its_row():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("[SiH4]", "silane")))
    dialog.table.selectRows([_row_index(dialog, "[SiH4]")])
    dialog.remove_selected_species()
    assert dialog.user_species == []
