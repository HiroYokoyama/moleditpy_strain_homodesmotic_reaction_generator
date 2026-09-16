"""
Checking an equation the user wrote rather than one the tool built.

The parser has to survive a line pasted straight out of the report, and the
verdict has to come from the same classifier the generated draft uses -- a
second opinion computed a second way would be free to drift.
"""

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
    build_equation_check_html,
    build_equation_check_text,
    check_equation,
    classify_sides,
    format_equation_sides,
    parse_equation,
)

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")

CYCLOPENTANE = "C1CCCC1"
#: The generated draft for cyclopentane, exactly as the report prints it.
REPORT_LINE = "C1CCCC1 + 5 CCC (propane) -> 5 CCCC"


class _FakeContext:
    def __init__(self, smiles=CYCLOPENTANE):
        self._molecule = Chem.MolFromSmiles(smiles) if smiles else None
        self.status_messages = []
        self.windows = {}

    @property
    def current_molecule(self):
        return self._molecule

    def get_main_window(self):
        return None

    def show_status_message(self, message, timeout=3000):
        self.status_messages.append((message, timeout))

    def load_from_smiles(self, smiles):
        pass

    def register_window(self, key, window):
        self.windows[key] = window

    def get_window(self, key):
        return self.windows.get(key)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_plain_equation_parses():
    left, right, errors = parse_equation("C1CC1 + 3 CCC -> 3 CCCC")
    assert left == [(1, "C1CC1"), (3, "CCC")]
    assert right == [(3, "CCCC")]
    assert errors == []


@pytest.mark.parametrize("arrow", ["->", "=>", "-->", "→", "⟶"])
def test_the_arrows_people_actually_type_are_accepted(arrow):
    left, right, errors = parse_equation(f"CC {arrow} CC")
    assert errors == []
    assert left == right == [(1, "CC")]


def test_a_line_copied_from_the_report_parses_unchanged():
    left, right, errors = parse_equation(REPORT_LINE)
    assert errors == []
    assert left == [(1, CYCLOPENTANE), (5, "CCC")]
    assert right == [(5, "CCCC")]


def test_the_user_annotations_the_report_adds_are_ignored():
    line = (
        "C1CCCC1 + 1 CCCCCCC (my heptane) [user] + 5 CCC (propane) "
        "-> 5 CCCC + 1 CCCCCC (Hexane) [user]"
    )
    left, right, errors = parse_equation(line)
    assert errors == []
    assert [smiles for _count, smiles in left] == [CYCLOPENTANE, "CCCCCCC", "CCC"]
    assert [smiles for _count, smiles in right] == ["CCCC", "CCCCCC"]


def test_a_charge_inside_brackets_does_not_split_the_term():
    """Splitting naively on '+' cuts [NH4+] in half."""
    left, right, errors = parse_equation("CC(=O)[O-] + [NH4+] -> CC(=O)[O-].[NH4+]")
    assert errors == []
    assert left == [(1, "CC(=O)[O-]"), (1, "[NH4+]")]
    assert right == [(1, "CC(=O)[O-].[NH4+]")]


def test_a_bracket_atom_alone_is_a_molecule_not_an_annotation():
    """[SiH4] looks exactly like the [user] tag the report appends."""
    left, _right, errors = parse_equation("C1CC1 + 3 [SiH4] -> 3 CCCC")
    assert errors == []
    assert left == [(1, "C1CC1"), (3, "[SiH4]")]


def test_a_name_after_the_smiles_is_not_kept_as_part_of_it():
    """RDKit reads 'CCC (propane)' as propane *named* '(propane)'."""
    left, _right, errors = parse_equation("CCC (propane) -> CCC")
    assert errors == []
    assert left == [(1, "CCC")]


def test_a_missing_arrow_is_reported():
    _left, _right, errors = parse_equation("C1CC1 + 3 CCC")
    assert errors == ["No arrow found. Write the reaction as 'A + B -> C'."]


def test_two_arrows_are_reported():
    _left, _right, errors = parse_equation("A -> B -> C")
    assert errors == ["More than one arrow found."]


def test_an_empty_side_is_reported():
    _left, _right, errors = parse_equation("-> CCCC")
    assert "Nothing on the left of the arrow." in errors


def test_an_unreadable_term_names_itself():
    _left, _right, errors = parse_equation("C1CC1 + 3 XYZ -> 3 CCCC")
    assert errors == ["Cannot read '3 XYZ' as a molecule."]


def test_a_zero_coefficient_drops_the_term():
    left, _right, errors = parse_equation("C1CC1 + 0 CCC -> 3 CCCC")
    assert errors == []
    assert left == [(1, "C1CC1")]


def test_empty_input_is_reported_rather_than_crashing():
    _left, _right, errors = parse_equation("")
    assert errors


def test_extra_whitespace_does_not_matter():
    left, right, errors = parse_equation("  C1CC1   +   3   CCC  ->   3   CCCC  ")
    assert errors == []
    assert left == [(1, "C1CC1"), (3, "CCC")]
    assert right == [(3, "CCCC")]


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_the_generated_draft_checks_out_as_what_it_claims():
    """The tester and the generator must agree about the same reaction."""
    generated = analyze_molecule(Chem.MolFromSmiles(CYCLOPENTANE))
    checked = check_equation(REPORT_LINE)
    assert checked.reaction_type == generated.reaction_type == "Hyperhomodesmotic"
    assert not checked.atom_delta
    assert not checked.group_delta
    assert not checked.bond_delta


def test_a_weaker_scheme_is_graded_down_with_the_bonds_that_broke():
    checked = check_equation("C1CCCC1 + 5 CC -> 5 CCC")
    assert checked.reaction_type == "Homodesmotic"
    assert not checked.atom_delta
    assert not checked.group_delta
    assert checked.bond_delta


def test_an_unbalanced_equation_is_called_unbalanced():
    checked = check_equation("C1CC1 -> CCC")
    assert checked.reaction_type == "Unbalanced"
    assert checked.atom_delta == {"H": 2}


def test_a_parse_failure_leaves_no_misleading_verdict():
    checked = check_equation("C1CC1 + 3 XYZ -> 3 CCCC")
    assert checked.errors
    assert checked.ok is False
    assert checked.atom_delta == {}
    assert checked.reaction_type == "Unbalanced"


def test_deltas_are_right_minus_left():
    checked = check_equation("CC -> CCC")
    assert checked.atom_delta["C"] == 1


def test_classify_sides_is_what_both_paths_use():
    """One classifier, so a paper's scheme and the solver's meet one standard."""
    left = [(1, CYCLOPENTANE), (5, "CCC")]
    right = [(5, "CCCC")]
    assert classify_sides(left, right) == "Hyperhomodesmotic"
    assert check_equation(REPORT_LINE).reaction_type == classify_sides(left, right)


def test_format_equation_sides_round_trips_through_the_parser():
    checked = check_equation(REPORT_LINE)
    rendered = format_equation_sides(checked.left, checked.right)
    again = check_equation(rendered)
    assert again.left == checked.left
    assert again.right == checked.right


def test_a_coefficient_of_one_is_not_printed():
    assert format_equation_sides([(1, "CC")], [(2, "C")]) == "CC -> 2 C"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_text_report_states_each_conservation_law():
    text = build_equation_check_text(check_equation("C1CCCC1 + 5 CC -> 5 CCC"))
    assert "Atoms: conserved" in text
    assert "Groups: conserved" in text
    assert "Bond types: NOT conserved" in text


def test_the_text_report_lists_the_parse_errors_instead_of_a_verdict():
    text = build_equation_check_text(check_equation("nonsense"))
    assert "No arrow found" in text
    assert "Reaction type" not in text


def test_both_reports_carry_the_check_it_yourself_note():
    checked = check_equation(REPORT_LINE)
    assert core.SELF_CHECK_NOTE in build_equation_check_text(checked)
    assert core.SELF_CHECK_NOTE in build_equation_check_html(checked)


def test_the_generated_draft_carries_the_note_too():
    result = analyze_molecule(Chem.MolFromSmiles(CYCLOPENTANE))
    assert core.SELF_CHECK_NOTE in result.equation_text
    assert core.SELF_CHECK_NOTE in result.equation_html


def test_the_html_report_escapes_what_it_shows():
    checked = check_equation("C1CC1 + 3 <script> -> 3 CCCC")
    assert "<script>" not in build_equation_check_html(checked)


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------


def _dialog():
    return ui.HomodesmoticAnalyzerDialog(_FakeContext())


def test_the_main_dialog_offers_the_tester():
    dialog = _dialog()
    assert dialog.test_button._text == "Test Equation..."


def test_the_tester_opens_on_the_current_draft():
    dialog = _dialog()
    assert dialog.current_equation_text() == "C1CCCC1 + 5 CCC -> 5 CCCC"


def test_the_draft_line_the_tester_starts_from_is_readable_by_it():
    dialog = _dialog()
    checked = check_equation(dialog.current_equation_text())
    assert checked.ok
    assert checked.reaction_type == dialog.last_result.reaction_type


def test_there_is_no_draft_line_without_a_molecule():
    dialog = ui.HomodesmoticAnalyzerDialog(_FakeContext(smiles=None))
    assert dialog.current_equation_text() == ""


def test_the_tester_checks_its_prefill_on_open():
    dialog = _dialog()
    tester = ui.TestEquationDialog(dialog, dialog.current_equation_text())
    assert tester.last_check.reaction_type == "Hyperhomodesmotic"


def test_the_tester_rechecks_what_was_typed():
    tester = ui.TestEquationDialog(None, "")
    tester.input_box.setPlainText("C1CCCC1 + 5 CC -> 5 CCC")
    tester.check()
    assert tester.last_check.reaction_type == "Homodesmotic"


def test_the_tester_shows_errors_rather_than_a_verdict():
    tester = ui.TestEquationDialog(None, "")
    tester.input_box.setPlainText("not an equation")
    tester.check()
    assert tester.last_check.errors
    assert "No arrow found" in tester.result_box.toHtml()


def test_enter_checks_the_equation():
    tester = ui.TestEquationDialog(None, "")
    assert tester.check_button.isDefault() is True
    assert tester.close_button.autoDefault() is False


# ---------------------------------------------------------------------------
# Exporting a check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".html", ".csv", ".txt"])
def test_a_check_exports_in_every_format(tmp_path, suffix):
    from strain_homodesmotic_reaction_generator.core import export_equation_check

    target = tmp_path / f"check{suffix}"
    export_equation_check(target, check_equation(REPORT_LINE), REPORT_LINE)
    text = target.read_text(encoding="utf-8")
    assert "Hyperhomodesmotic" in text


@pytest.mark.parametrize("suffix", [".html", ".csv", ".txt"])
def test_every_exported_check_names_the_version_that_made_it(tmp_path, suffix):
    from strain_homodesmotic_reaction_generator import PLUGIN_VERSION
    from strain_homodesmotic_reaction_generator.core import export_equation_check

    target = tmp_path / f"check{suffix}"
    export_equation_check(target, check_equation(REPORT_LINE), REPORT_LINE)
    assert PLUGIN_VERSION in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", [".html", ".csv", ".txt"])
def test_the_draft_report_names_its_version_too(tmp_path, suffix):
    from strain_homodesmotic_reaction_generator import PLUGIN_VERSION
    from strain_homodesmotic_reaction_generator.core import export_analysis

    target = tmp_path / f"report{suffix}"
    export_analysis(target, analyze_molecule(Chem.MolFromSmiles(CYCLOPENTANE)))
    assert PLUGIN_VERSION in target.read_text(encoding="utf-8")


def test_the_html_report_footer_carries_version_date_and_the_note(tmp_path):
    from strain_homodesmotic_reaction_generator import PLUGIN_VERSION
    from strain_homodesmotic_reaction_generator.core import export_analysis

    target = tmp_path / "report.html"
    export_analysis(target, analyze_molecule(Chem.MolFromSmiles(CYCLOPENTANE)))
    text = target.read_text(encoding="utf-8")
    footer = text[text.index('class="report-footer"') :]
    assert PLUGIN_VERSION in footer
    assert "generated 20" in footer
    assert core.SELF_CHECK_NOTE in footer


def test_an_exported_check_quotes_the_equation_as_typed(tmp_path):
    from strain_homodesmotic_reaction_generator.core import export_equation_check

    typed = "C1CCCC1 + 5 CCC (propane) -> 5 CCCC"
    target = tmp_path / "check.html"
    export_equation_check(target, check_equation(typed), typed)
    assert "(propane)" in target.read_text(encoding="utf-8")


def test_an_exported_check_falls_back_to_the_parsed_equation(tmp_path):
    from strain_homodesmotic_reaction_generator.core import export_equation_check

    target = tmp_path / "check.txt"
    export_equation_check(target, check_equation(REPORT_LINE))
    assert "C1CCCC1 + 5 CCC -> 5 CCCC" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", [".html", ".csv", ".txt"])
def test_a_failed_check_still_exports_its_errors(tmp_path, suffix):
    from strain_homodesmotic_reaction_generator.core import export_equation_check

    target = tmp_path / f"check{suffix}"
    export_equation_check(target, check_equation("nonsense"), "nonsense")
    assert "No arrow found" in target.read_text(encoding="utf-8")


def test_the_tester_offers_an_export_button():
    tester = ui.TestEquationDialog(None, "")
    assert tester.export_button._text == "Export Check"
    assert tester.export_button.autoDefault() is False


def test_the_tester_can_export_before_anything_was_checked(tmp_path):
    """Opening the tester with no draft leaves last_check unset otherwise."""
    import PyQt6.QtWidgets as qtw

    tester = ui.TestEquationDialog(None, "")
    target = tmp_path / "check.txt"
    qtw.QFileDialog._next_return = (str(target), "")
    try:
        tester.export()
    finally:
        qtw.QFileDialog._next_return = ("", "")
    assert target.read_text(encoding="utf-8")


def test_cancelling_the_export_writes_nothing(tmp_path):
    import PyQt6.QtWidgets as qtw

    tester = ui.TestEquationDialog(None, REPORT_LINE)
    qtw.QFileDialog._next_return = ("", "")
    tester.export()
    assert not list(tmp_path.iterdir())
