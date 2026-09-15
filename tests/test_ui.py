"""
Headless tests for ui.py (PyQt6 dialog / worker / menu wiring).

tests/conftest.py installs rich PyQt6 stubs before this module is imported,
so ``ui.QDialog is not None`` and the real dialog/worker classes defined in
``ui.py`` are exercised directly (no real Qt runtime, no GUI).
"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator import ui  # noqa: E402
from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    AnalysisResult,
    BalanceTerm,
    Chem,
    EnvironmentMatch,
)
from collections import Counter  # noqa: E402
import dataclasses  # noqa: E402


pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")


# ---------------------------------------------------------------------------
# Fake PluginContext
# ---------------------------------------------------------------------------


class _FakeContext:
    def __init__(self, molecule=None, load_from_smiles=False):
        self._molecule = molecule
        self.status_messages = []
        self.windows = {}
        self._loaded_smiles = []
        if load_from_smiles:
            self.load_from_smiles = self._load_from_smiles

    def _load_from_smiles(self, smiles):
        self._loaded_smiles.append(smiles)

    @property
    def current_molecule(self):
        return self._molecule

    def get_main_window(self):
        return None

    def show_status_message(self, message, timeout=3000):
        self.status_messages.append((message, timeout))

    def register_window(self, key, window):
        self.windows[key] = window

    def get_window(self, key):
        return self.windows.get(key)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_status_calls_show_status_message_when_present():
    ctx = _FakeContext()
    ui._status(ctx, "hello", 1234)
    assert ctx.status_messages == [("hello", 1234)]


def test_status_no_op_when_context_lacks_hook():
    class _Bare:
        pass

    # Must not raise.
    ui._status(_Bare(), "hello")


def test_current_molecule_reads_context_attribute():
    mol = Chem.MolFromSmiles("CC")
    ctx = _FakeContext(molecule=mol)
    assert ui._current_molecule(ctx) is mol


def test_load_smiles_with_host_true_when_supported():
    ctx = _FakeContext(load_from_smiles=True)
    assert ui._load_smiles_with_host(ctx, "CC") is True
    assert ctx._loaded_smiles == ["CC"]


def test_load_smiles_with_host_false_when_unsupported():
    ctx = _FakeContext()
    assert ui._load_smiles_with_host(ctx, "CC") is False


# ---------------------------------------------------------------------------
# Dialog: no molecule loaded
# ---------------------------------------------------------------------------


def test_dialog_refresh_analysis_no_molecule_loaded():
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    assert dialog.table.rowCount() == 0
    assert "No molecule is loaded" in dialog.equation_box.toHtml()
    assert dialog.warning_label.isVisible() is False
    assert ("No molecule is loaded.", 3000) in ctx.status_messages


def test_dialog_refresh_analysis_empty_molecule():
    mol = Chem.MolFromSmiles("")
    ctx = _FakeContext(molecule=mol)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    assert dialog.table.rowCount() == 0
    assert dialog.last_result.target_smiles == ""


def test_dialog_refresh_analysis_no_rdkit(monkeypatch):
    monkeypatch.setattr(ui, "Chem", None)
    ctx = _FakeContext(molecule=Chem.MolFromSmiles("CC"))
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    assert ("RDKit is not available.", 5000) in ctx.status_messages


# ---------------------------------------------------------------------------
# Dialog: worker success / error flow (QThread stub runs synchronously)
# ---------------------------------------------------------------------------


def test_dialog_refresh_analysis_success_populates_table():
    mol = Chem.MolFromSmiles("CC")
    ctx = _FakeContext(molecule=mol)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    assert dialog.table.rowCount() >= 1
    assert dialog.analyze_button.isEnabled() is True
    assert "CC" in dialog.equation_box.toHtml() or dialog.equation_box.toHtml() != ""
    assert any("Detected" in m for m, _t in ctx.status_messages)


def test_dialog_refresh_analysis_error_shows_message_box(monkeypatch):
    ui._QMessageBox_calls = []

    def _boom(mol, *args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(ui, "analyze_molecule", _boom)
    ctx = _FakeContext(molecule=Chem.MolFromSmiles("CC"))

    import PyQt6.QtWidgets as qtw

    qtw.QMessageBox.critical_calls.clear()
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    assert dialog.analyze_button.isEnabled() is True
    assert len(qtw.QMessageBox.critical_calls) == 1
    _parent, title, text = qtw.QMessageBox.critical_calls[0]
    assert title == "Analysis Failed"
    assert "kaboom" in text
    assert ("Analysis failed.", 5000) in ctx.status_messages


def test_on_analysis_success_elemental_balance_no_scipy(monkeypatch):
    monkeypatch.setattr(ui, "milp", None)
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    result = AnalysisResult(
        "CC",
        Counter({"C": 2}),
        Counter({"C": 2}),
        Counter(),
        (),
        (),
        Counter(),
        Counter(),
        (EnvironmentMatch("primary-primary", 1, "CC", "desc", (0, 1)),),
        "CC -> CC",
        "<p>CC -&gt; CC</p>",
        True,
        Counter(),
        Counter(),
        "Unbalanced",
        Counter(),
        Counter(),
    )
    dialog.on_analysis_success(result)

    assert dialog.warning_label.isVisible() is True
    assert "SciPy not found" in dialog.warning_label.text()


def test_on_analysis_success_elemental_balance_with_scipy(monkeypatch):
    monkeypatch.setattr(ui, "milp", lambda *a, **kw: None)
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    result = dataclasses.replace(dialog.last_result, is_elemental_balance=True)
    dialog.on_analysis_success(result)

    assert dialog.warning_label.isVisible() is True
    assert "No exact MILP balance possible" in dialog.warning_label.text()


# ---------------------------------------------------------------------------
# Table population: left/right balance terms and colored rows
# ---------------------------------------------------------------------------


def test_populate_table_includes_left_and_right_balance_terms():
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    result = AnalysisResult(
        "CC",
        Counter(),
        Counter(),
        Counter(),
        (BalanceTerm("left term", "C", 2, ""),),
        (BalanceTerm("right term", "N", 1, "custom desc"),),
        Counter(),
        Counter(),
        (EnvironmentMatch("match", 3, "CC", "desc", (0, 1)),),
        "text",
        "<p>html</p>",
        False,
        Counter(),
        Counter(),
        "Unbalanced",
        Counter(),
        Counter(),
    )

    dialog._populate_table(result)

    assert dialog.table.rowCount() == 3
    assert dialog.table.item(0, 0).text() == "Ref: match"
    assert dialog.table.item(1, 0).text() == "Left Balance: left term"
    assert dialog.table.item(2, 0).text() == "Right Balance: right term"
    # Load Species buttons were attached to every row.
    for row in range(3):
        assert dialog.table.cellWidget(row, 3) is not None


def test_populate_table_skips_zero_count_terms():
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    result = AnalysisResult(
        "CC",
        Counter(),
        Counter(),
        Counter(),
        (BalanceTerm("zero-left", "C", 0, ""),),
        (BalanceTerm("zero-right", "N", 0, ""),),
        Counter(),
        Counter(),
        (EnvironmentMatch("zero-match", 0, "CC", "desc", (0, 1)),),
        "text",
        "<p>html</p>",
        False,
        Counter(),
        Counter(),
        "Unbalanced",
        Counter(),
        Counter(),
    )

    dialog._populate_table(result)

    assert dialog.table.rowCount() == 0


# ---------------------------------------------------------------------------
# load_reference
# ---------------------------------------------------------------------------


def test_load_reference_uses_host_importer_when_available():
    ctx = _FakeContext(molecule=None, load_from_smiles=True)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    dialog.load_reference("CCO")

    assert ctx._loaded_smiles == ["CCO"]
    assert any("Loaded reference from SMILES: CCO" in m for m, _t in ctx.status_messages)


def test_load_reference_shows_information_box_when_unsupported():
    import PyQt6.QtWidgets as qtw

    qtw.QMessageBox.information_calls.clear()
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    dialog.load_reference("CCO")

    assert len(qtw.QMessageBox.information_calls) == 1
    assert any(
        "SMILES importer is not available." in m for m, _t in ctx.status_messages
    )


# ---------------------------------------------------------------------------
# export_analysis
# ---------------------------------------------------------------------------


def test_export_analysis_returns_early_when_no_path_chosen():
    import PyQt6.QtWidgets as qtw

    qtw.QFileDialog._next_return = ("", "")
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    before = len(ctx.status_messages)
    dialog.export_analysis()

    assert len(ctx.status_messages) == before


def test_export_analysis_success_writes_file(tmp_path):
    import PyQt6.QtWidgets as qtw

    out_file = tmp_path / "draft.html"
    qtw.QFileDialog._next_return = (str(out_file), "HTML Files (*.html)")
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    dialog.export_analysis()

    assert out_file.exists()
    assert any(
        f"Analysis exported to {out_file}" in m for m, _t in ctx.status_messages
    )


def test_export_analysis_failure_shows_message_box(tmp_path, monkeypatch):
    import PyQt6.QtWidgets as qtw

    out_file = tmp_path / "draft.html"
    qtw.QFileDialog._next_return = (str(out_file), "HTML Files (*.html)")
    qtw.QMessageBox.critical_calls.clear()

    def _boom(path, result, current_file_path=None):
        raise OSError("disk full")

    monkeypatch.setattr(ui, "export_analysis", _boom)
    ctx = _FakeContext(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    dialog.export_analysis()

    assert len(qtw.QMessageBox.critical_calls) == 1
    assert any("Export failed: disk full" in m for m, _t in ctx.status_messages)


def test_export_analysis_uses_current_file_path_from_main_window(tmp_path):
    import PyQt6.QtWidgets as qtw

    out_file = tmp_path / "draft.html"
    qtw.QFileDialog._next_return = (str(out_file), "HTML Files (*.html)")

    class _InitManager:
        current_file_path = "molecule.pmeprj"

    class _MainWindow:
        init_manager = _InitManager()

    class _CtxWithMainWindow(_FakeContext):
        def get_main_window(self):
            return _MainWindow()

    ctx = _CtxWithMainWindow(molecule=None)
    dialog = ui.HomodesmoticAnalyzerDialog(ctx)

    dialog.export_analysis()

    text = out_file.read_text(encoding="utf-8")
    assert "molecule.pmeprj" in text


# ---------------------------------------------------------------------------
# open_analyzer_dialog
# ---------------------------------------------------------------------------


def test_open_analyzer_dialog_reports_missing_pyqt6(monkeypatch):
    monkeypatch.setattr(ui, "QDialog", None)
    ctx = _FakeContext()

    ui.open_analyzer_dialog(ctx)

    assert ("PyQt6 is not available.", 5000) in ctx.status_messages


def test_open_analyzer_dialog_reports_missing_rdkit(monkeypatch):
    monkeypatch.setattr(ui, "Chem", None)
    ctx = _FakeContext()

    ui.open_analyzer_dialog(ctx)

    assert ("RDKit is not available.", 5000) in ctx.status_messages


def test_open_analyzer_dialog_reuses_existing_window():
    calls = []

    class _ExistingWindow:
        def show(self):
            calls.append("show")

        def raise_(self):
            calls.append("raise_")

        def activateWindow(self):
            calls.append("activateWindow")

        def refresh_analysis(self):
            calls.append("refresh_analysis")

    ctx = _FakeContext()
    ctx.windows[ui.WINDOW_ID] = _ExistingWindow()

    ui.open_analyzer_dialog(ctx)

    assert calls == ["show", "raise_", "activateWindow", "refresh_analysis"]


def test_open_analyzer_dialog_creates_and_registers_new_window():
    ctx = _FakeContext(molecule=None)

    ui.open_analyzer_dialog(ctx)

    assert ui.WINDOW_ID in ctx.windows
    assert isinstance(ctx.windows[ui.WINDOW_ID], ui.HomodesmoticAnalyzerDialog)
