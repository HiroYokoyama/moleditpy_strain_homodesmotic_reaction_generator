#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyQt6 user interface for Strain Homodesmotic Reaction Generator.
"""

from collections import Counter
from typing import Any

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
    )
    from PyQt6.QtGui import QColor, QBrush
except ImportError:  # pragma: no cover
    Qt = None  # type: ignore[assignment]
    QThread = None  # type: ignore[assignment]
    pyqtSignal = None  # type: ignore[assignment]
    QObject = None  # type: ignore[assignment]
    QAbstractItemView = None  # type: ignore[assignment]
    QCheckBox = None  # type: ignore[assignment]
    QComboBox = None  # type: ignore[assignment]
    QDialog = None  # type: ignore[assignment]
    QFileDialog = None  # type: ignore[assignment]
    QHeaderView = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QLineEdit = None  # type: ignore[assignment]
    QMessageBox = None  # type: ignore[assignment]
    QPushButton = None  # type: ignore[assignment]
    QTableWidget = None  # type: ignore[assignment]
    QTableWidgetItem = None  # type: ignore[assignment]
    QTextEdit = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]

from .core import (
    Chem,
    AnalysisResult,
    analyze_molecule,
    export_analysis,
    is_valid_smiles,
    milp,
)
from .data import ENVIRONMENTS, UserSpecies

WINDOW_ID = "strain_homodesmotic_reaction_generator"

#: Row roles, in the order the table groups them by default.
_ROLE_REFERENCE = 0
_ROLE_LEFT = 1
_ROLE_RIGHT = 2
#: Entries the user supplied that the equation does not contain.
_ROLE_UNUSED = 3

_ROLE_LABELS = {
    _ROLE_REFERENCE: "Reference",
    _ROLE_LEFT: "Left balance",
    _ROLE_RIGHT: "Right balance",
    _ROLE_UNUSED: "Not in equation",
}

_ROLE_COLORS = {
    _ROLE_REFERENCE: "#1a73e8",
    _ROLE_LEFT: "#1b7a2d",
    _ROLE_RIGHT: "#a142f4",
    _ROLE_UNUSED: "#9aa0a6",
}

#: Everything the user typed shares one colour, matching the exported report.
_USER_COLOR = "#ff8a65"

#: Analyze wears that colour while edits are waiting, so the button that
#: applies them is the one thing on screen that has changed.
_PENDING_BUTTON_STYLE = (
    f"background-color: {_USER_COLOR}; color: #202124; font-weight: bold;"
)

_SOURCE_YOURS = "yours"
_SOURCE_PENDING = "yours, pending"
_SOURCE_REMOVED = "removed, pending"
_SOURCE_NOT_USED = "yours, not used"
_SOURCE_UNPLACEABLE = "yours, cannot be placed"
_SOURCE_CANCELLED = "yours, cancelled out"
_SOURCE_NO_ENVIRONMENT = "yours, no such environment"

_COLUMNS = ("Role", "Environment / Species", "Count", "SMILES", "Source", "Action")
_COL_ROLE = 0
_COL_NAME = 1
_COL_COUNT = 2
_COL_SMILES = 3
_COL_SOURCE = 4
_COL_ACTION = 5

_ADD_BALANCE = "Balance species"
_ADD_REFERENCE = "Reference molecule override"


def _empty_result(message: str = "") -> AnalysisResult:
    text = message or "No molecule is loaded."
    return AnalysisResult(
        "",
        Counter(),
        Counter(),
        Counter(),
        (),
        (),
        Counter(),
        Counter(),
        (),
        text,
        f"<p>{text}</p>",
        False,
        Counter(),
        Counter(),
        "Unbalanced",
        Counter(),
        Counter(),
    )


def _status(context: Any, message: str, timeout: int = 3000) -> None:
    if hasattr(context, "show_status_message"):
        context.show_status_message(message, timeout)


def _current_molecule(context: Any) -> Any:
    return context.current_molecule


def _default_reference(environment: str) -> str:
    """The built-in reference molecule for an environment, "" if unknown."""
    for rule in ENVIRONMENTS:
        if rule.name == environment:
            return rule.reference_smiles
    return ""


def _load_smiles_with_host(context: Any, smiles: str) -> bool:
    """Load a SMILES string through MoleditPy's native importer when available."""
    if not hasattr(context, "load_from_smiles"):
        return False
    context.load_from_smiles(smiles)
    return True


if QDialog is not None:

    class AnalysisWorker(QObject):
        """Worker to run molecule analysis in a background thread."""

        finished = pyqtSignal(object)
        error = pyqtSignal(str)

        def __init__(
            self,
            mol: Any,
            reference_overrides: dict[str, str] | None = None,
            user_species: tuple[UserSpecies, ...] = (),
        ) -> None:
            super().__init__()
            self.mol = mol
            self.reference_overrides = dict(reference_overrides or {})
            self.user_species = tuple(user_species)

        def run(self) -> None:
            try:
                result = analyze_molecule(
                    self.mol, self.reference_overrides, self.user_species
                )
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    class LoadingDialog(QDialog):
        """Modal loading dialog shown during background analysis."""

        def __init__(self, parent: Any) -> None:
            super().__init__(parent)
            self.setWindowTitle("Analyzing...")
            self.setModal(True)
            self.setWindowFlags(
                self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
            )
            layout = QVBoxLayout(self)
            label = QLabel(
                "Analyzing molecule & calculating homodesmotic balance...\nPlease wait."
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.resize(320, 100)

    class AddSpeciesDialog(QDialog):
        """Small dialog asking which kind of entry to add, then its SMILES.

        Passing *existing* turns it into the edit dialog for that entry: the
        fields start filled in and the type is fixed, since changing a balance
        species into a reference override is really two separate edits.
        """

        def __init__(
            self,
            parent: Any,
            environments: tuple[str, ...],
            existing: tuple | None = None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Edit Species" if existing else "Add Species")
            self.setModal(True)
            self.environments = tuple(environments)
            self.existing = existing
            self.entry: tuple | None = None

            layout = QVBoxLayout(self)

            hint = QLabel(
                "A balance species joins the library the solver may draw on. "
                "A reference override replaces the molecule proposed for one "
                "detected environment."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

            type_row = QHBoxLayout()
            type_row.addWidget(self._field_label("Type:"))
            self.type_combo = QComboBox()
            self.type_combo.addItem(_ADD_BALANCE)
            if self.environments or self._existing_kind() == "reference":
                self.type_combo.addItem(_ADD_REFERENCE)
            type_row.addWidget(self.type_combo, 1)
            layout.addLayout(type_row)

            self.environment_row = QHBoxLayout()
            self.environment_label = self._field_label("Environment:")
            self.environment_row.addWidget(self.environment_label)
            self.environment_combo = QComboBox()
            for name in self._environment_choices():
                self.environment_combo.addItem(name)
            self.environment_row.addWidget(self.environment_combo, 1)
            layout.addLayout(self.environment_row)

            smiles_row = QHBoxLayout()
            smiles_row.addWidget(self._field_label("SMILES:"))
            self.smiles_edit = QLineEdit()
            self.smiles_edit.setPlaceholderText("e.g. CCCCCC")
            smiles_row.addWidget(self.smiles_edit, 1)
            layout.addLayout(smiles_row)

            name_row = QHBoxLayout()
            name_row.addWidget(self._field_label("Name:"))
            self.name_edit = QLineEdit()
            self.name_edit.setPlaceholderText("optional label for the report")
            name_row.addWidget(self.name_edit, 1)
            layout.addLayout(name_row)

            self.required_check = QCheckBox(
                "Required (the balance must actually use it)"
            )
            layout.addWidget(self.required_check)

            self.error_label = QLabel("")
            self.error_label.setStyleSheet("color: #f28b82; font-weight: bold;")
            self.error_label.setVisible(False)
            layout.addWidget(self.error_label)

            button_row = QHBoxLayout()
            button_row.addStretch()
            self.cancel_button = QPushButton("Cancel")
            self.add_button = QPushButton("Save" if existing else "Add")
            # Enter confirms: the whole dialog is one SMILES and a couple of
            # options, so reaching for the mouse to commit it is friction.
            self.add_button.setDefault(True)
            self.add_button.setAutoDefault(True)
            self.cancel_button.setAutoDefault(False)
            button_row.addWidget(self.cancel_button)
            button_row.addWidget(self.add_button)
            layout.addLayout(button_row)

            self.type_combo.currentTextChanged.connect(self._on_type_changed)
            self.cancel_button.clicked.connect(self.reject)
            self.add_button.clicked.connect(self.confirm)
            self._prefill()
            self._on_type_changed(self.type_combo.currentText())
            self.smiles_edit.setFocus()
            self.resize(460, 240)

        @staticmethod
        def _field_label(text: str) -> Any:
            label = QLabel(text)
            label.setMinimumWidth(86)
            return label

        def _existing_kind(self) -> str:
            return self.existing[0] if self.existing else ""

        def _environment_choices(self) -> tuple[str, ...]:
            """Detected environments, plus the one being edited if it dropped out."""
            names = list(self.environments)
            if self._existing_kind() == "reference" and self.existing[1] not in names:
                names.insert(0, self.existing[1])
            return tuple(names)

        def _prefill(self) -> None:
            if not self.existing:
                return
            if self.existing[0] == "reference":
                _kind, name, smiles = self.existing
                self.type_combo.setCurrentText(_ADD_REFERENCE)
                self.environment_combo.setCurrentText(name)
                self.smiles_edit.setText(smiles)
            else:
                species = self.existing[1]
                self.type_combo.setCurrentText(_ADD_BALANCE)
                self.smiles_edit.setText(species.smiles)
                self.name_edit.setText(species.name)
                self.required_check.setChecked(species.required)
            self.type_combo.setEnabled(False)

        def _on_type_changed(self, text: str) -> None:
            is_reference = text == _ADD_REFERENCE
            self.environment_label.setVisible(is_reference)
            self.environment_combo.setVisible(is_reference)
            self.name_edit.setVisible(not is_reference)
            self.required_check.setVisible(not is_reference)

        def confirm(self) -> None:
            smiles = self.smiles_edit.text().strip()
            if not is_valid_smiles(smiles):
                self.error_label.setText(
                    "RDKit cannot parse that SMILES." if smiles else "Enter a SMILES."
                )
                self.error_label.setVisible(True)
                return
            if self.type_combo.currentText() == _ADD_REFERENCE:
                self.entry = ("reference", self.environment_combo.currentText(), smiles)
            else:
                self.entry = (
                    "balance",
                    UserSpecies(
                        smiles,
                        self.name_edit.text().strip(),
                        self.required_check.isChecked(),
                    ),
                )
            self.accept()

    class HomodesmoticAnalyzerDialog(QDialog):
        """Qt dialog for the analyzer.

        One table holds the whole draft: the reference molecules, the balance
        species the solver chose, and anything the user supplied that did not
        make it into the equation. An entry the analysis could not honour still
        needs a row, or it would vanish with no way to edit or remove it.
        """

        def __init__(self, context: Any) -> None:
            super().__init__(parent=context.get_main_window())
            self.context = context
            self.last_result = _empty_result()
            self.reference_overrides: dict[str, str] = {}
            self.user_species: list[UserSpecies] = []
            self._table_rows: list[dict] = []
            self._sort_state: tuple[int, bool] | None = None

            from . import PLUGIN_VERSION

            self.setWindowTitle(
                f"Strain Homodesmotic Reaction Generator (v{PLUGIN_VERSION})"
            )
            self.resize(960, 900)

            layout = QVBoxLayout(self)

            intro = QLabel(
                "Detected environments are mapped to small reference molecules "
                "and balanced automatically. Double-click a row to substitute "
                "its reference, or Add your own balance species, then "
                "press Analyze: the reaction type is re-derived from whatever "
                "equation your choices produce."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.pending_label = QLabel("")
            self.pending_label.setStyleSheet(
                f"color: {_USER_COLOR}; font-weight: bold;"
            )
            self.pending_label.setVisible(False)
            layout.addWidget(self.pending_label)

            self.warning_label = QLabel("")
            self.warning_label.setStyleSheet("color: #b26b00; font-weight: bold;")
            self.warning_label.setVisible(False)
            layout.addWidget(self.warning_label)

            self.table = QTableWidget(0, len(_COLUMNS))
            self.table.setHorizontalHeaderLabels(list(_COLUMNS))
            self.table.horizontalHeader().setSectionResizeMode(
                _COL_NAME, QHeaderView.ResizeMode.Stretch
            )
            self.table.horizontalHeader().setSectionResizeMode(
                _COL_SMILES, QHeaderView.ResizeMode.ResizeToContents
            )
            self.table.horizontalHeader().setSectionsClickable(True)
            self.table.horizontalHeader().sectionClicked.connect(
                self._on_header_clicked
            )
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
            # A table sized by its stretch share alone collapses to a few rows,
            # so the floor is what actually sets its height; the cap only stops
            # it eating the report on a tall screen.
            self.table.setMinimumHeight(340)
            self.table.setMaximumHeight(520)
            layout.addWidget(self.table, 1)

            button_row = QHBoxLayout()
            self.add_species_button = QPushButton("Add Species...")
            self.edit_species_button = QPushButton("Edit Selected...")
            self.remove_species_button = QPushButton("Remove Selected")
            self.reset_species_button = QPushButton("Reset to Defaults")
            self.analyze_button = QPushButton("Analyze")
            self.export_button = QPushButton("Export Analysis")
            button_row.addWidget(self.add_species_button)
            button_row.addWidget(self.edit_species_button)
            button_row.addWidget(self.remove_species_button)
            button_row.addWidget(self.reset_species_button)
            button_row.addStretch()
            button_row.addWidget(self.analyze_button)
            button_row.addWidget(self.export_button)
            layout.addLayout(button_row)

            self.add_species_button.clicked.connect(self.add_species)
            self.edit_species_button.clicked.connect(self.edit_selected_species)
            self.remove_species_button.clicked.connect(self.remove_selected_species)
            self.reset_species_button.clicked.connect(self.reset_species)
            self.analyze_button.clicked.connect(self.refresh_analysis)
            self.export_button.clicked.connect(self.export_analysis)

            self.equation_box = QTextEdit()
            self.equation_box.setReadOnly(True)
            self.equation_box.setStyleSheet(
                "background-color: #202124; color: #e8eaed; "
                "font-family: Consolas, monospace; font-size: 15px;"
            )
            self.equation_box.setMinimumHeight(320)
            layout.addWidget(self.equation_box, 3)

            self.refresh_analysis()

        # -- user entries ------------------------------------------------------

        def _environment_names(self) -> tuple[str, ...]:
            """Detected environments first, falling back to the full rule list."""
            detected = tuple(
                match.name for match in self.last_result.matches if match.count > 0
            )
            return detected or tuple(rule.name for rule in ENVIRONMENTS)

        def add_species(self) -> None:
            dialog = AddSpeciesDialog(self, self._environment_names())
            dialog.exec()
            if dialog.entry is None:
                return
            self.apply_new_entry(dialog.entry)

        def _on_row_double_clicked(self, item: Any) -> None:
            """Double-clicking a row edits it; the cells themselves never do."""
            self.table.selectRow(item.row())
            self.edit_selected_species()

        def edit_selected_species(self) -> None:
            row = self._selected_row()
            if row is None:
                _status(self.context, "Select one row to edit.", 3000)
                return
            existing = row["edit_entry"]
            if existing is None:
                _status(
                    self.context,
                    "Only reference molecules and your own species can be edited.",
                    4000,
                )
                return
            dialog = AddSpeciesDialog(self, self._environment_names(), existing)
            dialog.exec()
            if dialog.entry is None:
                return
            self.apply_new_entry(dialog.entry, replacing=existing)

        def remove_selected_species(self) -> None:
            row = self._selected_row()
            if row is None:
                _status(self.context, "Select one row to remove.", 3000)
                return
            if not row["removable"]:
                _status(
                    self.context,
                    "That row is part of the generated draft; only your own "
                    "entries can be removed.",
                    4000,
                )
                return
            self._forget_entry(row["edit_entry"])
            self._stage()

        def reset_species(self) -> None:
            if not self.reference_overrides and not self.user_species:
                return
            self.reference_overrides = {}
            self.user_species = []
            self._stage()

        def apply_new_entry(self, entry: tuple, replacing: tuple | None = None) -> None:
            """Record an entry produced by AddSpeciesDialog and re-analyze.

            *replacing* is the entry an edit started from; dropping it first is
            what lets an edit change the very field the entry is keyed on.
            """
            if replacing is not None:
                self._forget_entry(replacing)
            if entry[0] == "reference":
                _kind, name, smiles = entry
                self.reference_overrides[name] = smiles
            else:
                species = entry[1]
                self.user_species = [
                    existing
                    for existing in self.user_species
                    if existing.smiles != species.smiles
                ]
                self.user_species.append(species)
            self._stage()

        def _stage(self) -> None:
            """Show the change without running the analysis.

            A full re-analysis takes seconds on a large molecule, so doing one
            per edit would put a modal wait between the user and every
            keystroke. Changes accumulate and Analyze applies them.
            """
            self._populate_table(self.last_result)
            self._update_pending()

        def _pending_changes(self) -> int:
            """How many entries differ from the ones the last analysis used.

            Counted against the result rather than against the table, because
            an entry the equation never contained has no row to mark: removing
            an unused species would otherwise look like nothing had changed.
            """
            analyzed_overrides = dict(self.last_result.reference_overrides)
            changed = set(analyzed_overrides) ^ set(self.reference_overrides)
            changed |= {
                name
                for name, smiles in self.reference_overrides.items()
                if analyzed_overrides.get(name) not in (None, smiles)
            }

            def key(species: Any) -> tuple:
                return (species.smiles, species.name, species.required)

            analyzed = {key(s) for s in self.last_result.user_species}
            current = {key(s) for s in self.user_species}
            return len(changed) + len({entry[0] for entry in analyzed ^ current})

        def _update_pending(self) -> None:
            pending = self._pending_changes()
            if pending:
                self.pending_label.setText(
                    f"{pending} change{'' if pending == 1 else 's'} pending - "
                    "press Analyze to apply."
                )
            self.pending_label.setVisible(bool(pending))
            self.analyze_button.setStyleSheet(_PENDING_BUTTON_STYLE if pending else "")

        def _forget_entry(self, entry: tuple | None) -> None:
            if entry is None:
                return
            if entry[0] == "reference":
                self.reference_overrides.pop(entry[1], None)
            else:
                self.user_species = [
                    species
                    for species in self.user_species
                    if species.smiles != entry[1].smiles
                ]

        def _selected_row(self) -> dict | None:
            rows = {index.row() for index in self.table.selectedIndexes()}
            if len(rows) != 1:
                return None
            row = rows.pop()
            if row >= len(self._table_rows):
                return None
            return self._table_rows[row]

        def _species_for(self, smiles: str) -> Any:
            return next(
                (item for item in self.user_species if item.smiles == smiles), None
            )

        # -- analysis ----------------------------------------------------------

        def refresh_analysis(self) -> None:
            if Chem is None:
                _status(self.context, "RDKit is not available.", 5000)
                return

            mol = _current_molecule(self.context)
            if mol is None or mol.GetNumAtoms() == 0:
                self.last_result = _empty_result()
                self._populate_table(self.last_result)
                self._update_pending()
                self.equation_box.setHtml(self.last_result.equation_html)
                self.warning_label.setVisible(False)
                _status(self.context, "No molecule is loaded.", 3000)
                return

            self.analyze_button.setEnabled(False)
            self.loading_dialog = LoadingDialog(self)

            self.analysis_thread = QThread()
            self.worker = AnalysisWorker(
                mol, dict(self.reference_overrides), tuple(self.user_species)
            )
            self.worker.moveToThread(self.analysis_thread)

            self.analysis_thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_analysis_success)
            self.worker.error.connect(self.on_analysis_error)

            self.worker.finished.connect(self.analysis_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.error.connect(self.analysis_thread.quit)
            self.worker.error.connect(self.worker.deleteLater)
            self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

            self.worker.finished.connect(self.loading_dialog.accept)
            self.worker.error.connect(self.loading_dialog.reject)

            self.analysis_thread.start()
            self.loading_dialog.exec()

        def on_analysis_success(self, result: AnalysisResult) -> None:
            self.analyze_button.setEnabled(True)
            self.last_result = result
            self._populate_table(self.last_result)
            self._update_pending()
            self.equation_box.setHtml(self.last_result.equation_html)
            self._update_warning(result)
            _status(
                self.context,
                f"Detected {len(self.last_result.matches)} environment types.",
                3000,
            )

        def _update_warning(self, result: AnalysisResult) -> None:
            messages = []
            if result.cancelled_references:
                messages.append(
                    "The reference for "
                    + ", ".join(result.cancelled_references)
                    + " cancelled out, so the equation no longer covers that "
                    "environment."
                )
            if result.unmet_required:
                messages.append(
                    "No balance exists that uses "
                    + ", ".join(result.unmet_required)
                    + "; it was left out."
                )
            if result.is_elemental_balance:
                if milp is None:
                    messages.append(
                        "SciPy not found - install scipy for exact MILP balance; "
                        "operating in elemental balance mode."
                    )
                else:
                    messages.append(
                        "No exact MILP balance possible for this molecule; "
                        "operating in elemental balance mode."
                    )
            self.warning_label.setText("\n".join(f"⚠️ {m}" for m in messages))
            self.warning_label.setVisible(bool(messages))

        def on_analysis_error(self, error_msg: str) -> None:
            self.analyze_button.setEnabled(True)
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"An error occurred during analysis:\n{error_msg}",
            )
            _status(self.context, "Analysis failed.", 5000)

        # -- the table ---------------------------------------------------------

        def _on_header_clicked(self, column: int) -> None:
            """Cycle a column through ascending, descending, grouped default."""
            if column == _COL_ACTION:
                return
            if self._sort_state is None or self._sort_state[0] != column:
                self._sort_state = (column, False)
            elif not self._sort_state[1]:
                self._sort_state = (column, True)
            else:
                self._sort_state = None
            self._populate_table(self.last_result)

        def _result_rows(self, result: AnalysisResult) -> list[dict]:
            """Every equation term, then every user entry that missed the cut.

            Rows show the entries as they stand now, which is not always what
            the last analysis used: an edit made since then is marked pending
            rather than silently displayed as though it had taken effect.
            """
            rows: list[dict] = []
            used_smiles: set[str] = set()
            analyzed_species = {species.smiles for species in result.user_species}

            for match in result.matches:
                if match.count == 0:
                    continue
                current = self.reference_overrides.get(
                    match.name, _default_reference(match.name)
                )
                overridden = match.name in self.reference_overrides
                if current != match.reference_smiles:
                    source = _SOURCE_PENDING if overridden else _SOURCE_REMOVED
                else:
                    source = _SOURCE_YOURS if match.user_defined else "default"
                rows.append(
                    {
                        "role": _ROLE_REFERENCE,
                        "name": match.name,
                        "count": match.count,
                        "smiles": current,
                        "source": source,
                        "tooltip": match.description,
                        "user": overridden or match.user_defined,
                        "edit_entry": ("reference", match.name, current),
                        "removable": overridden,
                    }
                )

            for role, terms, fallback in (
                (
                    _ROLE_LEFT,
                    result.left_balance_terms,
                    "Added left-side balance species",
                ),
                (
                    _ROLE_RIGHT,
                    result.right_balance_terms,
                    "Added right-side balance species",
                ),
            ):
                for term in terms:
                    if term.count == 0:
                        continue
                    species = (
                        self._species_for(term.smiles) if term.user_defined else None
                    )
                    if species is not None:
                        used_smiles.add(species.smiles)
                    if not term.user_defined:
                        source = "default"
                    elif species is None:
                        # Dropped since the analysis that placed it here.
                        source = _SOURCE_REMOVED
                    else:
                        source = _SOURCE_YOURS
                    rows.append(
                        {
                            "role": role,
                            "name": term.name,
                            "count": term.count,
                            "smiles": term.smiles,
                            "source": source,
                            "tooltip": term.description or fallback,
                            "user": term.user_defined,
                            "edit_entry": (
                                ("balance", species) if species is not None else None
                            ),
                            "removable": species is not None,
                        }
                    )

            shown_environments = {
                row["name"] for row in rows if row["role"] == _ROLE_REFERENCE
            }
            rows.extend(
                self._orphan_rows(
                    result, used_smiles, shown_environments, analyzed_species
                )
            )
            return rows

        def _orphan_rows(
            self,
            result: AnalysisResult,
            used_smiles: set[str],
            shown_environments: set[str],
            analyzed_species: set[str],
        ) -> list[dict]:
            """Rows for user entries the equation does not contain.

            Without these a species the solver declined, a requirement it could
            not meet, or a reference that cancelled itself away would disappear
            from the table with no way to edit or remove it.
            """
            rows: list[dict] = []
            unmet = set(result.unmet_required)
            cancelled = set(result.cancelled_references)
            analyzed_overrides = dict(result.reference_overrides)

            for name, smiles in sorted(self.reference_overrides.items()):
                if name in shown_environments:
                    continue
                if analyzed_overrides.get(name) != smiles:
                    source = _SOURCE_PENDING
                elif name in cancelled:
                    source = _SOURCE_CANCELLED
                else:
                    source = _SOURCE_NO_ENVIRONMENT
                rows.append(
                    {
                        "role": _ROLE_UNUSED,
                        "name": name,
                        "count": None,
                        "smiles": smiles,
                        "source": source,
                        "tooltip": (
                            "This override cancelled against itself, so the "
                            "equation no longer covers the environment."
                            if name in cancelled
                            else "No atom in this molecule matches that "
                            "environment, so the override has no effect."
                        ),
                        "user": True,
                        "edit_entry": ("reference", name, smiles),
                        "removable": True,
                    }
                )

            for species in self.user_species:
                if species.smiles in used_smiles:
                    continue
                if species.smiles not in analyzed_species:
                    source = _SOURCE_PENDING
                elif species.smiles in unmet:
                    source = _SOURCE_UNPLACEABLE
                else:
                    source = _SOURCE_NOT_USED
                rows.append(
                    {
                        "role": _ROLE_UNUSED,
                        "name": species.name or species.smiles,
                        "count": None,
                        "smiles": species.smiles,
                        "source": source,
                        "tooltip": (
                            "Not analyzed yet; press Analyze to apply it."
                            if source == _SOURCE_PENDING
                            else "Required, but no balance exists that uses it."
                            if source == _SOURCE_UNPLACEABLE
                            else "Available to the solver, which did not need "
                            "it. Mark it Required to force it in."
                        ),
                        "user": True,
                        "edit_entry": ("balance", species),
                        "removable": True,
                    }
                )
            return rows

        def _sorted_rows(self, rows: list[dict]) -> list[dict]:
            if self._sort_state is None:
                return sorted(rows, key=lambda row: (row["role"], row["name"]))
            column, descending = self._sort_state
            field = {
                _COL_ROLE: "role",
                _COL_NAME: "name",
                _COL_SMILES: "smiles",
                _COL_SOURCE: "source",
            }.get(column)
            if column == _COL_COUNT:
                # An entry with no place in the equation has no count; park it
                # at one end instead of letting None break the comparison.
                return sorted(
                    rows,
                    key=lambda row: (
                        row["count"] if row["count"] is not None else -1,
                        row["name"],
                    ),
                    reverse=descending,
                )
            return sorted(
                rows, key=lambda row: (row[field], row["name"]), reverse=descending
            )

        def _populate_table(self, result: AnalysisResult) -> None:
            rows = self._sorted_rows(self._result_rows(result))
            self._table_rows = rows
            # Drops the previous rows' Load buttons. Without it the replaced
            # widgets outlive their cells and paint over the new ones.
            self.table.clearContents()
            self.table.setRowCount(len(rows))

            for index, row in enumerate(rows):
                color = _USER_COLOR if row["user"] else _ROLE_COLORS[row["role"]]
                fg, bg = self._brushes(color)
                self._set_colored_row(index, row, fg, bg)
                self._add_load_button(index, row["smiles"], row["tooltip"])

        @staticmethod
        def _brushes(color: str) -> tuple[Any, Any]:
            foreground = QColor(color)
            background = QColor(color)
            background.setAlpha(30)
            return QBrush(foreground), QBrush(background)

        def _item(self, text: str, editable: bool = False) -> Any:
            item = QTableWidgetItem(text)
            if not editable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        def _set_colored_row(
            self, row: int, data: dict, fg_brush: Any, bg_brush: Any
        ) -> None:
            """Write one read-only row.

            Nothing here is editable: an edit that silently changed nothing was
            the original complaint. Add and Edit are the way in.
            """
            cells = (
                _ROLE_LABELS[data["role"]],
                data["name"],
                "-" if data["count"] is None else str(data["count"]),
                data["smiles"],
                data["source"],
            )
            for col, text in enumerate(cells):
                item = self._item(text)
                item.setForeground(fg_brush)
                item.setBackground(bg_brush)
                item.setToolTip(data["tooltip"])
                self.table.setItem(row, col, item)

        def _add_load_button(self, row: int, smiles: str, tooltip: str) -> None:
            load_button = QPushButton("Load Species")
            load_button.setToolTip(tooltip)
            load_button.clicked.connect(
                lambda _checked=False, s=smiles: self.load_reference(s)
            )
            self.table.setCellWidget(row, _COL_ACTION, load_button)

        def load_reference(self, smiles: str) -> None:
            if _load_smiles_with_host(self.context, smiles):
                _status(self.context, f"Loaded reference from SMILES: {smiles}", 3000)
                return

            QMessageBox.information(
                self,
                "SMILES Importer Unavailable",
                "The MoleditPy SMILES importer is not available in this context.",
            )
            _status(self.context, "SMILES importer is not available.", 5000)

        def export_analysis(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Homodesmotic Reaction Draft",
                "strain_homodesmotic_reaction_draft.html",
                "HTML Files (*.html);;CSV Files (*.csv);;Text Files (*.txt)",
            )
            if not path:
                return

            try:
                main_window = (
                    self.context.get_main_window()
                    if hasattr(self.context, "get_main_window")
                    else None
                )
                current_file_path = (
                    getattr(main_window.init_manager, "current_file_path", None)
                    if hasattr(main_window, "init_manager")
                    else None
                )
                export_analysis(path, self.last_result, current_file_path)
            except OSError as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))
                _status(self.context, f"Export failed: {exc}", 5000)
                return

            _status(self.context, f"Analysis exported to {path}", 3000)


def open_analyzer_dialog(context: Any) -> None:
    if QDialog is None:
        _status(context, "PyQt6 is not available.", 5000)
        return
    if Chem is None:
        _status(context, "RDKit is not available.", 5000)
        return

    window = context.get_window(WINDOW_ID)
    if window:
        window.show()
        window.raise_()
        window.activateWindow()
        if hasattr(window, "refresh_analysis"):
            window.refresh_analysis()
        return

    dialog = HomodesmoticAnalyzerDialog(context)
    context.register_window(WINDOW_ID, dialog)
    dialog.show()
