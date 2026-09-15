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
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGroupBox,
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
    QCheckBox = None  # type: ignore[assignment]
    QComboBox = None  # type: ignore[assignment]
    QDialog = None  # type: ignore[assignment]
    QFileDialog = None  # type: ignore[assignment]
    QGroupBox = None  # type: ignore[assignment]
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

#: Row kinds in the results table, in the order they are grouped by default.
_KIND_REFERENCE = 0
_KIND_LEFT = 1
_KIND_RIGHT = 2

#: Everything the user typed shares one colour, matching the exported report.
_USER_COLOR = "#ff8a65"

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
        """Small dialog asking which kind of entry to add, then its SMILES."""

        def __init__(self, parent: Any, environments: tuple[str, ...]) -> None:
            super().__init__(parent)
            self.setWindowTitle("Add Species")
            self.setModal(True)
            self.environments = tuple(environments)
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
            if self.environments:
                self.type_combo.addItem(_ADD_REFERENCE)
            type_row.addWidget(self.type_combo, 1)
            layout.addLayout(type_row)

            self.environment_row = QHBoxLayout()
            self.environment_label = self._field_label("Environment:")
            self.environment_row.addWidget(self.environment_label)
            self.environment_combo = QComboBox()
            for name in self.environments:
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
            self.add_button = QPushButton("Add")
            button_row.addWidget(self.cancel_button)
            button_row.addWidget(self.add_button)
            layout.addLayout(button_row)

            self.type_combo.currentTextChanged.connect(self._on_type_changed)
            self.cancel_button.clicked.connect(self.reject)
            self.add_button.clicked.connect(self.confirm)
            self._on_type_changed(self.type_combo.currentText())
            self.resize(460, 240)

        @staticmethod
        def _field_label(text: str) -> Any:
            label = QLabel(text)
            label.setMinimumWidth(86)
            return label

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
        """Qt dialog for the analyzer."""

        def __init__(self, context: Any) -> None:
            super().__init__(parent=context.get_main_window())
            self.context = context
            self.last_result = _empty_result()
            self.reference_overrides: dict[str, str] = {}
            self.user_species: list[UserSpecies] = []
            self._table_rows: list[dict] = []
            self._sort_state: tuple[int, bool] | None = None
            self._suspend_edits = False

            from . import PLUGIN_VERSION

            self.setWindowTitle(
                f"Strain Homodesmotic Reaction Generator (v{PLUGIN_VERSION})"
            )
            self.resize(920, 780)

            layout = QVBoxLayout(self)

            intro = QLabel(
                "Detected environments are mapped to small reference molecules. "
                "This table is the result; to change it, add a species or a "
                "reference override below. Every change re-analyzes at once and "
                "the reaction type is re-derived from the equation your choices "
                "produce."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.warning_label = QLabel("")
            self.warning_label.setStyleSheet("color: #b26b00; font-weight: bold;")
            self.warning_label.setVisible(False)
            layout.addWidget(self.warning_label)

            self.table = QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(
                ["Environment", "Count", "Reference SMILES", "Action"]
            )
            self.table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            self.table.horizontalHeader().setSectionResizeMode(
                2, QHeaderView.ResizeMode.ResizeToContents
            )
            self.table.horizontalHeader().setSectionsClickable(True)
            self.table.horizontalHeader().sectionClicked.connect(
                self._on_header_clicked
            )
            self.table.verticalHeader().setVisible(False)
            layout.addWidget(self.table, 1)

            layout.addWidget(self._build_user_group())

            self.equation_box = QTextEdit()
            self.equation_box.setReadOnly(True)
            self.equation_box.setStyleSheet(
                "background-color: #202124; color: #e8eaed; font-family: Consolas, monospace; font-size: 15px;"
            )
            self.equation_box.setMinimumHeight(190)
            layout.addWidget(self.equation_box)

            button_layout = QHBoxLayout()
            self.analyze_button = QPushButton("Analyze Current Molecule")
            self.export_button = QPushButton("Export Analysis")
            button_layout.addWidget(self.analyze_button)
            button_layout.addStretch()
            button_layout.addWidget(self.export_button)
            layout.addLayout(button_layout)

            self.analyze_button.clicked.connect(self.refresh_analysis)
            self.export_button.clicked.connect(self.export_analysis)

            self.refresh_analysis()

        # -- user-defined species box ---------------------------------------

        def _build_user_group(self) -> Any:
            group = QGroupBox("Your species and overrides")
            group_layout = QVBoxLayout()

            hint = QLabel(
                "Balance species extend the library the solver draws on; mark one "
                "Required to force it into the equation. Overrides replace the "
                "reference proposed for one environment. Reports quote these "
                "SMILES exactly as typed."
            )
            hint.setWordWrap(True)
            group_layout.addWidget(hint)

            self.user_table = QTableWidget(0, 4)
            self.user_table.setHorizontalHeaderLabels(
                ["Type", "Applies to / Name", "SMILES", "Required"]
            )
            self.user_table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch
            )
            self.user_table.verticalHeader().setVisible(False)
            self.user_table.setMaximumHeight(150)
            self.user_table.itemChanged.connect(self._on_user_item_changed)
            group_layout.addWidget(self.user_table)

            row = QHBoxLayout()
            self.add_species_button = QPushButton("Add Species...")
            self.remove_species_button = QPushButton("Remove Selected")
            self.reset_species_button = QPushButton("Reset to Defaults")
            row.addWidget(self.add_species_button)
            row.addWidget(self.remove_species_button)
            row.addStretch()
            row.addWidget(self.reset_species_button)
            group_layout.addLayout(row)

            self.add_species_button.clicked.connect(self.add_species)
            self.remove_species_button.clicked.connect(self.remove_selected_species)
            self.reset_species_button.clicked.connect(self.reset_species)

            group.setLayout(group_layout)
            self.user_group = group
            return group

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

        def apply_new_entry(self, entry: tuple) -> None:
            """Record an entry produced by AddSpeciesDialog and re-analyze."""
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
            self._populate_user_table()
            self.refresh_analysis()

        def remove_selected_species(self) -> None:
            rows = sorted(
                {index.row() for index in self.user_table.selectedIndexes()},
                reverse=True,
            )
            if not rows:
                _status(self.context, "Select a row to remove.", 3000)
                return
            entries = self._user_entries()
            for row in rows:
                if row >= len(entries):
                    continue
                kind, key, _smiles, _required = entries[row]
                if kind == "reference":
                    self.reference_overrides.pop(key, None)
                else:
                    self.user_species = [
                        species
                        for species in self.user_species
                        if species.smiles != key
                    ]
            self._populate_user_table()
            self.refresh_analysis()

        def reset_species(self) -> None:
            if not self.reference_overrides and not self.user_species:
                return
            self.reference_overrides = {}
            self.user_species = []
            self._populate_user_table()
            self.refresh_analysis()

        def _user_entries(self) -> list[tuple[str, str, str, bool]]:
            """Rows of the user table: (kind, key, smiles, required)."""
            entries: list[tuple[str, str, str, bool]] = [
                ("reference", name, smiles, False)
                for name, smiles in sorted(self.reference_overrides.items())
            ]
            entries.extend(
                ("balance", species.smiles, species.smiles, species.required)
                for species in self.user_species
            )
            return entries

        def _populate_user_table(self) -> None:
            self._suspend_edits = True
            try:
                entries = self._user_entries()
                self.user_table.setRowCount(len(entries))
                names = {species.smiles: species.name for species in self.user_species}
                for row, (kind, key, smiles, required) in enumerate(entries):
                    label = "Reference" if kind == "reference" else "Balance"
                    second = key if kind == "reference" else names.get(key, "")
                    self.user_table.setItem(row, 0, self._item(label, editable=False))
                    self.user_table.setItem(
                        row, 1, self._item(second, editable=kind == "balance")
                    )
                    self.user_table.setItem(row, 2, self._item(smiles, editable=True))
                    self.user_table.setItem(
                        row,
                        3,
                        self._item(
                            "-"
                            if kind == "reference"
                            else ("yes" if required else "no"),
                            editable=False,
                        ),
                    )
            finally:
                self._suspend_edits = False

        def _on_user_item_changed(self, item: Any) -> None:
            if self._suspend_edits:
                return
            entries = self._user_entries()
            row = item.row()
            if row >= len(entries):
                return
            kind, key, _smiles, required = entries[row]
            column = item.column()
            text = item.text().strip()

            if column == 2:
                if not is_valid_smiles(text):
                    QMessageBox.warning(
                        self,
                        "Invalid SMILES",
                        f"RDKit cannot parse '{item.text()}'.",
                    )
                    self._populate_user_table()
                    return
                if kind == "reference":
                    self.reference_overrides[key] = text
                else:
                    self.user_species = [
                        UserSpecies(text, species.name, species.required)
                        if species.smiles == key
                        else species
                        for species in self.user_species
                    ]
            elif column == 1 and kind == "balance":
                self.user_species = [
                    UserSpecies(species.smiles, text, species.required)
                    if species.smiles == key
                    else species
                    for species in self.user_species
                ]
            else:
                return

            self._populate_user_table()
            self.refresh_analysis()

        # -- analysis --------------------------------------------------------

        def refresh_analysis(self) -> None:
            if Chem is None:
                _status(self.context, "RDKit is not available.", 5000)
                return

            mol = _current_molecule(self.context)
            if mol is None or mol.GetNumAtoms() == 0:
                self.last_result = _empty_result()
                self._populate_table(self.last_result)
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
            self.warning_label.setText(
                "\n".join(f"⚠️ {message}" for message in messages)
            )
            self.warning_label.setVisible(bool(messages))

        def on_analysis_error(self, error_msg: str) -> None:
            self.analyze_button.setEnabled(True)
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"An error occurred during analysis:\n{error_msg}",
            )
            _status(self.context, "Analysis failed.", 5000)

        # -- results table ----------------------------------------------------

        def _on_header_clicked(self, column: int) -> None:
            """Cycle a column through ascending, descending, grouped default."""
            if column == 3:
                return
            if self._sort_state is None or self._sort_state[0] != column:
                self._sort_state = (column, False)
            elif not self._sort_state[1]:
                self._sort_state = (column, True)
            else:
                self._sort_state = None
            self._populate_table(self.last_result)

        def _result_rows(self, result: AnalysisResult) -> list[dict]:
            rows: list[dict] = []
            for match in result.matches:
                if match.count > 0:
                    rows.append(
                        {
                            "kind": _KIND_REFERENCE,
                            "label": f"Ref: {match.name}",
                            "environment": match.name,
                            "count": match.count,
                            "smiles": match.reference_smiles,
                            "tooltip": match.description,
                            "user": match.user_defined,
                        }
                    )
            for kind, terms, prefix, fallback in (
                (
                    _KIND_LEFT,
                    result.left_balance_terms,
                    "Left Balance",
                    "Added left-side balance species",
                ),
                (
                    _KIND_RIGHT,
                    result.right_balance_terms,
                    "Right Balance",
                    "Added right-side balance species",
                ),
            ):
                for term in terms:
                    if term.count > 0:
                        rows.append(
                            {
                                "kind": kind,
                                "label": f"{prefix}: {term.name}",
                                "environment": "",
                                "count": term.count,
                                "smiles": term.smiles,
                                "tooltip": term.description or fallback,
                                "user": term.user_defined,
                            }
                        )
            return rows

        def _sorted_rows(self, rows: list[dict]) -> list[dict]:
            if self._sort_state is None:
                return sorted(rows, key=lambda row: (row["kind"],))
            column, descending = self._sort_state
            if column == 1:
                key = lambda row: (row["count"], row["label"])  # noqa: E731
            elif column == 2:
                key = lambda row: (row["smiles"], row["label"])  # noqa: E731
            else:
                key = lambda row: (row["label"], row["count"])  # noqa: E731
            return sorted(rows, key=key, reverse=descending)

        def _populate_table(self, result: AnalysisResult) -> None:
            self._suspend_edits = True
            try:
                rows = self._sorted_rows(self._result_rows(result))
                self._table_rows = rows
                self.table.setRowCount(len(rows))

                brushes = {
                    _KIND_REFERENCE: self._brushes("#1a73e8"),
                    _KIND_LEFT: self._brushes("#1b7a2d"),
                    _KIND_RIGHT: self._brushes("#a142f4"),
                }

                for index, row in enumerate(rows):
                    fg, bg = brushes[row["kind"]]
                    if row["user"]:
                        fg, bg = self._brushes(_USER_COLOR)
                    self._set_colored_row(index, row, fg, bg)
                    self._add_load_button(index, row["smiles"], row["tooltip"])
            finally:
                self._suspend_edits = False

        @staticmethod
        def _brushes(color: str) -> tuple[Any, Any]:
            foreground = QColor(color)
            background = QColor(color)
            background.setAlpha(30)
            return QBrush(foreground), QBrush(background)

        def _item(self, text: str, editable: bool) -> Any:
            item = QTableWidgetItem(text)
            if not editable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        def _set_colored_row(
            self,
            row: int,
            data: dict,
            fg_brush: Any,
            bg_brush: Any,
        ) -> None:
            """Write one read-only result row.

            Nothing here is editable: this table reports what the solver
            produced, and an edit that silently changed nothing was the
            original complaint.
            """
            tooltip = (
                "Substitute this reference with Add Species -> Reference "
                "molecule override."
                if data["kind"] == _KIND_REFERENCE
                else "Balance species are chosen by the solver. Add your own "
                "below to change what it may use."
            )
            cells = (data["label"], str(data["count"]), data["smiles"])
            for col, text in enumerate(cells):
                item = self._item(text, editable=False)
                item.setForeground(fg_brush)
                item.setBackground(bg_brush)
                item.setToolTip(tooltip)
                self.table.setItem(row, col, item)

        def _add_load_button(self, row: int, smiles: str, tooltip: str) -> None:
            load_button = QPushButton("Load Species")
            load_button.setToolTip(tooltip)
            load_button.clicked.connect(
                lambda _checked=False, s=smiles: self.load_reference(s)
            )
            self.table.setCellWidget(row, 3, load_button)

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
