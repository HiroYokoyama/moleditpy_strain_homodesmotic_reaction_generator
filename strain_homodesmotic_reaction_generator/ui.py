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
        QDialog,
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
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
    QDialog = None  # type: ignore[assignment]
    QFileDialog = None  # type: ignore[assignment]
    QHeaderView = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
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
)

WINDOW_ID = "strain_homodesmotic_reaction_generator"


def _status(context: Any, message: str, timeout: int = 3000) -> None:
    if hasattr(context, "show_status_message"):
        context.show_status_message(message, timeout)


def _current_molecule(context: Any) -> Any:
    main_window = context.get_main_window() if hasattr(context, "get_main_window") else None
    state_manager = getattr(main_window, "state_manager", None)
    data = getattr(state_manager, "data", None)
    if hasattr(data, "to_rdkit_mol"):
        try:
            mol = data.to_rdkit_mol()
            if mol is not None:
                return mol
        except Exception:
            pass
    return getattr(context, "current_molecule", None)


def _load_smiles_with_host(context: Any, smiles: str) -> bool:
    """Load a SMILES string through MoleditPy's native importer when available."""
    main_window = context.get_main_window() if hasattr(context, "get_main_window") else None
    importer = getattr(main_window, "string_importer_manager", None)
    load_from_smiles = getattr(importer, "load_from_smiles", None)
    if not callable(load_from_smiles):
        return False

    load_from_smiles(smiles)
    return True


if QDialog is not None:

    class AnalysisWorker(QObject):
        """Worker to run molecule analysis in a background thread."""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)

        def __init__(self, mol: Any) -> None:
            super().__init__()
            self.mol = mol

        def run(self) -> None:
            try:
                result = analyze_molecule(self.mol)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))


    class LoadingDialog(QDialog):
        """Modal loading dialog shown during background analysis."""
        def __init__(self, parent: Any) -> None:
            super().__init__(parent)
            self.setWindowTitle("Analyzing...")
            self.setModal(True)
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
            layout = QVBoxLayout(self)
            label = QLabel("Analyzing molecule & calculating homodesmotic balance...\nPlease wait.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.resize(320, 100)


    class HomodesmoticAnalyzerDialog(QDialog):
        """Qt dialog for the analyzer."""

        def __init__(self, context: Any) -> None:
            super().__init__(parent=context.get_main_window())
            self.context = context
            self.last_result = AnalysisResult(
                "",
                Counter(),
                Counter(),
                Counter(),
                (),
                (),
                Counter(),
                Counter(),
                (),
                "",
                "",
                False,
                Counter(),
                Counter(),
                "Unbalanced",
                Counter(),
                Counter(),
            )

            from . import PLUGIN_VERSION
            self.setWindowTitle(f"Strain Homodesmotic Reaction Generator (v{PLUGIN_VERSION})")
            self.resize(820, 640)

            layout = QVBoxLayout(self)

            intro = QLabel(
                "Detected environments are mapped to small reference molecules. "
                "Use the atom-balance summary to complete the final reaction."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.warning_label = QLabel(
                "⚠️ SciPy not detected or exact MILP balance impossible; operating in elemental balance mode."
            )
            self.warning_label.setStyleSheet("color: #fde293; font-weight: bold;")
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
            self.table.verticalHeader().setVisible(False)
            layout.addWidget(self.table, 1)

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

        def refresh_analysis(self) -> None:
            if Chem is None:
                _status(self.context, "RDKit is not available.", 5000)
                return

            mol = _current_molecule(self.context)
            if mol is None or mol.GetNumAtoms() == 0:
                self.last_result = AnalysisResult(
                    "",
                    Counter(),
                    Counter(),
                    Counter(),
                    (),
                    (),
                    Counter(),
                    Counter(),
                    (),
                    "No molecule is loaded.",
                    "<p>No molecule is loaded.</p>",
                    False,
                    Counter(),
                    Counter(),
                    "Unbalanced",
                    Counter(),
                    Counter(),
                )
                self._populate_table(self.last_result)
                self.equation_box.setHtml(self.last_result.equation_html)
                self.warning_label.setVisible(False)
                _status(self.context, "No molecule is loaded.", 3000)
                return

            self.analyze_button.setEnabled(False)
            self.loading_dialog = LoadingDialog(self)

            self.analysis_thread = QThread()
            self.worker = AnalysisWorker(mol)
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
            self.warning_label.setVisible(self.last_result.is_elemental_balance)
            _status(
                self.context,
                f"Detected {len(self.last_result.matches)} environment types.",
                3000,
            )

        def on_analysis_error(self, error_msg: str) -> None:
            self.analyze_button.setEnabled(True)
            QMessageBox.critical(self, "Analysis Failed", f"An error occurred during analysis:\n{error_msg}")
            _status(self.context, "Analysis failed.", 5000)

        def _populate_table(self, result: AnalysisResult) -> None:
            matches = [m for m in result.matches if m.count > 0]
            left = [t for t in result.left_balance_terms if t.count > 0]
            right = [t for t in result.right_balance_terms if t.count > 0]
            
            self.table.setRowCount(len(matches) + len(left) + len(right))
            
            ref_color = QColor("#1a73e8")
            ref_bg = QColor("#1a73e8")
            ref_bg.setAlpha(30)
            ref_brush = QBrush(ref_color)
            ref_bg_brush = QBrush(ref_bg)
            
            left_color = QColor("#1b7a2d")
            left_bg = QColor("#1b7a2d")
            left_bg.setAlpha(30)
            left_brush = QBrush(left_color)
            left_bg_brush = QBrush(left_bg)
            
            right_color = QColor("#a142f4")
            right_bg = QColor("#a142f4")
            right_bg.setAlpha(30)
            right_brush = QBrush(right_color)
            right_bg_brush = QBrush(right_bg)
            
            row = 0
            for match in matches:
                self._set_colored_row(row, f"Ref: {match.name}", str(match.count), match.reference_smiles, ref_brush, ref_bg_brush)
                self._add_load_button(row, match.reference_smiles, match.description)
                row += 1
                
            for term in left:
                self._set_colored_row(row, f"Left Balance: {term.name}", str(term.count), term.smiles, left_brush, left_bg_brush)
                self._add_load_button(row, term.smiles, term.description or "Added left-side balance species")
                row += 1
                
            for term in right:
                self._set_colored_row(row, f"Right Balance: {term.name}", str(term.count), term.smiles, right_brush, right_bg_brush)
                self._add_load_button(row, term.smiles, term.description or "Added right-side balance species")
                row += 1

        def _set_colored_row(self, row: int, col0: str, col1: str, col2: str, fg_brush: QBrush, bg_brush: QBrush) -> None:
            for col, text in enumerate((col0, col1, col2)):
                item = QTableWidgetItem(text)
                item.setForeground(fg_brush)
                item.setBackground(bg_brush)
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
                main_window = self.context.get_main_window() if hasattr(self.context, "get_main_window") else None
                current_file_path = getattr(main_window.init_manager, "current_file_path", None) if hasattr(main_window, "init_manager") else None
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
