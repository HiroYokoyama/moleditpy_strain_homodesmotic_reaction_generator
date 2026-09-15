"""
Shared headless PyQt6 stubs for the Strain Homodesmotic Reaction Generator
test suite.

Installed at collection time (before any test module imports
``strain_homodesmotic_reaction_generator.ui``) so that ``ui.py``'s
``QDialog is not None`` branch executes and its dialog/worker classes are
actually defined -- letting ``tests/test_ui.py`` instantiate and drive them
directly without a real Qt runtime (CI does not install PyQt6 at all).

Uses real-ish stand-in classes (not bare ``MagicMock``) for anything that
``ui.py`` subclasses or wires signals/slots through, so ``connect``/``emit``
behave like the real PyQt6 signal-slot mechanism synchronously.
"""

import inspect
import sys
import types
from unittest.mock import MagicMock


def _install_qt_stubs():
    if "PyQt6" in sys.modules and getattr(
        sys.modules["PyQt6"], "_shrg_rich_stub", False
    ):
        return

    # -- Signal / slot machinery -------------------------------------------------

    class _BoundSignal:
        def __init__(self):
            self._slots = []

        def connect(self, fn):
            self._slots.append(fn)

        def disconnect(self, fn=None):
            if fn is None:
                self._slots.clear()
            else:
                self._slots = [s for s in self._slots if s != fn]

        def emit(self, *args):
            for slot in list(self._slots):
                # Mimic PyQt's leniency: a slot may declare fewer positional
                # parameters than the signal emits (e.g. connecting a
                # ``signal(object)`` to a plain ``thread.quit``/``deleteLater``).
                try:
                    sig = inspect.signature(slot)
                    n_params = len(
                        [
                            p
                            for p in sig.parameters.values()
                            if p.kind
                            in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        ]
                    )
                except (TypeError, ValueError):
                    n_params = len(args)
                slot(*args[:n_params])

    class pyqtSignal:  # noqa: N801 - mimic PyQt6's lowercase factory name
        def __init__(self, *args, **kwargs):
            self._name = None

        def __set_name__(self, owner, name):
            self._name = name

        def __get__(self, instance, owner):
            if instance is None:
                return self
            attr = f"__bound_signal_{self._name}"
            bound = instance.__dict__.get(attr)
            if bound is None:
                bound = _BoundSignal()
                instance.__dict__[attr] = bound
            return bound

    # -- Generic auto-mocking base ------------------------------------------------

    class _QBase:
        """Stand-in base for QObject/QDialog/etc.

        Unknown attributes auto-mock themselves on first access so any Qt
        API surface not explicitly modeled below can still be called.
        """

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            val = MagicMock(name=name)
            object.__setattr__(self, name, val)
            return val

    class _QObject(_QBase):
        pass

    class _QThread(_QBase):
        started = pyqtSignal()
        finished = pyqtSignal()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._running = False

        def start(self):
            self._running = True
            self.started.emit()

        def quit(self):
            self._running = False
            self.finished.emit()

        def isRunning(self):
            return self._running

        def wait(self, ms=0):
            return True

        def deleteLater(self):
            pass

    class _Qt:
        class WindowType:
            WindowCloseButtonHint = 1

        class AlignmentFlag:
            AlignCenter = 1

        class ItemFlag:
            ItemIsSelectable = 1
            ItemIsEditable = 2
            ItemIsEnabled = 32

    class _QDialog(_QBase):
        def __init__(self, parent=None, *args, **kwargs):
            super().__init__(parent, *args, **kwargs)
            self.parent = parent
            self._modal = False
            self._flags = 0
            self._exec_calls = 0
            self._result = None

        def setWindowTitle(self, title):
            self._title = title

        def setModal(self, modal):
            self._modal = modal

        def windowFlags(self):
            return self._flags

        def setWindowFlags(self, flags):
            self._flags = flags

        def resize(self, w, h):
            self._size = (w, h)

        def exec(self):
            self._exec_calls += 1
            return self._result

        def accept(self):
            self._result = 1

        def reject(self):
            self._result = 0

        def show(self):
            self._shown = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

        def close(self):
            pass

    class _QLabel(_QBase):
        def __init__(self, text="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text = text
            self._visible = True
            self._stylesheet = ""
            self._word_wrap = False

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

        def setVisible(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

        def setStyleSheet(self, style):
            self._stylesheet = style

        def setWordWrap(self, wrap):
            self._word_wrap = wrap

        def setAlignment(self, alignment):
            pass

    class _QPushButton(_QBase):
        def __init__(self, text="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text = text
            self._enabled = True
            self._tooltip = ""
            self.clicked = _BoundSignal()

        def setEnabled(self, enabled):
            self._enabled = enabled

        def isEnabled(self):
            return self._enabled

        def setToolTip(self, tooltip):
            self._tooltip = tooltip

        def toolTip(self):
            return self._tooltip

    class _QTableWidgetItem:
        def __init__(self, text=""):
            self._text = text
            self._foreground = None
            self._background = None
            self._tooltip = ""
            self._flags = (
                _Qt.ItemFlag.ItemIsSelectable
                | _Qt.ItemFlag.ItemIsEditable
                | _Qt.ItemFlag.ItemIsEnabled
            )
            self._table = None
            self._row = -1
            self._column = -1

        def text(self):
            return self._text

        def setText(self, text):
            self._text = text
            if self._table is not None:
                self._table._emit_item_changed(self)

        def setForeground(self, brush):
            self._foreground = brush

        def setBackground(self, brush):
            self._background = brush

        def foreground(self):
            return self._foreground

        def background(self):
            return self._background

        def setToolTip(self, tooltip):
            self._tooltip = tooltip

        def toolTip(self):
            return self._tooltip

        def flags(self):
            return self._flags

        def setFlags(self, flags):
            self._flags = flags

        def row(self):
            return self._row

        def column(self):
            return self._column

    class _QModelIndex:
        def __init__(self, row, column):
            self._row = row
            self._column = column

        def row(self):
            return self._row

        def column(self):
            return self._column

    class _QHeader(_QBase):
        def __init__(self):
            super().__init__()
            self.sectionClicked = _BoundSignal()
            self._visible = True

        def setSectionResizeMode(self, *args, **kwargs):
            pass

        def setSectionsClickable(self, clickable):
            self._clickable = clickable

        def setVisible(self, visible):
            self._visible = visible

    class _QTableWidget(_QBase):
        def __init__(self, rows=0, cols=0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rows = rows
            self._cols = cols
            self._items = {}
            self._cell_widgets = {}
            self._header_labels = []
            self._selected = []
            self._h_header = _QHeader()
            self._v_header = _QHeader()
            self.itemChanged = _BoundSignal()

        def horizontalHeader(self):
            return self._h_header

        def verticalHeader(self):
            return self._v_header

        def selectedIndexes(self):
            return [_QModelIndex(row, col) for row, col in self._selected]

        def selectRows(self, rows):
            """Test helper: pretend the user selected these rows."""
            self._selected = [(row, 0) for row in rows]

        def _emit_item_changed(self, item):
            self.itemChanged.emit(item)

        def setRowCount(self, n):
            self._rows = n
            self._items = {k: v for k, v in self._items.items() if k[0] < n}
            self._cell_widgets = {
                k: v for k, v in self._cell_widgets.items() if k[0] < n
            }

        def rowCount(self):
            return self._rows

        def setColumnCount(self, n):
            self._cols = n

        def columnCount(self):
            return self._cols

        def setHorizontalHeaderLabels(self, labels):
            self._header_labels = list(labels)

        def setItem(self, row, col, item):
            self._items[(row, col)] = item
            item._table = self
            item._row = row
            item._column = col

        def item(self, row, col):
            return self._items.get((row, col))

        def setCellWidget(self, row, col, widget):
            self._cell_widgets[(row, col)] = widget

        def cellWidget(self, row, col):
            return self._cell_widgets.get((row, col))

    class _QTextEdit(_QBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._html = ""
            self._read_only = False

        def setReadOnly(self, read_only):
            self._read_only = read_only

        def setHtml(self, html):
            self._html = html

        def toHtml(self):
            return self._html

        def setMinimumHeight(self, h):
            pass

        def setStyleSheet(self, style):
            pass

    class _QMessageBox:
        critical_calls = []
        information_calls = []
        warning_calls = []

        @staticmethod
        def critical(parent, title, text, *args, **kwargs):
            _QMessageBox.critical_calls.append((parent, title, text))

        @staticmethod
        def information(parent, title, text, *args, **kwargs):
            _QMessageBox.information_calls.append((parent, title, text))

        @staticmethod
        def warning(parent, title, text, *args, **kwargs):
            _QMessageBox.warning_calls.append((parent, title, text))

    class _QFileDialog:
        _next_return = ("", "")

        @staticmethod
        def getSaveFileName(parent, caption, default, filter_str):
            return _QFileDialog._next_return

    class _QColor:
        def __init__(self, *args):
            self._args = args
            self._alpha = 255

        def setAlpha(self, alpha):
            self._alpha = alpha

    class _QBrush:
        def __init__(self, color=None):
            self._color = color

    class _QHeaderView:
        class ResizeMode:
            Stretch = 1
            ResizeToContents = 2

    class _QGroupBox(_QBase):
        def __init__(self, title="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._title = title
            self._layout = None

        def setLayout(self, layout):
            self._layout = layout

        def title(self):
            return self._title

    class _QLineEdit(_QBase):
        def __init__(self, text="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text = text
            self._placeholder = ""
            self._visible = True

        def text(self):
            return self._text

        def setText(self, text):
            self._text = text

        def setPlaceholderText(self, text):
            self._placeholder = text

        def setVisible(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

    class _QCheckBox(_QBase):
        def __init__(self, text="", *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._text = text
            self._checked = False
            self._visible = True

        def isChecked(self):
            return self._checked

        def setChecked(self, checked):
            self._checked = checked

        def setVisible(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

    class _QComboBox(_QBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._items = []
            self._current = 0
            self._visible = True
            self.currentTextChanged = _BoundSignal()

        def addItem(self, text):
            self._items.append(text)

        def count(self):
            return len(self._items)

        def itemText(self, index):
            return self._items[index]

        def currentText(self):
            if not self._items:
                return ""
            return self._items[self._current]

        def setCurrentText(self, text):
            if text in self._items:
                self._current = self._items.index(text)
                self.currentTextChanged.emit(text)

        def setVisible(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

    class _QLayout(_QBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._items = []

        def addWidget(self, widget, *args, **kwargs):
            self._items.append(widget)

        def addLayout(self, layout, *args, **kwargs):
            self._items.append(layout)

        def addStretch(self, *args, **kwargs):
            pass

    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = _Qt
    qt_core.QThread = _QThread
    qt_core.pyqtSignal = pyqtSignal
    qt_core.QObject = _QObject

    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_widgets.QCheckBox = _QCheckBox
    qt_widgets.QComboBox = _QComboBox
    qt_widgets.QDialog = _QDialog
    qt_widgets.QFileDialog = _QFileDialog
    qt_widgets.QGroupBox = _QGroupBox
    qt_widgets.QLineEdit = _QLineEdit
    qt_widgets.QHeaderView = _QHeaderView
    qt_widgets.QHBoxLayout = _QLayout
    qt_widgets.QLabel = _QLabel
    qt_widgets.QMessageBox = _QMessageBox
    qt_widgets.QPushButton = _QPushButton
    qt_widgets.QTableWidget = _QTableWidget
    qt_widgets.QTableWidgetItem = _QTableWidgetItem
    qt_widgets.QTextEdit = _QTextEdit
    qt_widgets.QVBoxLayout = _QLayout

    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_gui.QColor = _QColor
    qt_gui.QBrush = _QBrush

    pyqt6 = types.ModuleType("PyQt6")
    pyqt6.QtCore = qt_core
    pyqt6.QtWidgets = qt_widgets
    pyqt6.QtGui = qt_gui
    pyqt6._shrg_rich_stub = True

    for name, mod in [
        ("PyQt6", pyqt6),
        ("PyQt6.QtCore", qt_core),
        ("PyQt6.QtWidgets", qt_widgets),
        ("PyQt6.QtGui", qt_gui),
    ]:
        sys.modules[name] = mod


_install_qt_stubs()
