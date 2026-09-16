"""
Saving the user's entries into the project file and reading them back.

A substituted reference is part of how a strain energy was defined, so it has
to survive closing the document. Everything restored here came off disk and
may have been hand-edited, so the load path is tested against rubbish as well
as against its own output.
"""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strain_homodesmotic_reaction_generator as plugin  # noqa: E402
from strain_homodesmotic_reaction_generator import ui  # noqa: E402
from strain_homodesmotic_reaction_generator.core import Chem  # noqa: E402
from strain_homodesmotic_reaction_generator.data import (  # noqa: E402
    SIDE_ANY,
    SIDE_LEFT,
    SIDE_RIGHT,
    UserSpecies,
)

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")

CYCLOPROPANE = "C1CC1"
CP_ENVIRONMENT = "secondary carbon - secondary carbon"


class _FakeContext:
    def __init__(self, smiles=CYCLOPROPANE):
        self._molecule = Chem.MolFromSmiles(smiles) if smiles else None
        self.status_messages = []
        self.windows = {}
        self.handlers = {}

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

    def add_analysis_tool(self, label, callback):
        self.handlers["tool"] = (label, callback)

    def register_save_handler(self, callback):
        self.handlers["save"] = callback

    def register_load_handler(self, callback):
        self.handlers["load"] = callback

    def register_document_reset_handler(self, callback):
        self.handlers["reset"] = callback


def _dialog(context=None):
    return ui.HomodesmoticAnalyzerDialog(context or _FakeContext())


def _populated():
    """A session with one of each kind of entry."""
    ui._session["reference_overrides"] = {CP_ENVIRONMENT: "CCCCC"}
    ui._session["user_species"] = [
        UserSpecies("CCCCCCC", "my heptane", True, SIDE_RIGHT)
    ]
    ui._session["excluded"] = ["CCC"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_initialize_registers_all_three_handlers():
    context = _FakeContext()
    plugin.initialize(context)
    assert context.handlers["save"] is ui.session_state
    assert context.handlers["load"] is ui.restore_session_state
    assert context.handlers["reset"] is ui.reset_session_state


def test_initialize_survives_a_host_without_the_handlers():
    """Older hosts are in the supported range and must not break on load."""

    class _Old:
        def __init__(self):
            self.label = None

        def add_analysis_tool(self, label, callback):
            self.label = label

    context = _Old()
    plugin.initialize(context)
    assert context.label == "Homodesmotic Reaction"


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_nothing_is_written_when_nothing_was_specified():
    """An untouched plugin should not leave a mark in every project file."""
    assert ui.session_state() == {}


def test_the_saved_state_is_json_serialisable():
    _populated()
    assert json.loads(json.dumps(ui.session_state())) == ui.session_state()


def test_every_kind_of_entry_is_saved():
    _populated()
    state = ui.session_state()
    assert state["reference_overrides"] == {CP_ENVIRONMENT: "CCCCC"}
    assert state["excluded"] == ["CCC"]
    assert state["user_species"] == [
        {
            "smiles": "CCCCCCC",
            "name": "my heptane",
            "required": True,
            "side": SIDE_RIGHT,
        }
    ]


def test_the_dialog_writes_its_entries_into_the_session():
    dialog = _dialog()
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))
    dialog.apply_new_entry(("reference", CP_ENVIRONMENT, "CCCCC"))
    state = ui.session_state()
    assert state["reference_overrides"] == {CP_ENVIRONMENT: "CCCCC"}
    assert [entry["smiles"] for entry in state["user_species"]] == ["CCCCCCC"]


def test_an_exclusion_is_saved_too():
    dialog = _dialog()
    row = next(
        index
        for index, data in enumerate(dialog._table_rows)
        if data["remove"][0] == "exclude"
    )
    dialog.table.selectRows([row])
    dialog.remove_selected_species()
    assert ui.session_state()["excluded"] == [dialog._table_rows[-1]["smiles"]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_saved_session_round_trips():
    _populated()
    saved = ui.session_state()
    ui.reset_session_state()
    assert ui.session_state() == {}

    ui.restore_session_state(saved)
    assert ui.session_state() == saved


def test_a_restored_session_seeds_the_next_dialog():
    _populated()
    saved = ui.session_state()
    ui.reset_session_state()
    ui.restore_session_state(saved)

    dialog = _dialog()
    assert dialog.reference_overrides == {CP_ENVIRONMENT: "CCCCC"}
    assert dialog.user_species == [
        UserSpecies("CCCCCCC", "my heptane", True, SIDE_RIGHT)
    ]
    assert dialog.excluded == ["CCC"]


def test_a_restored_session_reaches_an_open_dialog_as_pending():
    """Loading a project must not silently start a multi-second analysis."""
    dialog = _dialog()
    ui._open_dialog = dialog
    assert dialog.pending_label.isVisible() is False

    ui.restore_session_state(
        {"reference_overrides": {CP_ENVIRONMENT: "CCCCC"}, "user_species": []}
    )
    assert dialog.reference_overrides == {CP_ENVIRONMENT: "CCCCC"}
    assert dialog.pending_label.isVisible() is True


@pytest.mark.parametrize(
    "rubbish",
    [
        None,
        "not a dict",
        42,
        [],
        {"reference_overrides": "not a dict"},
        {"user_species": "not a list"},
        {"excluded": None},
    ],
)
def test_rubbish_on_disk_is_ignored_rather_than_loaded(rubbish):
    ui.restore_session_state(rubbish)
    assert ui.session_state() == {}


def test_a_newer_format_is_left_alone():
    _populated()
    before = ui.session_state()
    ui.restore_session_state({"format": ui.SESSION_FORMAT + 1, "excluded": ["CC"]})
    assert ui.session_state() == before


def test_an_unparseable_smiles_is_dropped_on_load():
    ui.restore_session_state(
        {
            "reference_overrides": {CP_ENVIRONMENT: "not-a-molecule"},
            "user_species": [{"smiles": "also-not-a-molecule"}],
            "excluded": ["nor-this"],
        }
    )
    assert ui.session_state() == {}


def test_an_override_for_an_unknown_environment_is_dropped_on_load():
    """The rule list can change between versions; a stale name means nothing."""
    ui.restore_session_state(
        {"reference_overrides": {"an environment that no longer exists": "CC"}}
    )
    assert ui.session_state() == {}


def test_a_malformed_species_entry_is_dropped_but_its_neighbours_survive():
    ui.restore_session_state(
        {
            "user_species": [
                "not a dict",
                {"smiles": "CCCCCCC", "name": "good"},
                {"name": "no smiles at all"},
            ]
        }
    )
    assert [entry.smiles for entry in ui._session["user_species"]] == ["CCCCCCC"]


def test_an_unknown_side_falls_back_to_either():
    ui.restore_session_state(
        {"user_species": [{"smiles": "CC", "required": True, "side": "sideways"}]}
    )
    assert ui._session["user_species"][0].side == SIDE_ANY


@pytest.mark.parametrize("side", [SIDE_ANY, SIDE_LEFT, SIDE_RIGHT])
def test_a_valid_side_survives_the_round_trip(side):
    ui._session["user_species"] = [UserSpecies("CC", "", True, side)]
    saved = ui.session_state()
    ui.reset_session_state()
    ui.restore_session_state(saved)
    assert ui._session["user_species"][0].side == side


def test_loading_replaces_rather_than_merges():
    """A project's entries are its own; the previous document's must not linger."""
    _populated()
    ui.restore_session_state({"excluded": ["CC"]})
    assert ui._session["reference_overrides"] == {}
    assert ui._session["user_species"] == []
    assert ui._session["excluded"] == ["CC"]


# ---------------------------------------------------------------------------
# File -> New
# ---------------------------------------------------------------------------


def test_a_document_reset_clears_the_entries():
    _populated()
    ui.reset_session_state()
    assert ui.session_state() == {}


def test_a_document_reset_reaches_an_open_dialog():
    dialog = _dialog()
    ui._open_dialog = dialog
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane")))

    ui.reset_session_state()
    assert dialog.user_species == []
    assert dialog.reference_overrides == {}
    assert dialog.excluded == []


def test_a_document_reset_with_no_dialog_open_is_harmless():
    _populated()
    ui._open_dialog = None
    ui.reset_session_state()
    assert ui.session_state() == {}


# ---------------------------------------------------------------------------
# Dialog lifetime
# ---------------------------------------------------------------------------


def test_closing_the_dialog_unregisters_it():
    """A closed dialog left registered is how the editors got dead picking."""
    context = _FakeContext()
    ui.open_analyzer_dialog(context)
    dialog = context.windows[ui.WINDOW_ID]
    assert ui._open_dialog is dialog

    dialog.closeEvent(object())
    assert context.windows[ui.WINDOW_ID] is None
    assert ui._open_dialog is None


def test_entries_survive_closing_and_reopening_the_dialog():
    context = _FakeContext()
    ui.open_analyzer_dialog(context)
    dialog = context.windows[ui.WINDOW_ID]
    dialog.apply_new_entry(("balance", UserSpecies("CCCCCCC", "heptane", True)))
    dialog.closeEvent(object())

    ui.open_analyzer_dialog(context)
    reopened = context.windows[ui.WINDOW_ID]
    assert reopened is not dialog
    assert [entry.smiles for entry in reopened.user_species] == ["CCCCCCC"]
