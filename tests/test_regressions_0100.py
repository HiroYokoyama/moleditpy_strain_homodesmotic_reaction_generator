"""Regressions fixed in 0.10.0: species identity by molecule, and charge."""

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator import core, ui  # noqa: E402
from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
    balance_pool,
    build_equation_check_text,
    check_equation,
    normalize_user_species,
)
from strain_homodesmotic_reaction_generator.data import (  # noqa: E402
    BalanceSpecies,
    UserSpecies,
)


def _sides(result):
    left = Counter()
    for term in result.left_balance_terms:
        if term.count:
            left[core.canonical_smiles(term.smiles)] += term.count
    right = Counter()
    for term in result.right_balance_terms:
        if term.count:
            right[core.canonical_smiles(term.smiles)] += term.count
    for match in result.matches:
        if match.count:
            right[core.canonical_smiles(match.reference_smiles)] += match.count
    return left, right


# -- the same molecule written two ways ---------------------------------------


def test_one_molecule_never_sits_on_both_sides():
    """The library's CC[SiH2]C and the reference C[SiH2]CC did not cancel."""
    result = analyze_molecule(Chem.MolFromSmiles("C[SiH2]CC1CC1"))
    left, right = _sides(result)
    assert not set(left) & set(right), set(left) & set(right)
    assert result.reaction_type != "Unbalanced"


def test_a_cancelled_reference_written_differently_is_reported():
    result = analyze_molecule(Chem.MolFromSmiles("C[SiH2]CC1CC1"))
    assert "primary carbon - silane silicon - secondary carbon" in (
        result.cancelled_references
    )


def test_an_exclusion_matches_the_molecule_not_the_spelling():
    pool, _ = balance_pool((), ("OCC",))  # the library writes ethanol as CCO
    assert "CCO" not in {species.smiles for species in pool}


def test_a_user_species_matching_a_library_entry_is_not_duplicated():
    pool, user = balance_pool((BalanceSpecies("mine", "OCC", "User-defined"),))
    ethanol = [s for s in pool if core.canonical_smiles(s.smiles) == "CCO"]
    assert len(ethanol) == 1
    assert "CCO" in user


def test_duplicate_user_species_collapse_by_molecule():
    kept, _ = normalize_user_species(
        (UserSpecies("OCC", "a"), UserSpecies("CCO", "b", True))
    )
    assert [entry.smiles for entry in kept] == ["OCC"]
    assert kept[0].required


def test_a_required_species_is_found_under_another_spelling():
    """Requiring OCC must pin the library's CCO, not report it unplaceable."""
    result = analyze_molecule(
        Chem.MolFromSmiles("C1CC1"), user_species=(UserSpecies("OCC", "", True),)
    )
    assert "OCC" not in result.unmet_required
    used = {
        core.canonical_smiles(term.smiles)
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count
    }
    assert "CCO" in used


# -- charge ----------------------------------------------------------------------


def test_charge_must_balance():
    """[Na+] -> [Na] balances every element and passed as hyperhomodesmotic."""
    check = check_equation("[Na+] -> [Na]")
    assert check.reaction_type == "Unbalanced"
    assert check.charge_delta == -1
    assert "Charge: NOT conserved (-1)" in build_equation_check_text(check)


def test_balanced_ions_still_classify():
    check = check_equation("[NH4+] + [OH-] -> N + O")
    assert check.charge_delta == 0
    assert check.reaction_type != "Unbalanced"


def test_neutral_equations_are_unchanged():
    check = check_equation("C1CC1 + 3 CC -> 3 CCC")
    assert check.charge_delta == 0
    assert "Charge" not in build_equation_check_text(check)
    assert check.reaction_type == "Homodesmotic"


# -- the dialog follows the same identity ----------------------------------------


class _Context:
    def __init__(self, molecule):
        self._molecule = molecule
        self.windows = {}

    @property
    def current_molecule(self):
        return self._molecule

    def get_main_window(self):
        return None

    def show_status_message(self, *_args):
        pass

    def register_window(self, key, window):
        self.windows[key] = window

    def get_window(self, key):
        return self.windows.get(key)


def test_dialog_finds_a_user_species_under_the_library_spelling():
    dialog = ui.HomodesmoticAnalyzerDialog(_Context(None))
    dialog.user_species = [UserSpecies("OCC", "mine", True)]
    assert dialog._species_for("CCO") is dialog.user_species[0]


def test_dialog_replaces_an_entry_written_differently():
    dialog = ui.HomodesmoticAnalyzerDialog(_Context(None))
    dialog.user_species = [UserSpecies("OCC", "old")]
    dialog.excluded = ["CCO"]
    dialog.apply_new_entry(("balance", UserSpecies("CCO", "new", True)))
    assert [s.name for s in dialog.user_species] == ["new"]
    assert dialog.excluded == []
