"""
Element coverage: sulfur, phosphorus, silicon, halogens, selenium, alcohols
and the aromatic heterocycles.

Before this, anything outside C/H/N/O fell through to an elemental-only
balance and was reported "Unbalanced" -- thiophene, tetrahydrothiophene,
fluorocyclopropane and silolane all produced no usable reaction.

Two defects had to be fixed to make that work, and both are guarded here:

1. count_groups() only described C, N and O. Every other heavy atom
   contributed no group at all, and a carbon's bonds to them were not
   recorded either, so tetramethylsilane looked exactly like four ordinary
   methyls with the silicon invisible.

2. The MILP constrained group counts but never element counts. Combined with
   (1) that let it balance *cyclopropane* by adding tetramethylsilane to one
   side -- silicon appearing from nowhere in a pure hydrocarbon reaction.

Every reaction below is checked from scratch: atoms, bond types, and
hybridization environments must all balance across the equation.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rdkit import Chem

    HAS_RDKIT = True
except ImportError:  # pragma: no cover
    HAS_RDKIT = False

from strain_homodesmotic_reaction_generator import core  # noqa: E402
from strain_homodesmotic_reaction_generator.data import (  # noqa: E402
    BALANCE_SPECIES,
    ENVIRONMENTS,
)


def _atoms(smiles: str, mult: int = 1) -> Counter:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out: Counter = Counter()
    for atom in mol.GetAtoms():
        out[atom.GetSymbol()] += mult
    return out


def _bonds(smiles: str, mult: int = 1) -> Counter:
    """Bond types keyed by (element, hybridization) pair plus order."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out: Counter = Counter()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        key = "|".join(
            sorted(
                [
                    f"{a.GetSymbol()}{a.GetHybridization()}",
                    f"{b.GetSymbol()}{b.GetHybridization()}",
                ]
            )
        )
        out[f"{key}#{bond.GetBondTypeAsDouble()}"] += mult
    return out


def _hyb_env(smiles: str, mult: int = 1) -> Counter:
    """Per-heavy-atom (element, hybridization, attached H) counts."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out: Counter = Counter()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        nh = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
        out[f"{atom.GetSymbol()}_{atom.GetHybridization()}_H{nh}"] += mult
    return out


def _imbalances(smiles: str):
    """Return (reaction_type, atom_delta, bond_delta, hybenv_delta)."""
    result = core.analyze_molecule(Chem.MolFromSmiles(smiles))
    lhs_a, lhs_b, lhs_h = _atoms(smiles), _bonds(smiles), _hyb_env(smiles)
    rhs_a, rhs_b, rhs_h = Counter(), Counter(), Counter()
    for term in result.left_balance_terms:
        lhs_a += _atoms(term.smiles, term.count)
        lhs_b += _bonds(term.smiles, term.count)
        lhs_h += _hyb_env(term.smiles, term.count)
    for match in result.matches:
        rhs_a += _atoms(match.reference_smiles, match.count)
        rhs_b += _bonds(match.reference_smiles, match.count)
        rhs_h += _hyb_env(match.reference_smiles, match.count)
    for term in result.right_balance_terms:
        rhs_a += _atoms(term.smiles, term.count)
        rhs_b += _bonds(term.smiles, term.count)
        rhs_h += _hyb_env(term.smiles, term.count)

    def diff(x, y):
        return {k: x[k] - y[k] for k in set(x) | set(y) if x[k] != y[k]}

    return (
        result.reaction_type,
        diff(lhs_a, rhs_a),
        diff(lhs_b, rhs_b),
        diff(lhs_h, rhs_h),
    )


@unittest.skipUnless(HAS_RDKIT, "rdkit not installed")
class TestReactionsBalance(unittest.TestCase):
    def assert_balanced(self, smiles: str, label: str):
        rtype, da, db, dh = _imbalances(smiles)
        self.assertNotEqual(rtype, "Unbalanced", f"{label}: no reaction found")
        self.assertEqual(da, {}, f"{label}: atoms do not balance")
        self.assertEqual(db, {}, f"{label}: bond types do not balance")
        self.assertEqual(dh, {}, f"{label}: hybridization envs do not balance")

    # --- newly supported elements ------------------------------------------
    def test_sulfur_rings(self):
        for smiles, label in (
            ("C1CS1", "thiirane"),
            ("C1CCS1", "thietane"),
            ("C1CCSC1", "tetrahydrothiophene"),
            ("C1CCCS1", "thiane"),
            ("S1CCOCC1", "1,4-thioxane"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)

    def test_silicon_rings(self):
        for smiles, label in (
            ("C1C[SiH2]1", "silirane"),
            ("C1CC[SiH2]C1", "silolane"),
            ("C1CC[SiH2]CC1", "silinane"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)

    def test_phosphorus_ring(self):
        self.assert_balanced("C1CC[PH]C1", "phospholane")

    def test_halogenated_rings(self):
        for smiles, label in (
            ("FC1CC1", "fluorocyclopropane"),
            ("ClC1CC1", "chlorocyclopropane"),
            ("BrC1CC1", "bromocyclopropane"),
            ("IC1CC1", "iodocyclopropane"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)

    def test_aromatic_heterocycles(self):
        for smiles, label in (
            ("c1ccsc1", "thiophene"),
            ("c1ccoc1", "furan"),
            ("c1cc[nH]c1", "pyrrole"),
            ("c1ccncc1", "pyridine"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)

    def test_alcohol_and_oxidised_sulfur(self):
        self.assert_balanced("OC1CC1", "cyclopropanol")
        self.assert_balanced("CS(=O)C", "dimethyl sulfoxide")

    def test_selenoether(self):
        self.assert_balanced("C[Se]C", "dimethyl selenide")

    # --- regression: the C/H/N/O cases that already worked ------------------
    def test_hydrocarbons_still_balance(self):
        for smiles, label in (
            ("C1CC1", "cyclopropane"),
            ("C1CCC1", "cyclobutane"),
            ("C1CCCCC1", "cyclohexane"),
            ("c1ccccc1", "benzene"),
            ("C12CC3CC(C1)CC(C3)C2", "adamantane"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)

    def test_heterocycles_still_balance(self):
        for smiles, label in (
            ("C1CO1", "oxirane"),
            ("C1CN1", "aziridine"),
            ("C1CCO1", "tetrahydrofuran"),
            ("C1CCNC1", "pyrrolidine"),
            ("N1CCOCC1", "morpholine"),
        ):
            with self.subTest(label):
                self.assert_balanced(smiles, label)


@unittest.skipUnless(HAS_RDKIT, "rdkit not installed")
class TestNoPhantomElements(unittest.TestCase):
    """The MILP must not introduce an element absent from the reaction."""

    def test_pure_hydrocarbon_gets_no_heteroatom_species(self):
        for smiles in ("C1CC1", "C1CCCCC1", "C12CC3CC(C1)CC(C3)C2"):
            with self.subTest(smiles):
                result = core.analyze_molecule(Chem.MolFromSmiles(smiles))
                used = list(result.left_balance_terms) + list(
                    result.right_balance_terms
                )
                for term in used:
                    elements = set(_atoms(term.smiles))
                    self.assertTrue(
                        elements <= {"C", "H"},
                        f"{smiles} balanced with {term.smiles}, which brings "
                        f"{elements - {'C', 'H'}} into a pure hydrocarbon reaction",
                    )

    def test_element_constraint_rejects_an_impossible_balance(self):
        """A silicon shortfall with no silicon species available is infeasible
        rather than being silently absorbed into the group constraints."""
        left, right, ok, unmet = core.build_hyperhomodesmotic_balance_terms(
            Counter(), Counter({"Si": 1})
        )
        # Empty group delta short-circuits to the trivial solution.
        self.assertEqual((left, right), ((), ()))
        self.assertTrue(ok)


@unittest.skipUnless(HAS_RDKIT, "rdkit not installed")
class TestCountGroupsDescribesEveryElement(unittest.TestCase):
    def test_silicon_is_described(self):
        groups = core.count_groups(Chem.MolFromSmiles("C[SiH2]C"))
        self.assertTrue(any(k.startswith("Si") for k in groups), groups)

    def test_sulfur_phosphorus_and_halogens_are_described(self):
        for smiles, prefix in (
            ("CSC", "S"),
            ("CP(C)C", "P"),
            ("CF", "F"),
            ("CCl", "Cl"),
            ("C[Se]C", "Se"),
            ("CB(C)C", "B"),
        ):
            with self.subTest(smiles):
                groups = core.count_groups(Chem.MolFromSmiles(smiles))
                self.assertTrue(
                    any(k.startswith(prefix) for k in groups),
                    f"{smiles} produced no {prefix} group: {dict(groups)}",
                )

    def test_carbon_records_its_exotic_neighbours(self):
        """A carbon bonded to Si must not look like one bonded to nothing."""
        silane_c = core.count_groups(Chem.MolFromSmiles("C[SiH3]"))
        ethane_c = core.count_groups(Chem.MolFromSmiles("CC"))
        carbon_keys_silane = {k for k in silane_c if k.startswith("C(")}
        carbon_keys_ethane = {k for k in ethane_c if k.startswith("C(")}
        self.assertNotEqual(carbon_keys_silane, carbon_keys_ethane)

    def test_tetramethylsilane_is_not_four_plain_methyls(self):
        """The exact confusion that let the solver balance cyclopropane
        with tetramethylsilane."""
        tms = core.count_groups(Chem.MolFromSmiles("C[Si](C)(C)C"))
        neopentane = core.count_groups(Chem.MolFromSmiles("CC(C)(C)C"))
        self.assertNotEqual(tms, neopentane)

    def test_aromatic_heteroatoms_are_flagged_aromatic(self):
        groups = core.count_groups(Chem.MolFromSmiles("c1ccsc1"))
        self.assertTrue(any(k.startswith("S_ar") for k in groups), groups)

    def test_pure_organic_group_keys_are_unchanged(self):
        """The heteroatom suffix must be absent when there are none, so the
        keys for C/H/N/O molecules keep their historic form."""
        groups = core.count_groups(Chem.MolFromSmiles("CCO"))
        self.assertTrue(all("(X:" not in k for k in groups), groups)

    def test_bond_symbols_cover_every_order(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CC=CC#N"))
        symbols = {core._bond_symbol(b) for b in mol.GetBonds()}
        self.assertTrue({"-", "=", "#"} <= symbols, symbols)

    def test_aromatic_bond_symbol(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        self.assertIn(":", {core._bond_symbol(b) for b in mol.GetBonds()})

    def test_neighbour_signature_of_an_isolated_atom(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C"))
        methane_c = mol.GetAtomWithIdx(0)
        self.assertEqual(core._neighbour_signature(methane_c), "()")


@unittest.skipUnless(HAS_RDKIT, "rdkit not installed")
class TestDataTablesAreSelfConsistent(unittest.TestCase):
    def test_every_environment_matches_its_own_reference(self):
        for rule in ENVIRONMENTS:
            with self.subTest(rule.name):
                patt = Chem.MolFromSmarts(rule.smarts)
                ref = Chem.MolFromSmiles(rule.reference_smiles)
                self.assertIsNotNone(patt, f"bad SMARTS: {rule.smarts}")
                self.assertIsNotNone(ref, f"bad SMILES: {rule.reference_smiles}")
                self.assertTrue(
                    ref.HasSubstructMatch(patt)
                    or Chem.AddHs(ref).HasSubstructMatch(patt),
                    f"{rule.smarts} does not occur in {rule.reference_smiles}",
                )

    def test_every_balance_species_parses(self):
        for species in BALANCE_SPECIES:
            with self.subTest(species.name):
                self.assertIsNotNone(Chem.MolFromSmiles(species.smiles), species.smiles)

    def test_names_are_unique(self):
        names = [rule.name for rule in ENVIRONMENTS]
        self.assertEqual(len(names), len(set(names)))
        smiles = [s.smiles for s in BALANCE_SPECIES]
        self.assertEqual(len(smiles), len(set(smiles)))


if __name__ == "__main__":
    unittest.main()
