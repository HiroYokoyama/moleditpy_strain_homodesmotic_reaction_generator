"""
Run a wide spread of molecules through the analyzer and audit every answer.

Nothing here asserts a hard-coded equation: the solver is free to pick any
balance it likes. What is checked is that whatever it picks obeys the laws it
claims to obey, recounted here from the equation terms with code that shares
nothing with ``core`` beyond RDKit itself. A verdict the plugin computes and a
test that reads that verdict back would agree however wrong both were.
"""

from collections import Counter
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator import core  # noqa: E402
from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
    check_equation,
    export_analysis,
    format_equation_sides,
)
from strain_homodesmotic_reaction_generator.data import UserSpecies  # noqa: E402

pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")


# ---------------------------------------------------------------------------
# The molecules
# ---------------------------------------------------------------------------

STRAINED_RINGS = {
    "cyclopropane": "C1CC1",
    "cyclobutane": "C1CCC1",
    "cyclopentane": "C1CCCC1",
    "cyclohexane": "C1CCCCC1",
    "cycloheptane": "C1CCCCCC1",
    "cyclooctane": "C1CCCCCCC1",
    "cyclodecane": "C1CCCCCCCCC1",
    "bicyclobutane": "C1C2CC12",
    "spiropentane": "C1CC12CC2",
    "cubane": "C1(C2C3C4C1C5C4C3C25)",
    "prismane": "C12C3C1C1C2C31",
    "bicyclo[1.1.0]pentane": "C1C2CC2C1",
    "bicyclo[2.2.0]hexane": "C1CC2CCC12",
    "norbornane": "C1CC2CCC1C2",
    "bicyclo[2.2.2]octane": "C1CC2CCC1CC2",
    "adamantane": "C1C2CC3CC1CC(C2)C3",
    "spiro[3.3]heptane": "C1CC2(C1)CCC2",
    "spiro[2.5]octane": "C1CC12CCCCC2",
    "housane": "C1C2CC2C1",
    "methylcyclopropane": "CC1CC1",
    "ethylcyclopropane": "CCC1CC1",
    "cyclopropylcyclopropane": "C1CC1C1CC1",
    "trans-decalin": "C1CCC2CCCCC2C1",
}

HETEROCYCLES = {
    "oxirane": "C1CO1",
    "oxetane": "C1COC1",
    "tetrahydrofuran": "C1CCOC1",
    "tetrahydropyran": "C1CCOCC1",
    "oxepane": "C1CCCOCC1",
    "aziridine": "C1CN1",
    "azetidine": "C1CNC1",
    "pyrrolidine": "C1CCNC1",
    "piperidine": "C1CCNCC1",
    "thiirane": "C1CS1",
    "thietane": "C1CSC1",
    "tetrahydrothiophene": "C1CCSC1",
    "dioxolane": "C1COCO1",
    "dioxane": "C1COCCO1",
    "morpholine": "C1COCCN1",
    "piperazine": "C1CNCCN1",
    "oxaspiropentane": "C1CC12CO2",
    "2-methyloxirane": "CC1CO1",
    "N-methylaziridine": "CN1CC1",
}

UNSATURATED = {
    "cyclopropene": "C1=CC1",
    "cyclobutene": "C1=CCC1",
    "cyclopentene": "C1=CCCC1",
    "cyclohexene": "C1=CCCCC1",
    "cycloheptene": "C1=CCCCCC1",
    "1,3-cyclohexadiene": "C1=CC=CCC1",
    "1,4-cyclohexadiene": "C1=CCC=CC1",
    "cyclooctatetraene": "C1=CC=CC=CC=C1",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "o-xylene": "Cc1ccccc1C",
    "styrene": "C=Cc1ccccc1",
    "biphenyl": "c1ccc(-c2ccccc2)cc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "indane": "C1Cc2ccccc2C1",
    "1,3-butadiene": "C=CC=C",
    "1,3-pentadiene": "C=CC=CC",
    "allene": "C=C=C",
    "propyne": "CC#C",
    "2-butyne": "CC#CC",
    "phenylacetylene": "C#Cc1ccccc1",
    "methylenecyclopropane": "C=C1CC1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrrole": "c1cc[nH]c1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "imidazole": "c1c[nH]cn1",
}

ACYL = {
    "formic acid": "OC=O",
    "acetic acid": "CC(=O)O",
    "propanoic acid": "CCC(=O)O",
    "isobutyric acid": "CC(C)C(=O)O",
    "pivalic acid": "CC(C)(C)C(=O)O",
    "cyclopropanecarboxylic acid": "OC(=O)C1CC1",
    "methyl formate": "COC=O",
    "methyl acetate": "COC(C)=O",
    "ethyl acetate": "CCOC(C)=O",
    "methyl propanoate": "CCC(=O)OC",
    "beta-propiolactone": "O=C1CCO1",
    "gamma-butyrolactone": "O=C1CCCO1",
    "delta-valerolactone": "O=C1CCCCO1",
    "formamide": "NC=O",
    "acetamide": "CC(N)=O",
    "propanamide": "CCC(N)=O",
    "N-methylacetamide": "CNC(C)=O",
    "N,N-dimethylacetamide": "CN(C)C(C)=O",
    "beta-lactam": "O=C1CCN1",
    "gamma-lactam": "O=C1CCCN1",
    "succinimide": "O=C1CCC(=O)N1",
    "urea": "NC(N)=O",
    "dimethyl carbonate": "COC(=O)OC",
    "oxalic acid dimethyl ester": "COC(=O)C(=O)OC",
}

FUNCTIONALISED = {
    "acetone": "CC(=O)C",
    "acetaldehyde": "CC=O",
    "butanone": "CCC(C)=O",
    "cyclohexanone": "O=C1CCCCC1",
    "cyclobutanone": "O=C1CCC1",
    "cyclopropanone": "O=C1CC1",
    "acetonitrile": "CC#N",
    "propanenitrile": "CCC#N",
    "cyclopropanecarbonitrile": "N#CC1CC1",
    "methanol": "CO",
    "ethanol": "CCO",
    "isopropanol": "CC(C)O",
    "cyclopropanol": "OC1CC1",
    "ethylene glycol": "OCCO",
    "dimethyl ether": "COC",
    "diethyl ether": "CCOCC",
    "methylamine": "CN",
    "dimethylamine": "CNC",
    "trimethylamine": "CN(C)C",
    "cyclopropylamine": "NC1CC1",
    "dimethyl sulfide": "CSC",
    "methanethiol": "CS",
    "dimethyl sulfoxide": "CS(=O)C",
    "dimethyl sulfone": "CS(=O)(=O)C",
    "trimethylphosphine": "CP(C)C",
    "tetramethylsilane": "C[Si](C)(C)C",
    "chlorocyclopropane": "ClC1CC1",
    "fluorocyclobutane": "FC1CCC1",
    "bromoethane": "CCBr",
    "iodomethane": "CI",
    "1,2-dichloroethane": "ClCCCl",
    "trimethylborane": "CB(C)C",
    "dimethyl selenide": "C[Se]C",
}

BRANCHED = {
    "isobutane": "CC(C)C",
    "neopentane": "CC(C)(C)C",
    "2,3-dimethylbutane": "CC(C)C(C)C",
    "2,2,3,3-tetramethylbutane": "CC(C)(C)C(C)(C)C",
    "tert-butylcyclopropane": "CC(C)(C)C1CC1",
    "1,1-dimethylcyclopropane": "CC1(C)CC1",
    "isooctane": "CC(C)CC(C)(C)C",
    "octane": "CCCCCCCC",
    "hexadecane": "CCCCCCCCCCCCCCCC",
}

ALL_MOLECULES = {}
for group in (
    STRAINED_RINGS,
    HETEROCYCLES,
    UNSATURATED,
    ACYL,
    FUNCTIONALISED,
    BRANCHED,
):
    ALL_MOLECULES.update(group)

NAMES = sorted(ALL_MOLECULES)


# ---------------------------------------------------------------------------
# An independent recount
# ---------------------------------------------------------------------------


def _explicit(smiles: str):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _label(atom) -> str:
    hydrogens = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    aromatic = "a" if atom.GetIsAromatic() else ""
    return f"{atom.GetSymbol()}{aromatic}H{hydrogens}"


def _atoms(smiles: str) -> Counter:
    return Counter(atom.GetSymbol() for atom in _explicit(smiles).GetAtoms())


def _groups(smiles: str) -> Counter:
    return Counter(
        _label(atom)
        for atom in _explicit(smiles).GetAtoms()
        if atom.GetAtomicNum() != 1
    )


def _state(atom) -> str:
    """Element, hybridisation and hydrogen count.

    Deliberately the same distinction the plugin makes. Labelling by hydrogen
    count alone was coarser: it could not tell a carbonyl carbon from a
    quaternary one, and passed three lactones whose bond types the plugin had
    correctly found not conserved.
    """
    if atom.GetAtomicNum() == 1:
        return "H"
    hydrogens = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    return f"{atom.GetSymbol()}({str(atom.GetHybridization()).lower()},H{hydrogens})"


def _bonds(smiles: str) -> Counter:
    counts: Counter = Counter()
    for bond in _explicit(smiles).GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
            continue
        pair = "-".join(sorted((_state(begin), _state(end))))
        counts[f"{pair} {bond.GetBondType()}"] += 1
    return counts


def _sides(result):
    left = [(1, result.target_smiles)]
    left += [
        (term.count, term.smiles)
        for term in result.left_balance_terms
        if term.count > 0
    ]
    right = [
        (match.count, match.reference_smiles)
        for match in result.matches
        if match.count > 0
    ]
    right += [
        (term.count, term.smiles)
        for term in result.right_balance_terms
        if term.count > 0
    ]
    return left, right


def _total(side, counter) -> Counter:
    total: Counter = Counter()
    for count, smiles in side:
        for key, value in counter(smiles).items():
            total[key] += count * value
    return total


def _conserved(result) -> dict:
    left, right = _sides(result)
    return {
        "atoms": _total(left, _atoms) == _total(right, _atoms),
        "groups": _total(left, _groups) == _total(right, _groups),
        "bonds": _total(left, _bonds) == _total(right, _bonds),
    }


def _analyse(name):
    return analyze_molecule(Chem.MolFromSmiles(ALL_MOLECULES[name]))


# ---------------------------------------------------------------------------
# Every molecule, every time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_analysis_completes_and_says_something(name):
    result = _analyse(name)
    assert result.target_smiles
    assert result.equation_text
    assert result.equation_html
    assert result.reaction_type in {
        "Hyperhomodesmotic",
        "Homodesmotic",
        "Isodesmic",
        "Elemental Balance",
        "Unbalanced",
    }


@pytest.mark.parametrize("name", NAMES)
def test_a_balanced_verdict_really_balances(name):
    """Anything but Unbalanced is a claim about atoms; check the claim."""
    result = _analyse(name)
    if result.reaction_type == "Unbalanced":
        pytest.skip("reported as unbalanced, which is its own answer")
    assert _conserved(result)["atoms"], result.equation_text.splitlines()[3]


@pytest.mark.parametrize("name", NAMES)
def test_a_hyperhomodesmotic_verdict_conserves_every_bond_type(name):
    result = _analyse(name)
    if result.reaction_type != "Hyperhomodesmotic":
        pytest.skip("not claimed to be hyperhomodesmotic")
    conserved = _conserved(result)
    assert conserved["atoms"]
    assert conserved["groups"]
    assert conserved["bonds"], result.equation_text.splitlines()[3]


@pytest.mark.parametrize("name", NAMES)
def test_a_homodesmotic_verdict_conserves_the_groups(name):
    result = _analyse(name)
    if result.reaction_type != "Homodesmotic":
        pytest.skip("not claimed to be homodesmotic")
    conserved = _conserved(result)
    assert conserved["atoms"]
    assert conserved["groups"], result.equation_text.splitlines()[3]


@pytest.mark.parametrize("name", NAMES)
def test_nothing_is_left_unresolved_without_being_called_unbalanced(name):
    """A guard, not a measurement: no molecule here currently leaves atoms over."""
    result = _analyse(name)
    leftover = result.unresolved_left_atoms or result.unresolved_right_atoms
    if leftover:
        assert result.reaction_type == "Unbalanced"


def test_the_conservation_checks_are_not_all_skipping():
    """Those tests skip on the verdict, so a bad change could empty them out."""
    verdicts = Counter(_analyse(name).reaction_type for name in NAMES)
    assert verdicts["Hyperhomodesmotic"] >= 50
    assert verdicts["Homodesmotic"] >= 10
    assert sum(verdicts.values()) == len(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_no_species_appears_on_both_sides(name):
    """The cancellation pass exists to stop this; it must actually run."""
    result = _analyse(name)
    left, right = _sides(result)
    shared = {smiles for _c, smiles in left} & {smiles for _c, smiles in right}
    assert shared <= {result.target_smiles}


@pytest.mark.parametrize("name", NAMES)
def test_a_cancelled_term_never_reaches_the_equation(name):
    """Terms drop to zero during cancellation and must then disappear."""
    result = _analyse(name)
    cancelled = [
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count == 0
    ] + [match.reference_smiles for match in result.matches if match.count == 0]
    line = result.equation_text.splitlines()[3]
    surviving = {smiles for _count, smiles in _sides(result)[0]}
    surviving |= {smiles for _count, smiles in _sides(result)[1]}
    for smiles in cancelled:
        if smiles not in surviving:
            assert f" {smiles} " not in f" {line} ", smiles
    # A standalone zero coefficient, not the 0 inside "10 CC".
    assert not re.search(r"(^|\+ )0 ", line)


@pytest.mark.parametrize("name", NAMES)
def test_the_equation_line_round_trips_through_the_tester(name):
    """The report's own line must be readable by the tool that reads lines."""
    result = _analyse(name)
    left, right = _sides(result)
    if not right:
        pytest.skip("no reference molecules survived cancellation")
    checked = check_equation(format_equation_sides(left, right))
    assert checked.ok, checked.errors
    assert checked.reaction_type == result.reaction_type


@pytest.mark.parametrize("name", NAMES)
def test_every_report_format_is_produced(name, tmp_path):
    result = _analyse(name)
    for suffix in (".html", ".csv", ".txt"):
        target = tmp_path / f"report{suffix}"
        export_analysis(target, result)
        assert target.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# The same sweep, with the user interfering
# ---------------------------------------------------------------------------

#: A handful of targets is enough here: this exercises the solver paths, not
#: the environment detection, and every case costs a MILP solve.
SAMPLE = [
    "cyclopropane",
    "cyclobutane",
    "cyclopentane",
    "oxetane",
    "norbornane",
    "chlorocyclopropane",
]


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
@pytest.mark.parametrize("name", SAMPLE)
def test_a_required_species_never_breaks_the_balance(name):
    result = analyze_molecule(
        Chem.MolFromSmiles(ALL_MOLECULES[name]),
        user_species=[UserSpecies("CCCCCC", "hexane", required=True)],
    )
    if result.reaction_type == "Unbalanced":
        pytest.skip("reported as unbalanced, which is its own answer")
    assert _conserved(result)["atoms"], result.equation_text.splitlines()[3]


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
@pytest.mark.parametrize("name", SAMPLE)
def test_a_required_species_is_either_used_or_named_as_unusable(name):
    """It must never be silently dropped."""
    result = analyze_molecule(
        Chem.MolFromSmiles(ALL_MOLECULES[name]),
        user_species=[UserSpecies("CCCCCC", "hexane", required=True)],
    )
    used = {
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count > 0
    }
    assert ("CCCCCC" in used) or ("CCCCCC" in result.unmet_required)


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
@pytest.mark.parametrize("name", SAMPLE)
def test_an_exclusion_is_honoured_or_reported(name):
    result = analyze_molecule(
        Chem.MolFromSmiles(ALL_MOLECULES[name]), excluded_smiles=("CCC",)
    )
    used = {
        term.smiles
        for term in result.left_balance_terms + result.right_balance_terms
        if term.count > 0
    }
    assert ("CCC" not in used) or ("CCC" in result.ignored_exclusions)


@pytest.mark.skipif(core.milp is None, reason="SciPy is required for the MILP path")
@pytest.mark.parametrize("name", SAMPLE)
def test_an_exclusion_never_breaks_the_balance(name):
    result = analyze_molecule(
        Chem.MolFromSmiles(ALL_MOLECULES[name]), excluded_smiles=("CCC", "CC")
    )
    if result.reaction_type == "Unbalanced":
        pytest.skip("reported as unbalanced, which is its own answer")
    assert _conserved(result)["atoms"], result.equation_text.splitlines()[3]


@pytest.mark.parametrize("name", SAMPLE)
def test_overriding_a_reference_never_breaks_the_atom_balance(name):
    """Substituting a same-element reference must stay balanced."""
    baseline = _analyse(name)
    detected = [match.name for match in baseline.matches if match.count > 0]
    if not detected:
        pytest.skip("no environment detected")
    result = analyze_molecule(
        Chem.MolFromSmiles(ALL_MOLECULES[name]), {detected[0]: "CCCCC"}
    )
    if result.reaction_type == "Unbalanced":
        pytest.skip("reported as unbalanced, which is its own answer")
    assert _conserved(result)["atoms"], result.equation_text.splitlines()[3]


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    ["C", "CC", "[H][H]", "O", "N", "[SiH4]", "C1CC1.C1CC1", "CCCCCCCCCCCCCCCCCC"],
)
def test_odd_targets_do_not_crash(smiles):
    result = analyze_molecule(Chem.MolFromSmiles(smiles))
    assert result.equation_text
    if result.reaction_type != "Unbalanced":
        assert _conserved(result)["atoms"]


def test_an_empty_molecule_reports_rather_than_crashing():
    result = analyze_molecule(Chem.MolFromSmiles(""))
    assert result.reaction_type == "Unbalanced"
    assert "No molecule is loaded." in result.equation_text


def test_a_none_molecule_reports_rather_than_crashing():
    assert analyze_molecule(None).reaction_type == "Unbalanced"


# ---------------------------------------------------------------------------
# A known gap, pinned so it cannot become a silent wrong answer
# ---------------------------------------------------------------------------

#: These were unbalanceable until acid, ester and amide rules were added: the
#: plain carbonyl rule claimed them and proposed a ketone reference carrying
#: neither the second oxygen nor the nitrogen. They are pinned here because
#: they are the cases that regress first if those rules are ever loosened.
ACYL_TARGETS = {
    "acetic acid": "CC(=O)O",
    "ethyl acetate": "CCOC(C)=O",
    "acetamide": "CC(N)=O",
    "beta-propiolactone": "O=C1CCO1",
    "formic acid": "OC=O",
    "methyl formate": "COC=O",
    "urea": "NC(N)=O",
}


@pytest.mark.parametrize("name", sorted(ACYL_TARGETS))
def test_an_acyl_target_balances(name):
    result = analyze_molecule(Chem.MolFromSmiles(ACYL_TARGETS[name]))
    assert result.reaction_type in {"Hyperhomodesmotic", "Homodesmotic"}
    assert _conserved(result)["atoms"], result.equation_text.splitlines()[3]
    assert not result.unresolved_left_atoms
    assert not result.unresolved_right_atoms


@pytest.mark.parametrize("name", sorted(ACYL_TARGETS))
def test_an_acyl_target_gets_a_reference_of_its_own_kind(name):
    """A ketone reference for an ester is the bug these rules were added for."""
    result = analyze_molecule(Chem.MolFromSmiles(ACYL_TARGETS[name]))
    assert [match for match in result.matches if match.count > 0]


#: The ketone and aldehyde rules, which an acid, ester or amide must not match.
#: Listed explicitly rather than matched by suffix, because Urea-Carbonyl is a
#: carbonyl rule too and it is supposed to claim a urea.
PLAIN_CARBONYL_RULES = {
    "Primary-Carbonyl",
    "Secondary-Carbonyl",
    "Tertiary-Carbonyl",
    "Quaternary-Carbonyl",
}


def test_the_plain_carbonyl_rules_no_longer_claim_an_acyl_carbon():
    from strain_homodesmotic_reaction_generator.core import (
        unique_substructure_matches,
    )
    from strain_homodesmotic_reaction_generator.data import ENVIRONMENTS

    plain = [rule for rule in ENVIRONMENTS if rule.name in PLAIN_CARBONYL_RULES]
    assert len(plain) == len(PLAIN_CARBONYL_RULES)
    for name, smiles in sorted(ACYL_TARGETS.items()):
        mol = Chem.MolFromSmiles(smiles)
        for rule in plain:
            assert not unique_substructure_matches(
                mol, rule.smarts, rule.atom_centric
            ), f"{rule.name} still claims {name}"


@pytest.mark.parametrize("name", ["acetone", "acetaldehyde", "cyclohexanone"])
def test_a_real_ketone_or_aldehyde_is_still_claimed(name):
    """Tightening the rule must not throw the ketones out with the esters."""
    from strain_homodesmotic_reaction_generator.core import (
        unique_substructure_matches,
    )
    from strain_homodesmotic_reaction_generator.data import ENVIRONMENTS

    mol = Chem.MolFromSmiles(ALL_MOLECULES[name])
    plain = [rule for rule in ENVIRONMENTS if rule.name in PLAIN_CARBONYL_RULES]
    assert any(
        unique_substructure_matches(mol, rule.smarts, rule.atom_centric)
        for rule in plain
    )


def test_no_two_environment_rules_claim_the_same_atoms():
    """Overlapping rules would count one group twice and inflate the references."""
    from collections import defaultdict

    from strain_homodesmotic_reaction_generator.core import (
        unique_substructure_matches,
    )
    from strain_homodesmotic_reaction_generator.data import ENVIRONMENTS

    targets = dict(ALL_MOLECULES)
    targets.update(ACYL_TARGETS)
    for name, smiles in sorted(targets.items()):
        mol = Chem.MolFromSmiles(smiles)
        owners = defaultdict(list)
        for rule in ENVIRONMENTS:
            for match in unique_substructure_matches(
                mol, rule.smarts, rule.atom_centric
            ):
                owners[match].append(rule.name)
        clashes = {k: v for k, v in owners.items() if len(v) > 1}
        assert not clashes, f"{name}: {clashes}"


def test_every_reference_molecule_contains_the_environment_it_stands_for():
    """A reference that lacks its own environment makes the mapping meaningless."""
    from strain_homodesmotic_reaction_generator.core import (
        unique_substructure_matches,
    )
    from strain_homodesmotic_reaction_generator.data import ENVIRONMENTS

    for rule in ENVIRONMENTS:
        reference = Chem.MolFromSmiles(rule.reference_smiles)
        assert reference is not None, rule.name
        assert unique_substructure_matches(reference, rule.smarts, rule.atom_centric), (
            rule.name
        )


def test_the_library_has_no_duplicate_species():
    """Two identical columns would make 'required' ambiguous for the solver."""
    from strain_homodesmotic_reaction_generator.data import (
        BALANCE_SPECIES,
        ENVIRONMENTS,
    )

    smiles = Counter(species.smiles for species in BALANCE_SPECIES)
    names = Counter(species.name for species in BALANCE_SPECIES)
    environments = Counter(rule.name for rule in ENVIRONMENTS)
    assert [s for s, c in smiles.items() if c > 1] == []
    assert [n for n, c in names.items() if c > 1] == []
    assert [n for n, c in environments.items() if c > 1] == []
