from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from caged_homodesmotic_reaction_builder import (  # noqa: E402
    Chem,
    analyze_molecule,
    build_balance_terms,
    export_analysis,
    format_counter,
)


pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")


def test_analyze_ethane_detects_primary_primary_bond():
    mol = Chem.MolFromSmiles("CC")

    result = analyze_molecule(mol)

    assert len(result.matches) == 1
    assert result.matches[0].name == "primary carbon - primary carbon"
    assert result.matches[0].count == 1
    assert result.matches[0].reference_smiles == "CC"
    assert result.atom_delta == Counter()
    assert "CC -> 1 CC" in result.equation_text


def test_analyze_dimethyl_ether_detects_internal_ether_oxygen():
    mol = Chem.MolFromSmiles("COC")

    result = analyze_molecule(mol)

    names = {match.name for match in result.matches}
    assert "primary carbon - ether oxygen" in names
    assert any(match.reference_smiles == "COC" for match in result.matches)
    assert "#8ab4f8" in result.equation_html
    assert "#81c995" in result.equation_html


def test_terminal_heteroatoms_are_not_treated_as_cage_ether_environments():
    methanol = Chem.MolFromSmiles("CO")
    mol = Chem.MolFromSmiles("CN")

    methanol_result = analyze_molecule(methanol)
    methylamine_result = analyze_molecule(mol)

    assert "primary carbon - ether oxygen" not in {
        match.name for match in methanol_result.matches
    }
    assert not any("amine" in match.name for match in methylamine_result.matches)


def test_build_balance_terms_uses_simple_reference_species():
    terms, unresolved = build_balance_terms(Counter({"C": 2, "H": 6}))

    assert unresolved == Counter()
    assert [(term.name, term.smiles, term.count) for term in terms] == [
        ("ethane", "CC", 1)
    ]


def test_build_balance_terms_resolves_large_hydrocarbon_balance():
    terms, unresolved = build_balance_terms(Counter({"C": 88, "H": 224}))

    atom_total = Counter()
    for term in terms:
        mol = Chem.AddHs(Chem.MolFromSmiles(term.smiles))
        for atom in mol.GetAtoms():
            atom_total[atom.GetSymbol()] += term.count

    assert unresolved == Counter()
    assert atom_total == Counter({"C": 88, "H": 224})


def test_format_counter_is_stable_and_english():
    assert format_counter(Counter({"H": 6, "C": 2})) == "C: 2, H: 6"
    assert format_counter(Counter()) == "none"


def test_export_analysis_writes_csv(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles("CC"))
    output = tmp_path / "analysis.csv"

    export_analysis(output, result)

    text = output.read_text(encoding="utf-8")
    assert "Environment,Count,Reference SMILES,Description" in text
    assert "primary carbon - primary carbon,1,CC" in text
    assert "Left-side balancing species" in text


def test_export_analysis_writes_colored_html(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles("COC"))
    output = tmp_path / "analysis.html"

    export_analysis(output, result)

    text = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in text
    assert "#8ab4f8" in text
    assert "#81c995" in text
    assert "primary carbon - ether oxygen" in text
