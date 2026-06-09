from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
    build_balance_terms,
    export_analysis,
    format_counter,
)
from strain_homodesmotic_reaction_generator import PLUGIN_DEPENDENCIES, PLUGIN_VERSION



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
    assert "primary carbon - ether oxygen - primary carbon" in names
    assert any(match.reference_smiles == "COC" for match in result.matches)
    assert "#8ab4f8" in result.equation_html


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

    export_analysis(output, result, current_file_path="my_test_molecule.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "Homodesmotic Reaction Report" in text
    assert "File,my_test_molecule.pmeprj" in text
    assert "Generated on," in text
    assert "Environment,Count,Reference SMILES,Description" in text
    assert "primary carbon - primary carbon,1,CC" in text
    assert "Left-side balancing species" in text


def test_export_analysis_writes_colored_html(tmp_path):
    # Dimethyl ether (no balance)
    result = analyze_molecule(Chem.MolFromSmiles("COC"))
    output = tmp_path / "analysis.html"

    export_analysis(output, result, current_file_path="my_test_ether.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in text
    assert "File: my_test_ether.pmeprj" in text
    assert "Generated on:" in text
    assert f"v{PLUGIN_VERSION}" in text
    assert "row-ref" in text
    assert "<td>Ref</td>" in text
    assert "<td>primary carbon - ether oxygen - primary carbon</td>" in text

    # Biphenyl (has left balance)
    result_biphenyl = analyze_molecule(Chem.MolFromSmiles("c1ccccc1-c2ccccc2"))
    output_biphenyl = tmp_path / "analysis_biphenyl.html"
    export_analysis(output_biphenyl, result_biphenyl)
    text_biphenyl = output_biphenyl.read_text(encoding="utf-8")
    assert "<td>Ref</td>" in text_biphenyl
    assert "<td>Aromatic-CH</td>" in text_biphenyl
    assert "<td>Left Balance</td>" in text_biphenyl
    assert "<td>ethane</td>" in text_biphenyl


def test_export_analysis_writes_txt(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles("CC"))
    output = tmp_path / "analysis.txt"

    export_analysis(output, result, current_file_path="my_test_ethane.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "Homodesmotic Reaction Report" in text
    assert "File: my_test_ethane.pmeprj" in text
    assert "Generated on:" in text
    assert "CC -> 1 CC" in text


def test_analyze_adamantanone_detects_carbonyls():
    mol = Chem.MolFromSmiles("O=C1C2CC3CC1CC(C2)C3")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Tertiary-Carbonyl" in names
    assert not result.is_elemental_balance


def test_analyze_methylamine_detects_primary_amine():
    mol = Chem.MolFromSmiles("CN")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Primary-Primary-Amine" in names


def test_analyze_unsolvable_molecule_triggers_elemental_balance():
    mol = Chem.MolFromSmiles("C1N2CN3CN1CN(C2)C3") # Hexamethylenetetramine
    result = analyze_molecule(mol)
    assert result.is_elemental_balance
    assert "Calculated in elemental balance mode" in result.equation_html


def test_analyze_trimethylamine_detects_tertiary_amine():
    mol = Chem.MolFromSmiles("CN(C)C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Primary-Tertiary-Amine" in names
    assert not result.is_elemental_balance


def test_analyze_butanone_detects_secondary_carbonyl():
    mol = Chem.MolFromSmiles("CCC(=O)C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Secondary-Carbonyl" in names
    assert not result.is_elemental_balance


def test_plugin_metadata():
    assert "numpy" in PLUGIN_DEPENDENCIES
    assert "scipy" in PLUGIN_DEPENDENCIES


def test_analyze_fallback_triggered_when_milp_is_missing(monkeypatch):
    import strain_homodesmotic_reaction_generator.core as core
    monkeypatch.setattr(core, "milp", None)
    mol = Chem.MolFromSmiles("COC")
    result = core.analyze_molecule(mol)
    assert result.is_elemental_balance
    assert "Calculated in elemental balance mode" in result.equation_html


def test_analyze_benzene_aromatic():
    mol = Chem.MolFromSmiles("c1ccccc1")
    result = analyze_molecule(mol)
    assert len(result.matches) == 1
    assert result.matches[0].name == "Aromatic-CH"
    assert result.matches[0].count == 6
    assert not result.is_elemental_balance
    assert "c1ccccc1 -> 1 c1ccccc1" in result.equation_text


def test_analyze_toluene_aromatic():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-C-Carbon" in names
    assert not result.is_elemental_balance
    assert "cc1ccccc1 -> 1 cc1ccccc1" in result.equation_text.lower()


def test_analyze_naphthalene_aromatic():
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-fusion-carbon" in names
    assert not result.is_elemental_balance


def test_analyze_biphenyl_aromatic():
    mol = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-C-Carbon" in names
    assert not result.is_elemental_balance
    assert "c1ccc(-c2ccccc2)cc1 + 1 cc (ethane) -> 2 cc1ccccc1" in result.equation_text.lower()


def test_analyze_five_cpp_aromatic():
    mol = Chem.MolFromSmiles("c1cc2ccc1-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc-2cc1")
    result = analyze_molecule(mol)
    assert not result.is_elemental_balance
    assert format_counter(result.target_atoms) == "C: 30, H: 20"
    assert format_counter(result.reference_atoms) == "C: 70, H: 80"
    assert format_counter(result.atom_delta) == "C: 40, H: 60"
    
    names_counts = {match.name: match.count for match in result.matches}
    assert names_counts["Aromatic-CH"] == 20
    assert names_counts["Aromatic-C-Carbon"] == 10
    
    left_names = {term.name: term.count for term in result.left_balance_terms}
    assert left_names["Benzene"] == 5
    assert left_names["ethane"] == 5
    
    right_active = [term for term in result.right_balance_terms if term.count > 0]
    assert len(right_active) == 0


def test_analyze_acetylene_sp():
    mol = Chem.MolFromSmiles("C#C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Terminal-alkyne" in names
    assert not result.is_elemental_balance
    assert "C#C -> 1 C#C" in result.equation_text


def test_analyze_propyne_sp():
    mol = Chem.MolFromSmiles("CC#C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Substituted-alkyne" in names
    assert not result.is_elemental_balance
    assert "C#CC -> 1 CC#C" in result.equation_text


def test_analyze_acetonitrile_sp():
    mol = Chem.MolFromSmiles("CC#N")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Nitrile" in names
    assert not result.is_elemental_balance
    assert "CC#N -> 1 CC#N" in result.equation_text


def test_analyze_ethene_sp2():
    mol = Chem.MolFromSmiles("C=C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Secondary-alkene" in names
    assert not result.is_elemental_balance
    assert "C=C -> 1 C=C" in result.equation_text


def test_analyze_propene_sp2():
    mol = Chem.MolFromSmiles("CC=C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Tertiary-alkene" in names
    assert not result.is_elemental_balance
    assert "C=CC -> 1 CC=C" in result.equation_text




