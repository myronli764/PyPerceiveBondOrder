import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import MDAnalysis as mda
import math
from typing import List, Tuple, Union
from rdkit.Geometry import Point3D

atomic_number_dict = {
    "H": 1,  # Hydrogen
    "He": 2,  # Helium
    "Li": 3,  # Lithium
    "Be": 4,  # Beryllium
    "B": 5,  # Boron
    "C": 6,  # Carbon
    "N": 7,  # Nitrogen
    "O": 8,  # Oxygen
    "F": 9,  # Fluorine
    "Ne": 10,  # Neon
    "Na": 11,  # Sodium
    "Mg": 12,  # Magnesium
    "Al": 13,  # Aluminum
    "Si": 14,  # Silicon
    "P": 15,  # Phosphorus
    "S": 16,  # Sulfur
    "Cl": 17,  # Chlorine
    "Ar": 18,  # Argon
    "K": 19,  # Potassium
    "Ca": 20,  # Calcium
    "Sc": 21,  # Scandium
    "Ti": 22,  # Titanium
    "V": 23,  # Vanadium
    "Cr": 24,  # Chromium
    "Mn": 25,  # Manganese
    "Fe": 26,  # Iron
    "Co": 27,  # Cobalt
    "Ni": 28,  # Nickel
    "Cu": 29,  # Copper
    "Zn": 30,  # Zinc
    "Ga": 31,  # Gallium
    "Ge": 32,  # Germanium
    "As": 33,  # Arsenic
    "Se": 34,  # Selenium
    "Br": 35,  # Bromine
    "Kr": 36,  # Krypton
    "Rb": 37,  # Rubidium
    "Sr": 38,  # Strontium
    "Y": 39,  # Yttrium
    "Zr": 40,  # Zirconium
    "Nb": 41,  # Niobium
    "Mo": 42,  # Molybdenum
    "Tc": 43,  # Technetium
    "Ru": 44,  # Ruthenium
    "Rh": 45,  # Rhodium
    "Pd": 46,  # Palladium
    "Ag": 47,  # Silver
    "Cd": 48,  # Cadmium
    "In": 49,  # Indium
    "Sn": 50,  # Tin
    "Sb": 51,  # Antimony
    "Te": 52,  # Tellurium
    "I": 53,  # Iodine
    "Xe": 54,  # Xenon
    "Cs": 55,  # Cesium
    "Ba": 56,  # Barium
    "La": 57,  # Lanthanum
    "Ce": 58,  # Cerium
    "Pr": 59,  # Praseodymium
    "Nd": 60,  # Neodymium
    "Pm": 61,  # Promethium
    "Sm": 62,  # Samarium
    "Eu": 63,  # Europium
    "Gd": 64,  # Gadolinium
    "Tb": 65,  # Terbium
    "Dy": 66,  # Dysprosium
    "Ho": 67,  # Holmium
    "Er": 68,  # Erbium
    "Tm": 69,  # Thulium
    "Yb": 70,  # Ytterbium
    "Lu": 71,  # Lutetium
    "Hf": 72,  # Hafnium
    "Ta": 73,  # Tantalum
    "W": 74,  # Tungsten
    "Re": 75,  # Rhenium
    "Os": 76,  # Osmium
    "Ir": 77,  # Iridium
    "Pt": 78,  # Platinum
    "Au": 79,  # Gold
    "Hg": 80,  # Mercury
    "Tl": 81,  # Thallium
    "Pb": 82,  # Lead
    "Bi": 83,  # Bismuth
    "Po": 84,  # Polonium
    "At": 85,  # Astatine
    "Rn": 86,  # Radon
    "Fr": 87,  # Francium
    "Ra": 88,  # Radium
    "Ac": 89,  # Actinium
    "Th": 90,  # Thorium
    "Pa": 91,  # Protactinium
    "U": 92,  # Uranium
    "Np": 93,  # Neptunium
    "Pu": 94,  # Plutonium
    "Am": 95,  # Americium
    "Cm": 96,  # Curium
    "Bk": 97,  # Berkelium
    "Cf": 98,  # Californium
    "Es": 99,  # Einsteinium
    "Fm": 100,  # Fermium
    "Md": 101,  # Mendelevium
    "No": 102,  # Nobelium
    "Lr": 103,  # Lawrencium
    "Rf": 104,  # Rutherfordium
    "Db": 105,  # Dubnium
    "Sg": 106,  # Seaborgium
    "Bh": 107,  # Bohrium
    "Hs": 108,  # Hassium
    "Mt": 109,  # Meitnerium
    "Ds": 110,  # Darmstadtium
    "Rg": 111,  # Roentgenium
    "Cn": 112,  # Copernicium
    "Nh": 113,  # Nihonium
    "Fl": 114,  # Flerovium
    "Mc": 115,  # Moscovium
    "Lv": 116,  # Livermorium
    "Ts": 117,  # Tennessine
    "Og": 118   # Oganesson
}
maxb = {
    1: 1,   # Hydrogen (H)
    2: 1,   # Helium (He)
    3: 1,   # Lithium (Li)
    4: 4,   # Beryllium (Be)
    5: 3,   # Boron (B)
    6: 4,   # Carbon (C)
    7: 3,   # Nitrogen (N)
    8: 2,   # Oxygen (O)
    9: 1,   # Fluorine (F)
    10: 0,  # Neon (Ne)
    11: 1,  # Sodium (Na)
    12: 2,  # Magnesium (Mg)
    13: 3,  # Aluminum (Al)
    14: 4,  # Silicon (Si)
    15: 3,  # Phosphorus (P)
    16: 2,  # Sulfur (S)
    17: 1,  # Chlorine (Cl)
    18: 0   # Argon (Ar)
}

def GetConnectedMolFromTpr(tpr,xtc):
    u = mda.Universe(tpr,xtc)
    bonds = u.bonds
    graphs = nx.Graph()
    for b in bonds:
        bonded_atoms = b.atoms
        a1 = bonded_atoms[0]
        a2 = bonded_atoms[1]
        graphs.add_edge(a1.index, a2.index)
        graphs.nodes[a1.index]['element'] = a1.name
        graphs.nodes[a2.index]['element'] = a2.name
        #print(a1.name, a2.name, a1.index, a2.index)
    mol_graphs = [graphs.subgraph(c).copy() for c in nx.connected_components(graphs)]
    aa_mols = []
    for molg in mol_graphs:
        mol = Chem.RWMol()
        node_to_idx = {}
        for node in sorted(list(molg.nodes())):
            data = molg.nodes[node]
            atomNum = atomic_number_dict[data['element']]
            atom = Chem.Atom(atomNum)
            idx = mol.AddAtom(atom)
            node_to_idx[node] = idx
        for edge in molg.edges():
            n1, n2 = edge
            mol.AddBond(node_to_idx[n1], node_to_idx[n2], Chem.BondType.SINGLE)
        aa_mols.append(mol)
    return aa_mols, mol_graphs



def read_gro_connection(gro,xtc):
    u = mda.Universe(gro,xtc)
    atoms = u.atoms
    positions = atoms.positions
    graphs = nx.Graph()
    for i, atom in enumerate(atoms):
        graphs.add_node(i, element=atom.name, position=positions[i])
    tCMol = TCMol(graphs)
    tCMol.connect_the_dots()
    mol_graphs = [tCMol.graphs.subgraph(c).copy() for c in nx.connected_components(tCMol.graphs)]
    return tCMol, mol_graphs

def GuessBondtype(mol: Union[Chem.RWMol, Chem.Mol]) -> None:
    """
    Guess bond types based on atomic numbers and connectivity.
    This is a simplified version and may not cover all cases.
    """
    return


if __name__ == '__main__':
    #aa_mols, mol_graphs = GetConnectedMolFromTpr('eq9.tpr','eq9.xtc')
    #print(len(aa_mols))
    #tCMol, mol_graphs = read_gro_connection('eq9.gro','eq9.xtc')
    #print(len(mol_graphs))
    #raise
    #mol = Chem.MolFromSmiles('CN1C=NC2=C1C(=O)N(C(=O)N2C)C(Cc3ccccc3)C')
    mol = Chem.MolFromSmiles('c1ccccc1')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    graphs = nx.Graph()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        graphs.add_node(idx, element=atom.GetSymbol(), position=mol.GetConformer().GetAtomPosition(idx))
    print(graphs.nodes(data=True))
    tCMol = TCMol(graphs)
    tCMol.connect_the_dots()
    new_mol = tCMol.TCMolToMol()
    print(new_mol.GetNumAtoms())
    Chem.SanitizeMol(new_mol)
    print(new_mol.GetNumAtoms())
    print(Chem.MolToSmiles(new_mol))
    #for bond,bond_,bond2 in zip(tCMol.bonds,mol.GetBonds(),new_mol.GetBonds()):
    #    ai, aj = bond_.GetBeginAtomIdx(), bond_.GetEndAtomIdx()
    #    if not tCMol.HasBond(tCMol.get_atom_with_idx(ai), tCMol.get_atom_with_idx(aj)):
    #        raise ValueError(f'Bond missing in TCMol: {ai}-{aj}')
    #    print('--------------------')
    #    print('TCMol      ',ai, aj, f'{bond.a1.element}-{bond.a2.element}')
    #    print('RDMol      ',ai, aj, f'{bond_.GetBeginAtom().GetSymbol()}-{bond_.GetEndAtom().GetSymbol()}')
    #    print('TCMol2RWMol',ai, aj, f'{bond2.GetBeginAtom().GetSymbol()}-{bond2.GetEndAtom().GetSymbol()}')
    #    print('--------------------')
    from BondTyper import BondTyper
    bondtyper = BondTyper(bondtyp_db='bondtyp.txt')
    bondtyper.assign_functional_group_bonds(new_mol)
    #print(Draw.MolToFile(new_mol,'test.png'))
    for bond,bond_,bond2 in zip(tCMol.bonds,mol.GetBonds(),new_mol.GetBonds()):
       ai, aj = bond_.GetBeginAtomIdx(), bond_.GetEndAtomIdx()
       if not tCMol.HasBond(tCMol.get_atom_with_idx(ai), tCMol.get_atom_with_idx(aj)):
           raise ValueError(f'Bond missing in TCMol: {ai}-{aj}')
       bond = tCMol.get_bond(ai, aj)
       bond2 = new_mol.GetBondBetweenAtoms(ai, aj)
       print('--------------------')
       print('TCMol      ',ai, aj, f'{1:>2.2f} {bond.a1.element}-{bond.a2.element}')
       print('RDMol      ',ai, aj, f'{bond_.GetBondTypeAsDouble():>2.2f} {bond_.GetBeginAtom().GetSymbol()}-{bond_.GetEndAtom().GetSymbol()}')
       print('TCMol2RWMol',ai, aj, f'{bond2.GetBondTypeAsDouble():>2.2f} {bond2.GetBeginAtom().GetSymbol()}-{bond2.GetEndAtom().GetSymbol()}')
       print('--------------------')