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

en_map = {
    "H": 2.300, "He": 4.160,
    "Li": 0.912, "Be": 1.576,
    "B": 2.051, "C": 2.544,
    "N": 3.066, "O": 3.610,
    "F": 4.193, "Ne": 4.787,
    "Na": 0.869, "Mg": 1.293,
    "Al": 1.613, "Si": 1.916,
    "P": 2.253, "S": 2.589,
    "Cl": 2.869, "Ar": 3.242,
    "K": 0.734, "Ca": 1.034,
    "Sc": 1.19, "Ti": 1.38,
    "V": 1.53, "Cr": 1.65,
    "Mn": 1.75, "Fe": 1.80,
    "Co": 1.84, "Ni": 1.88,
    "Cu": 1.85, "Zn": 1.588,
    "Ga": 1.756, "Ge": 1.994,
    "As": 2.211, "Se": 2.424,
    "Br": 2.685, "Kr": 2.966,
    "Rb": 0.706, "Sr": 0.963,
    "Y": 1.12, "Zr": 1.32,
    "Nb": 1.41, "Mo": 1.47,
    "Tc": 1.51, "Ru": 1.54,
    "Rh": 1.56, "Pd": 1.58,
    "Ag": 1.87, "Cd": 1.521,
    "In": 1.656, "Sn": 1.824,
    "Sb": 1.984, "Te": 2.158,
    "I": 2.359, "Xe": 2.582,
    "Cs": 0.659, "Ba": 0.881,
    "Lu": 1.09, "Hf": 1.16,
    "Ta": 1.34, "W": 1.47,
    "Re": 1.60, "Os": 1.65,
    "Ir": 1.68, "Pt": 1.72,
    "Au": 1.92, "Hg": 1.765,
    "Tl": 1.789, "Pb": 1.854,
    "Bi": 2.01, "Po": 2.19,
    "At": 2.39, "Rn": 2.60,
    "Fr": 0.67, "Ra": 0.89
}


class Atom:
    """
    Minimal Atom interface assumed by connect_the_dots().
    """
    def __init__(self, idx: int, coord: Tuple[float, float, float],
                 atomic_num: int, element: str, hybridization: int = 0,
                 formal_charge: int = 0, explicit_bonds = List['Bond'],
                 is_aromatic: bool = False):
        self.idx = idx                  # 1-based
        self.coord = coord              # (x, y, z)
        self.atomic_num = atomic_num
        self.element = element          # e.g. "C"
        self.formal_charge = formal_charge
        self.bonds = explicit_bonds   # explicit bonds
        self.hybridization = hybridization
        self.is_aromatic = is_aromatic
        self.en = en_map[element] if element in en_map else 1.0

    def explicit_degree(self) -> int:
        return len(self.bonds)

    def explicit_valence(self) -> int:
        # Here we treat all bonds as single for degree==valence
        return len(self.bonds)

    def add_bond(self, other: 'Atom') -> 'Bond':
        bond = Bond(self, other)
        self.bonds.append(bond)
        other.bonds.append(bond)
        return bond

    def delete_bond(self, bond: 'Bond') -> None:
        if bond in self.bonds:
            self.bonds.remove(bond)
        nbr = bond.other(self)
        if bond in nbr.bonds:
            nbr.bonds.remove(bond)

    def is_connected(self, other: 'Atom') -> bool:
        return any(b.other(self) is other for b in self.bonds)

    def neighbors(self) -> List['Atom']:
        return [b.other(self) for b in self.bonds]

    def covalent_radius(self) -> float:
        # Stub: real code would dispatch by atomic_num
        radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}
        return radii.get(self.atomic_num, 0.75)

    def max_bonds(self) -> int:
        # Stub: max single bonds by atomic number
        maxb = {1: 1, 6: 4, 7: 3, 8: 2, 16: 2}
        return maxb.get(self.atomic_num, 4)

    def explicit_hydrogen_count(self) -> int:
        return sum(1 for b in self.bonds
                   if b.other(self).atomic_num == 1)

    def is_in_ring(self) -> bool:
        """
        Stub: Real code would check ring membership.
        """
        return False

    def has_non_single_bond(self) -> bool:
        return any(b.order > 1 for b in self.bonds)

    def distance_sq(self, other: 'Atom') -> float:
        dx = self.coord[0] - other.coord[0]
        dy = self.coord[1] - other.coord[1]
        dz = self.coord[2] - other.coord[2]
        return dx*dx + dy*dy + dz*dz

    def smallest_bond_angle(self) -> float:
        # Stub: if fewer than 2 neighbors, return large angle
        neigh = self.neighbors()
        if len(neigh) < 2:
            return 180.0
        # compute all angles between neighbor pairs
        def vec(a, b):
            return (b.coord[0]-a.coord[0],
                    b.coord[1]-a.coord[1],
                    b.coord[2]-a.coord[2])
        min_angle = 180.0
        for i in range(len(neigh)):
            for j in range(i+1, len(neigh)):
                v1 = vec(self, neigh[i])
                v2 = vec(self, neigh[j])
                dot = sum(u*v for u,v in zip(v1, v2))
                norm1 = math.sqrt(sum(u*u for u in v1))
                norm2 = math.sqrt(sum(u*u for u in v2))
                if norm1*norm2 == 0:
                    continue
                angle = math.degrees(math.acos(dot/(norm1*norm2)))
                min_angle = min(min_angle, angle)
        return min_angle


    def set_hybridization(self, hyb: int) -> None:
        self.hybridization = hyb

    def set_formal_charge(self, charge: int) -> None:
        self.formal_charge = charge

    def set_aromatic(self, aromatic: bool) -> None:
        self.is_aromatic = aromatic

class Bond:
    """
    Minimal Bond interface.
    """
    def __init__(self, a1: Atom, a2: Atom, order: int = 1):
        self.a1 = a1
        self.a2 = a2
        self.order = order

    def other(self, atom: Atom) -> Atom:
        return self.a2 if atom is self.a1 else self.a1

    def length_sq(self) -> float:
        return self.a1.distance_sq(self.a2)

    def set_order(self, o: int) -> None:
        self.order = o

    def length(self) -> float:
        return math.sqrt(self.length_sq())

class UnitCell:
    """
    Stub for periodic boundary minimum-image.
    """
    def __init__(self, vectors: Tuple[Tuple[float, float, float],
                                     Tuple[float, float, float],
                                     Tuple[float, float, float]]):
        self.v = vectors
    def minimum_image(self, disp: Tuple[float, float, float]) \
            -> Tuple[float, float, float]:
        # Very simplified: no real lattice math
        return disp

def pbc(x, box):
    return x - box * np.rint(x / box)


class TCMol:
    """
    Molecule container with atoms and optional periodicity.
    """
    def __init__(self, graphs,
                 is_periodic: bool = False,
                 box: np.ndarray = None):
        atoms = []
        for n in sorted(list(graphs.nodes)):
            atom = Atom(idx=n,
                        coord=graphs.nodes[n]['position'],
                        atomic_num=atomic_number_dict[graphs.nodes[n]['element']],
                        element=graphs.nodes[n]['element'],
                        formal_charge=0,
                        explicit_bonds=[])
            atoms.append(atom)
        #for a in atoms:
        #    print(a.element, a.idx)
        for n in sorted(list(graphs.nodes)):
            atom = atoms[n]
            exist_bonds = []
            for nbr in graphs.neighbors(n):
                exist_bonds.append(Bond(atoms[n], atoms[nbr]))
            atom.bonds = exist_bonds
        self.atoms = atoms
        self.is_periodic = is_periodic
        self.box = box
        self.graphs = graphs
        self.bonds: List['Bond'] = []
        self.bonds_hash = {}
        for i,j in self.graphs.edges():
            a1 = self.get_atom_with_idx(i)
            a2 = self.get_atom_with_idx(j)
            bond = Bond(a1, a2)
            self.bonds.append(bond)
            self.bonds_hash[(i, j)] = bond
            #self.bonds_hash[(i, j)] = bond
        self._sssr = None  # cached SSSR rings

    def get_atom_with_idx(self, idx: int) -> Atom:
        #for atom in self.atoms:
        if idx < len(self.atoms):
            return self.atoms[idx]
        raise ValueError(f"Atom with idx {idx} out of range")

    def connect_the_dots(self) -> None:
        """
        Reconstruct bonds from 3D coords using covalent radii and
        valence rules, mirroring OBMol::ConnectTheDots().
        """
        if not self.atoms:
            return

        # 1) Prepare working arrays
        bond_count = [atom.explicit_degree() for atom in self.atoms]
        # filter atoms that still can accept bonds
        filt: List[Atom] = []
        for atom in self.atoms:
            if atom.explicit_valence() >= atom.max_bonds():
                continue
            # special-case neutral N
            if (atom.atomic_num == 7 and atom.formal_charge == 0
                    and atom.explicit_valence() >= 3):
                continue
            filt.append(atom)

        # sort by z
        filt.sort(key=lambda a: a.coord[2])
        n = len(filt)
        # precompute radii and track max radius
        radii = [atom.covalent_radius() for atom in filt]
        maxrad = max(radii) if radii else 0.0

        # 2) Add candidate bonds
        for i in range(n):
            ai = filt[i]
            ri = radii[i]
            maxcutoff_sq = (ri + maxrad + 0.45)**2
            for j in range(i+1, n):
                aj = filt[j]
                dz = (aj.coord[2] - ai.coord[2])**2
                if dz > maxcutoff_sq:
                    break  # all further j will have larger z-diff

                cutoff_sq = (ri + radii[j] + 0.45)**2
                # compute full distance^2
                if self.is_periodic and self.box:
                    # build displacement vector
                    disp = (aj.coord[0]-ai.coord[0],
                            aj.coord[1]-ai.coord[1],
                            aj.coord[2]-ai.coord[2])
                    dx, dy, dz = pbc(np.array(disp), self.box)
                    d2 = dx*dx + dy*dy + dz*dz
                else:
                    # non-periodic
                    dx = aj.coord[0]-ai.coord[0]
                    dy = aj.coord[1]-ai.coord[1]
                    d2 = dx*dx + dy*dy + dz

                if d2 < 0.16 or d2 > cutoff_sq:
                    continue

                if not ai.is_connected(aj):
                    # assume validAdditionalBond always true in this stub
                    ai.add_bond(aj)
                    self.graphs.add_edge(ai.idx, aj.idx)

        # 3) Prune over-bonded atoms or bad angles
        for atom in self.atoms:
            # recalc while invalid
            while (atom.explicit_valence() > atom.max_bonds()
                   or atom.smallest_bond_angle() < 45.0):
                # first try hydrogen–hydrogen bonds if atom is H
                to_delete = None
                if atom.atomic_num == 1:
                    for b in list(atom.bonds):
                        other = b.other(atom)
                        if other.atomic_num == 1:
                            to_delete = b
                            break
                # if no H–H, pick the longest bond
                if not to_delete:
                    max_bond = None
                    max_len = -1.0
                    for b in atom.bonds:
                        l2 = b.length_sq()
                        if l2 > max_len:
                            max_len = l2
                            max_bond = b
                    to_delete = max_bond
                if to_delete is None:
                    break
                atom.delete_bond(to_delete)
                self.graphs.remove_edge(to_delete.a1.idx, to_delete.a2.idx)

        for i,j in self.graphs.edges():
            a1 = self.get_atom_with_idx(i)
            a2 = self.get_atom_with_idx(j)
            bond = Bond(a1, a2)
            self.bonds.append(bond)
            if (i, j) not in self.bonds_hash and (j, i) not in self.bonds_hash:
                self.bonds_hash[(i, j)] = bond

    def HasBond(self, a1: Atom, a2: Atom) -> bool:
        if a1.is_connected(a2):
            return True
        return False

    def get_bond(self, i: int, j: int) -> Bond:
        #a1 = self.get_atom_with_idx(i)
        #a2 = self.get_atom_with_idx(j)
        if (i, j) in self.bonds_hash:
            return self.bonds_hash[(i, j)]
        elif (j, i) in self.bonds_hash:
            return self.bonds_hash[(j, i)]
        else:
            raise ValueError(f'No bond between {i} and {j}')

    def get_sssr(self) -> List[List[int]]:
        """
        Return list of ring atom‐index paths (1-based).
        Stub: return cached or empty.
        """
        if self._sssr is None:
            self._sssr = nx.cycle_basis(self.graphs)
        return self._sssr

    def _average_ring_torsion(self, path: List[int]) -> float:
        """
        Average absolute torsion over a ring path.
        Stub: return 0.0 to force sp2.
        """
        #print(path)
        edges = []
        for i in path:
            for j in path:
                if i == j:
                    continue
                if (i, j) in edges or (j, i) in edges:
                    continue
                if self.HasBond(self.get_atom_with_idx(i), self.get_atom_with_idx(j)):
                    edges.append((i, j))
        torsions = []
        for i, j in edges:
            ai = self.get_atom_with_idx(i)
            aj = self.get_atom_with_idx(j)
            for ak in ai.neighbors():
                if ak.idx in path and ak.idx != j:
                    for al in aj.neighbors():
                        if al.idx in path and al.idx != i and al.idx != ak.idx:
                            # compute torsion angle ak-ai-aj-al
                            p1 = np.array(ak.coord)
                            p2 = np.array(ai.coord)
                            p3 = np.array(aj.coord)
                            p4 = np.array(al.coord)
                            b1 = p2 - p1
                            b2 = p3 - p2
                            b3 = p4 - p3
                            b2 /= np.linalg.norm(b2)
                            v = b1 - np.dot(b1, b2) * b2
                            w = b3 - np.dot(b3, b2) * b2
                            x = np.dot(v, -w)
                            #print(v, w)
                            cos_angle = x / (np.linalg.norm(v) * np.linalg.norm(w))
                            #print(cos_angle)
                            angle = np.arccos(round(cos_angle,5))/np.pi*180.0
                            #y = np.dot(np.cross(b2, v), w)
                            #angle = math.degrees(math.atan2(y, x))

                            torsions.append(abs(angle))
        #print(torsions)
        if torsions:
            return sum(torsions) / len(torsions)
        else:
            return 0.0

    def _is_double_geometry(self, a1: Atom, a2: Atom) -> bool:
        """
        Stub for GetBond(a1,a2)->IsDoubleBondGeometry()
        """
        return True

    def TCMolToMol(self) -> Chem.RWMol:
        mol = Chem.RWMol()
        node_to_idx = {}
        for atom in self.atoms:
            atomNum = atom.atomic_num
            rdkit_atom = Chem.Atom(atomNum)
            idx = mol.AddAtom(rdkit_atom)
            node_to_idx[atom.idx] = idx
        for bond in self.bonds:
            n1 = bond.a1.idx
            n2 = bond.a2.idx
            mol.AddBond(node_to_idx[n1], node_to_idx[n2], Chem.BondType.SINGLE)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in self.atoms:
            conf.SetAtomPosition(node_to_idx[atom.idx], Point3D(*atom.coord))
        mol.AddConformer(conf,assignId=True)
        Chem.Kekulize(mol)
        return mol.GetMol()


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CC(C)(c1ccc(Oc2ccc3c(c2)C(=O)OC3=O)cc1)c1ccc(Oc2ccc3c(c2)C(=O)OC3=O)cc1')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    graphs = nx.Graph()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        graphs.add_node(idx, element=atom.GetSymbol(), position=mol.GetConformer().GetAtomPosition(idx))
    #for bond in mol.GetBonds():
    #    a1 = bond.GetBeginAtomIdx()
    #    a2 = bond.GetEndAtomIdx()
    #    graphs.add_edge(a1, a2)
    tCMol = TCMol(graphs)
    tCMol.connect_the_dots()
    new_mol = tCMol.TCMolToMol()
    # print(Chem.MolToSmiles(new_mol))
    from BondTyper import BondTyper

    bondtyper = BondTyper('bondtyp.txt')
    #bondtyper.assign_functional_group_bonds(new_mol,tCMol)
    bondtyper.perceive_bond_order(new_mol, tCMol)
    for bond, bond_, bond2 in zip(tCMol.bonds, mol.GetBonds(), new_mol.GetBonds()):
        ai, aj = bond_.GetBeginAtomIdx(), bond_.GetEndAtomIdx()
        if not tCMol.HasBond(tCMol.get_atom_with_idx(ai), tCMol.get_atom_with_idx(aj)):
            raise ValueError(f'Bond missing in TCMol: {ai}-{aj}')
        bond = tCMol.get_bond(ai, aj)
        bond2 = new_mol.GetBondBetweenAtoms(ai, aj)
        print('--------------------')
        print('TCMol      ', ai, aj, f'{1:>2.2f} {bond.a1.element}-{bond.a2.element}')
        print('RDMol      ', ai, aj,
              f'{bond_.GetBondTypeAsDouble():>2.2f} {bond_.GetBeginAtom().GetSymbol()}-{bond_.GetEndAtom().GetSymbol()}')
        print('TCMol2RWMol', ai, aj,
              f'{bond2.GetBondTypeAsDouble():>2.2f} {bond2.GetBeginAtom().GetSymbol()}-{bond2.GetEndAtom().GetSymbol()}')
        print('--------------------')
        if bond_.GetBondTypeAsDouble() != bond2.GetBondTypeAsDouble():
            raise ValueError(f'Bond order mismatch: {ai}-{aj} {bond_.GetBondTypeAsDouble()} != {bond2.GetBondTypeAsDouble()}')