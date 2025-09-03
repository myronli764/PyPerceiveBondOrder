"""
bondtyper.py

Assign bond orders/types for molecules without explicit bond-order
information, driven by SMARTS patterns and optional geometric tests.
"""

import os
import logging
from typing import List, Tuple, Union, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from Meta import Atom, TCMol, Bond
import networkx as nx
import math

bondtyp_txt = """##############################################################################
#                                                                            #
#    Open Babel file: bondtyp.txt                                            #
#                                                                            #
#  Copyright (c) 2002-2005 by Geoffrey R Hutchison                           #
#  Part of the Open Babel package, under the GNU General Public License (GPL)#
#                                                                            #
# Used by bondtyper.cpp::OBBondTyper (and thus OBMol::PerceiveBondOrders()   #
#                                                                            #
# List of functional groups with double, triple (etc.) bond orders           #
# Pattern       Atom1 Atom2 Bond Order (repeat as needed)                    #
# * = Any atom (for setting non-multiple bonds)                              #
#                                                                            #
# NOTE: These are applied in order, first to last.                           #
#     So it's generally better to have a long (specifc) SMARTS first.        #
#     (all bonds must be single bonds to match)                              #
#                                                                            #
##############################################################################

# Porphyrin / phthalocyanine (3 patterns for three separate bonding motifs)
# this one has explicit bonds to all four nitrogens (e.g., metal bond or hydrogens)
# X2,X3 is needed to avoid mis-typing coproporphyrinogen
#0        1    2       3   4   5  6        7     8      9  10  11  12       13   14     15 16   17  18       19  20       21 22  23
[X2,X3]1[#6]([#7D3]2)[#6][#6][#6]2[X2,X3][#6]([#7D3]3)[#6][#6][#6]3[X2,X3][#6]([#7D3]4)[#6][#6][#6]4[X2,X3][#6]([#7D3]5)[#6][#6][#6]51	0 1 2 1 2 1 1 3 1 3 4 2 4 5 1 5 2 1 5 6 2 6 7 1 7 8 2 7 9 1 9 10 2 10 11 1 11 8 1 11 12 2 12 13 1 13 14 1 13 15 2 15 16 1 16 17 2 17 14 1 17 18 1 18 19 2 19 20 1 19 21 1 21 22 2 22 23 1 23 20 2

# this one has explicit bonds to two nitrogens (12 and 14)
#0        1    2       3   4   5  6        7    8     9  10  11  12       13   14     15 16   17  18      19  20    21 22  23
[X2,X3]1[#6]([#7D3]2)[#6][#6][#6]2[X2,X3][#6]([#7]3)[#6][#6][#6]3[X2,X3][#6]([#7D3]4)[#6][#6][#6]4[X2,X3][#6]([#7]5)[#6][#6][#6]51	0 1 2 1 2 1 1 3 1 3 4 2 4 5 1 5 2 1 5 6 2 6 7 1 7 8 2 7 9 1 9 10 2 10 11 1 11 8 1 11 12 2 12 13 1 13 14 1 13 15 2 15 16 1 16 17 2 17 14 1 17 18 1 18 19 2 19 20 1 19 21 1 21 22 2 22 23 1 23 20 2

# and this one doesn't have any explicit bonds to the nitrogens
#0        1    2     3   4   5  6        7    8     9  10  11  12      13   14   15  16  17 18        19   20    21 22  23
[X2,X3]1[#6]([#7]2)[#6][#6][#6]2[X2,X3][#6]([#7]3)[#6][#6][#6]3[X2,X3][#6]([#7]4)[#6][#6][#6]4[X2,X3][#6]([#7]5)[#6][#6][#6]51	0 1 2 1 2 1 1 3 1 3 4 2 4 5 1 5 2 1 5 6 2 6 7 1 7 8 2 7 9 1 9 10 2 10 11 1 11 8 1 11 12 2 12 13 1 13 14 1 13 15 2 15 16 1 16 17 2 17 14 1 17 18 1 18 19 2 19 20 1 19 21 1 21 22 2 22 23 1 23 20 2

# Azide
[#7D2][#7D2^1][#7D1]		0 1 2 1 2 2
# Nitro
[#8D1][#7D3^2]([#8D1])*		0 1 2 1 2 2 1 3 1

# Sulfones
[#16D4]([#8D1])([#8D1])([*!#8])([*!#8]) 	0 1 2 0 2 2 0 3 1 0 4 1
# Sulfates
[#16D4]([#8D1])([#8D1])([#8-,#8D1])([#8-,#8D1])	0 1 2 0 2 2 0 3 1 0 4 1
# Thiosulfates
[#16D4]([#16D1])([#8D1])([#8-,#8])([#8-,#8])	0 1 2 0 2 2 0 3 1 0 4 1
# Sulfoxides
[#16D3]([#8D1])([*!#8])([*!#8])			0 1 2 0 2 1 0 3 1
# Sulfite
[#16D3]([#8D1])([#8D1-])([#8D1-])		0 1 2 0 2 1 0 3 1
# Sulfur trioxide
[#16D3^2]([#8D1])([#8D1])([#8D1])                      0 1 2 0 2 2 0 3 2
# Sulfites
[#16D3]([#8D1])([#8])([#8])			0 1 2 0 2 1 0 3 1
# Disulfur monoxide
[#16D2]([#8D1])([#16D1])			0 1 2 0 2 2
# Sulfmonoxides
[#16D2]([#8D1])([*!#8])				0 1 2 0 2 1
# Sulfur dioxide
[#16D2]([#8D1])([#8D1])				0 1 2 0 2 2

#Phosphite
[#15D3]([#8D1])([#8D1])([#8D2])		0 1 2 0 2 2 0 3 1

#oxophosphane
#[#15D2]([#8D1])([#1])      0 1 2 0 2 1

#Nitrosyl Hydride
[#7D2]([#8D1])([#1])		0 1 2 0 2 1

# Phosphone
[#15D4]([#8D1])(*)(*)(*)	0 1 2 0 2 1 0 3 1 0 4 1


# Carboxylic Acid, ester, etc.
[#6D3^2]([#8D1])([#8])*		0 1 2 0 2 1 0 3 1
# Carbon dioxide
[#8D1][#6D2^1][#8D1]		0 1 2 1 2 2
# Amide C(=O)N  - no negative charge on O (2aix_u1k.sdf)
[#6D3^2]([#8D1;!-])([#7])*		0 1 2 0 2 1 0 3 1
# Seleninic acid Se(=O)OH
[#34D3^2]([#8D1])([#8])*		0 1 2 0 2 1 0 3 1
# Thioacid / Thioester C(=O)S
[#6D3^2]([#8D1])([#16])*		0 1 2 0 2 1 0 3 1
# dithioacid / dithioester C(=S)S
[#6D3^2]([#16D1])([#16])*		0 1 2 0 2 1 0 3 1
# thioamide C(=S)N
# avoid aromatics (pdb_ligands_sdf/1yry_msg.sdf)
[CD3^2]([#16D1])([N])*		0 1 2 0 2 1 0 3 1

# allene C=C=C
# (this is problematic -- need to make sure the center carbon is sp)
[#6^2][#6D2^1][#6^2]		0 1 2 1 2 2
# ene-one C=C=O
[#6^2][#6D2^1][#8D1]		0 1 2 1 2 2  

# isonitrile / isocyano
[#6D1][#7D2^1]*  0 1 3 1 2 1

# if three N are present in R-N-guanidine-ish, prefer double bond to the
#  non-terminal N (i.e. D2 if present)
[#6D3^2;!R]([#7D2;!R])([#7D1;!R])~[#7D1;!R] 0 1 2 0 2 1 0 3 1
# guanidinium and amidine -C(=NH)NH2 without hydrogens
[#6D3^2;!R]([#7D1H0;!R])([#7;!R])*	0 1 2 0 2 1 0 3 1
# and also with hydrogens
# (this can normally be figured out, but is needed to avoid matching the next SMARTS)
[#6D3^2;!R]([#7D2H1;!R])([#7;!R])*	0 1 2 0 2 1 0 3 1
# and also with more hydrogens than normal (protonated)
[#6D3^2;!R]([#7D3H2;!R])([#7;!R])*	0 1 2 0 2 1 0 3 1

# Schiff base, protonated
[#6D3^2;!R]([#1,#6])([#1,#6])[#7D3^2;!R]([#1])[#6]	0 1 1 0 2 1 0 3 2 3 4 1 3 5 1

### other potential functional groups that may (or may not) be useful to add
# imidines ( N=C/N\C=N
"""


# Stub SMARTS engine. Replace with RDKit, OpenBabel API, or your own.
class SmartsPattern:
    """
    A placeholder SMARTS matcher. In production, wrap RDKit or OpenBabel.
    """
    def __init__(self, smarts: str):
        self.smarts = smarts
        self._matches: List[List[int]] = []
        self.init_done = False

    def init(self) -> bool:
        """
        Initialize the SMARTS pattern. Return True if valid.
        """
        pat = Chem.MolFromSmarts(self.smarts)
        if not pat:
            print(f'Pattern initialization failed. The SMARTS {self.smarts} is not valid.')
            return False
        else:
            self.pattern = pat
            self.init_done = True
            return True

    def match(self, mol: Union[Chem.RWMol, Chem.Mol]) -> bool:
        """
        Attempt to match against `mol`. Populate self._matches.
        Return True if one or more matches found.
        """
        # stub: no matches
        self._matches = []
        if not self.init_done:
            if self.init():
                pass
            else:
                raise ValueError(f'Do not match the pattern with invalid SMARTS {self.smarts}.')
        _matches = mol.GetSubstructMatches(self.pattern)
        if not _matches:
            self._matches = _matches
            return True
        return False

    def get_umap_list(self) -> List[List[int]]:
        """
        Return the list of atom‐index maps for each match.
        """
        self._umap = {i:m for i,m in enumerate(self._matches)}
        return self._umap


def tokenize(line: str) -> List[str]:
    """Split a line into whitespace‐delimited tokens."""
    return line.strip().split()


o2t = {
    1.0: Chem.rdchem.BondType.SINGLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
    2.0: Chem.rdchem.BondType.DOUBLE,
    3.0: Chem.rdchem.BondType.TRIPLE,
    0.0: Chem.rdchem.BondType.UNSPECIFIED,
}



class BondTyper:
    """
    Python port of OpenBabel's OBBondTyper.
    """
    def __init__(self, bondtyp_db = None,):
        self._init_done = False
        self.bondtyp_db = bondtyp_db
        # Each entry: (SmartsPattern, [i1, j1, order1, i2, j2, order2, ...])
        self._fgbonds: List[Tuple[SmartsPattern, List[int]]] = []
        self.logger = logging.getLogger(__name__)

    def _init(self) -> None:
        """Read the data file and parse each non‐comment line."""
        #path = os.path.join(self.data_dir, self.subdir, self.filename)
        bondtyp_txt = open(self.bondtyp_db, 'r').readlines()
        try:
            for lineno, line in enumerate(bondtyp_txt):
                self.parse_line(line, lineno)
            self._init_done = True
        except FileNotFoundError:
            self.logger.error(f"BondTyp database not found: {bondtyp_txt}")
            self._init_done = True  # prevent retry

    def average_angle(self,mol: Union[Chem.RWMol,Chem.Mol],a1: [Union[Chem.Atom]]) -> float:
        pos = mol.GetConformer().GetPositions()
        nbrs = a1.GetNeighbors()
        x0 = pos[a1.GetIdx()]
        av_angs = 0
        angs = []
        for nbr_i in nbrs:
            for nbr_j in nbrs:
                if nbr_i.GetIdx() == nbr_j.GetIdx():
                    continue
                xi = pos[nbr_i.GetIdx()]
                xj = pos[nbr_j.GetIdx()]
                xi = xi - x0
                xj = xj - x0
                ang = xi.dot(xj)/np.linalg.norm(xi)/np.linalg.norm(xj)
                angs.append(ang)
        if len(angs) != 0:
            av_angs = float(np.arccos(np.mean(angs)))/np.pi*180
        return av_angs#float(np.mean(angs))

    def d2_distance(self,mol, a1, a2) -> float:
        pos = mol.GetConformer().GetPositions()
        xi = pos[a1.GetIdx()]
        xj = pos[a2.GetIdx()]
        return np.linalg.norm(xi-xj)

    def has_double_bond(self,a1) -> bool:
        bonds = a1.GetBonds()
        for bond in bonds:
            if bond.GetBondTypeAsDouble() == 2:
                return True
        return False

    def parse_line(self, buffer: str, lineno: int = 0) -> None:
        """
        Parse one line of bondtyp.txt.
        On success, store a (SmartsPattern, assignments) entry.
        """
        if buffer == '\n' or buffer.lstrip().startswith("#"):
            return
        tokens = tokenize(buffer)
        if len(tokens) < 4:
            return
        if (len(tokens) - 1) % 3 != 0:
            self.logger.warning(
                f"Line {lineno}: wrong token count ({len(tokens)}) in: {buffer.strip()}"
            )
            return
        smarts = tokens[0]
        patt = SmartsPattern(smarts)
        if not patt.init():
            return
        # parse the integer triplets
        try:
            ints = [int(t) for t in tokens[1:]]
        except ValueError:
            self.logger.warning(f"Line {lineno}: non‐integer token in {buffer!r}")
            return
        self._fgbonds.append((patt, ints))

    def _sanitize(self,mol: Union[Chem.RWMol, Chem.Mol], tcmol: TCMol) -> None:
        Chem.SanitizeMol(mol)
        for atom in mol.GetAtoms():
            a_tc = tcmol.get_atom_with_idx(atom.GetIdx())
            if a_tc:
                a_tc.set_aromatic(atom.GetIsAromatic())


    def assign_functional_group_bonds(self, mol: Union[Chem.RWMol, Chem.Mol], tcmol: TCMol) -> None:
        """
        For each SMARTS rule and each match, set bond orders on `mol`.
        Also apply special rules for carbonyls, thiones, etc., with geometry.
        `mol` must implement:
            - get_atom(idx) -> Atom or None
            - Atom.get_bond(other) -> Bond or None
            - Bond.set_order(n)
            - Atom.distance_to(other) -> float
            - Atom.average_bond_angle() -> float
            - Atom.has_double_bond() -> bool
            - Atom.set_formal_charge(int)
            - Atom.atomic_num
        """
        if not self._init_done:
            self._init()

        # 1) Generic functional groups from _fgbonds
        for patt, assigns in self._fgbonds:
            if patt.match(mol):
                for umap in patt.get_umap_list():
                    for k in range(0, len(assigns), 3):
                        i1, i2, order = assigns[k:k+3]
                        a1 = mol.GetAtomWithIdx(umap[i1])
                        a2 = mol.GetAtomWithIdx(umap[i2])
                        a1_tcmol = tcmol.get_atom_with_idx(umap[i1])
                        a2_tcmol = tcmol.get_atom_with_idx(umap[i2])
                        if not a1 or not a2:
                            continue
                        b = mol.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
                        b_tc = tcmol.get_bond(umap[i1], umap[i2])
                        if b:
                            b.SetBondType(o2t[order])
                            b_tc.set_order(order)

        # 2) Hard‐coded geometric rules
        def apply_rule(smarts, idx_i, idx_j, min_angle, max_angle, max_dist, set_order=None, set_charge=None):
            patt = SmartsPattern(smarts)
            if not patt.init() or not patt.match(mol):
                return
            for umap in patt.get_umap_list():
                a1 = mol.get_atom(umap[idx_i])
                a2 = mol.get_atom(umap[idx_j])
                a1_tcmol = tcmol.get_atom_with_idx(umap[idx_i])
                a2_tcmol = tcmol.get_atom_with_idx(umap[idx_j])
                if not a1 or not a2:
                    continue
                angle = self.average_angle(mol,a2)
                dist = self.d2_distance(mol,a1,a2)
                if min_angle < angle < max_angle and dist < max_dist:
                    b = mol.GetBondBetweenAtoms(a1.GetIdx(),a2.GetIdx())
                    b_tc = tcmol.get_bond(umap[idx_i], umap[idx_j])
                    if set_order and b:
                        if set_order == 2 and self.has_double_bond(a1):
                            continue
                        b.SetBondType(o2t[set_order])
                        b_tc.set_order(set_order)
                    if set_charge:
                        a1.set_formal_charge(set_charge[0])
                        a2.set_formal_charge(set_charge[1])
                        a1_tcmol.set_formal_charge(set_charge[0])
                        a2_tcmol.set_formal_charge(set_charge[1])


        # carbonyl C=O
        apply_rule("[#8D1;!-][#6](*)(*)", idx_i=0, idx_j=1,
                   min_angle=115, max_angle=150, max_dist=1.28, set_order=2)
        # thione C=S
        apply_rule("[#16D1][#6](*)(*)", idx_i=0, idx_j=1,
                   min_angle=115, max_angle=150, max_dist=1.72, set_order=2)
        # oxime C=N–O
        apply_rule("[#6D3][#7D2][#8D2]", idx_i=0, idx_j=1,
                   min_angle=110, max_angle=150, max_dist=1.40, set_order=2)
        # pyridine N‐oxide
        apply_rule("[#8D1][#7D3r6]", idx_i=0, idx_j=1,
                   min_angle=110, max_angle=150, max_dist=1.35,
                   set_charge=(-1, +1))

        # Isocyanate/isothiocyanate special handling
        # We do this inline because it needs two bonds and atomic‐number check.
        iso = SmartsPattern("[#8,#16;D1][#6D2][#7D2]")
        if iso.init() and iso.match(mol):
            for umap in iso.get_umap_list():
                a1 = mol.GetAtomWithIdx(umap[0])
                a2 = mol.GetAtomWithIdx(umap[1])
                a3 = mol.GetAtomWithIdx(umap[2])
                a1_tcmol = tcmol.get_atom_with_idx(umap[0])
                a2_tcmol = tcmol.get_atom_with_idx(umap[1])
                a3_tcmol = tcmol.get_atom_with_idx(umap[2])
                if not (a1 and a2 and a3):
                    continue
                angle = self.average_angle(mol,a2)#a2.average_bond_angle()
                d12 = self.d2_distance(mol,a2,a1)#a1.distance_to(a2)
                d23 = self.d2_distance(mol,a2,a3)#a2.distance_to(a3)
                if a1.atomic_num == 8:
                    ok12 = (d12 < 1.28)
                else:
                    ok12 = (d12 < 1.72)
                if angle > 150 and ok12 and d23 < 1.34:
                    b12 = mol.GetBondBetweenAtoms(a2.GetIdx(),a1.GetIdx())#a1.get_bond(a2)
                    b23 = mol.GetBondBetweenAtoms(a2.GetIdx(),a3.GetIdx())#a2.get_bond(a3)
                    b12_tc = tcmol.get_bond(umap[1], umap[0])
                    b23_tc = tcmol.get_bond(umap[1], umap[2])
                    if b12:
                        b12.SetBondType(o2t[2])
                        b12_tc.set_order(2)
                    if b23:
                        b23.SetBondType(o2t[2])
                        b23_tc.set_order(2)

    def perceive_bond_order(self, mol: Union[Chem.RWMol, Chem.Mol], tcmol: TCMol) -> None:
        """
        Main driver for bond‐order perception.
        """
        if not tcmol.atoms:
            return

        # Pass 1: Hybridization from average angles
        for atom in tcmol.atoms:
            rdatom = mol.GetAtomWithIdx(atom.idx)
            ang = self.average_angle(mol, rdatom)
            #print(atom.element, atom.hybridization)
            if ang > 155.0:
                atom.set_hybridization(1)
            elif ang > 115.0:
                atom.set_hybridization(2)
            #else:
            #    atom.set_hybridization(3)
            # special imine/azete cases
            if (atom.atomic_num == 7
               and atom.explicit_hydrogen_count() == 1
               and atom.explicit_degree() == 2
               and ang > 109.5):
                atom.set_hybridization(2)
            elif (atom.atomic_num == 7
                  and atom.explicit_degree() == 2
                  and atom.is_in_ring()):
                atom.set_hybridization(2)
            #print(atom.element, atom.hybridization)


        # Pass 2: Ring torsion → sp2 in small planar rings
        rings = tcmol.get_sssr()
        for path in rings:
            size = len(path)
            tors = tcmol._average_ring_torsion(path)
            #print(f'Ring size {size}, torsion {tors:.1f}')
            if (size == 5 and tors <= 7.5) or (size == 6 and tors <= 12.0):
                for idx in path:
                    atom = tcmol.get_atom_with_idx(idx)
                    deg = atom.explicit_degree()
                    if (size == 5 and deg == 2) or (size == 6 and deg in (2,3)):
                        atom.set_hybridization(2)

        # Pass 3: Antialiasing
        for atom in tcmol.atoms:
            h = atom.hybridization or 3
            #print(atom.element,atom.hybridization, h)
            if h in (1,2):
                open_nbr = False
                for nbr_bond in atom.bonds:
                    nbr = nbr_bond.other(atom)
                    h2 = nbr.hybridization or 3
                    if h2 < 3 or nbr.explicit_degree() == 1:
                        #print('fuck')
                        open_nbr = True
                        break
                if not open_nbr:
                    #print('fuck again')
                    atom.set_hybridization(h+1)
                #print(atom.element, atom.hybridization)

        # Pass 4: Functional group bonds
        self.assign_functional_group_bonds(mol, tcmol)

        # Pass 5: Aromatic ring typing & Kekulize (stub)
        needs_sanitiza = False
        for path in rings:
            size = len(path)
            if size in (5,6,7):
                # check if any atom in path has multiple bond or non‐sp2
                typed = any(
                    (tcmol.atoms[idx].has_non_single_bond()
                     or (tcmol.atoms[idx].hybridization != 2))
                    for idx in path
                )
                if not typed:
                    # mark all bonds aromatic
                    edges = []
                    for i in path:
                        for j in path:
                            if i == j:
                                continue
                            if (i, j) in edges or (j, i) in edges:
                                continue
                            if tcmol.HasBond(tcmol.get_atom_with_idx(i), tcmol.get_atom_with_idx(j)):
                                edges.append((i, j))
                    for i,j in edges:
                        b = mol.GetBondBetweenAtoms(i, j)
                        b_tc = tcmol.get_bond(i, j)
                        if b:
                            b.SetBondType(o2t[1.5])
                            b_tc.set_order(1.5)
                            needs_sanitiza = True
        #return
        if needs_sanitiza:
            self._sanitize(mol, tcmol)
        #raise
        # Pass 6: Electronegativity‐driven multiple bond promotion
        # Sort atoms by (EN*1e6 + shortestBond)
        #en_map = {1:2.20,6:2.55,7:3.04,8:3.44,16:2.58}

        def atom_score(atom: Atom) -> float:
            sb = min((b.length() for b in atom.bonds
                      if b.other(atom).atomic_num != 1), default=1e5)
            en = atom.en #en_map.get(atom.atomic_num, 2.5)
            return en*1e6 + sb

        for atom in sorted(tcmol.atoms, key=atom_score):
            deg = atom.explicit_degree()
            hyb = atom.hybridization or 3
            # candidate triple bond (sp or terminal)
            if (hyb == 1 or deg == 1) and deg+2 <= atom.max_bonds():
                if atom.has_non_single_bond(): continue
                best: Optional[Bond] = None
                best_en = 0.0
                best_len = 1e5
                for b in atom.bonds:
                    nbr = b.other(atom)
                    hy2 = nbr.hybridization or 3
                    if ((hy2 == 1 or nbr.explicit_degree() == 1)
                        and nbr.explicit_degree()+2 <= nbr.max_bonds()
                        and not nbr.has_non_single_bond()):
                        # test bond length
                        L = b.length()
                        if ((atom.explicit_degree()==1 or nbr.explicit_degree()==1)
                            and L > 0.9*(atom.covalent_radius()+nbr.covalent_radius())):
                            continue
                        en2 = nbr.en #en_map.get(nbr.atomic_num, 2.5)
                        if en2 > best_en or (math.isclose(en2, best_en) and L<best_len):
                            best_en, best_len, best = en2, L, b
                if best:
                    best.set_order(3)
                    rdbond = mol.GetBondBetweenAtoms(atom.idx, best.other(atom).idx)
                    rdbond.SetBondType(o2t[3])
            # candidate double bond (sp2 or terminal)
            elif (hyb == 2 or deg == 1) and deg+1 <= atom.max_bonds():
                if atom.has_non_single_bond(): continue
                best: Optional[Bond] = None
                best_en = 0.0
                best_len = 1e5
                for b in atom.bonds:
                    nbr = b.other(atom)
                    hy2 = nbr.hybridization or 3
                    if ((hy2 == 2 or nbr.explicit_degree() == 1)
                        and nbr.explicit_degree()+1 <= nbr.max_bonds()
                        and tcmol._is_double_geometry(atom, nbr)):
                        # ring‐sulfur special case
                        if (atom.is_in_ring() and atom.atomic_num==16):
                            continue
                        L = b.length()
                        if ((atom.explicit_degree()==1 or nbr.explicit_degree()==1)
                            and L > 0.93*(atom.covalent_radius()+nbr.covalent_radius())):
                            continue
                        en2 = nbr.en #en_map.get(nbr.atomic_num, 2.5)
                        better = False
                        if en2 > best_en:
                            better = True
                        elif math.isclose(en2, best_en) and (
                            (best and (not atom.is_in_ring() or not best.other(atom).is_in_ring() or nbr.is_in_ring()))
                        ):
                            better = True
                        if better:
                            best_en, best_len, best = en2, L, b
                if best:
                    best.set_order(2)
                    rdbond = mol.GetBondBetweenAtoms(atom.idx, best.other(atom).idx)
                    rdbond.SetBondType(o2t[2])
        # Final sanitize
        self._sanitize(mol, tcmol)




if __name__ == "__main__":
    bondtyper = BondTyper(bondtyp_db='bondtyp.txt')
    mol = Chem.MolFromSmiles('CC(C)(c1ccc(Oc2ccc3c(c2)C(=O)OC3=O)cc1)c1ccc(Oc2ccc3c(c2)C(=O)OC3=O)cc1')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    graphs = nx.Graph()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        graphs.add_node(idx, element=atom.GetSymbol(), position=mol.GetConformer().GetAtomPosition(idx))
    tCMol = TCMol(graphs)
    tCMol.connect_the_dots()
    new_mol = tCMol.TCMolToMol()
    Chem.Kekulize(new_mol)
    print(Chem.MolToSmiles(new_mol))
    bondtyper.perceive_bond_order(new_mol, tCMol)
    Chem.SanitizeMol(new_mol)
    print(Chem.MolToSmiles(new_mol))


