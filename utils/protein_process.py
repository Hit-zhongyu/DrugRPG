import os
import numpy as np
from rdkit import Chem
import rdkit

# ATOM_MAP = {
#     'Backbone':{'C':'6', 'N':'7', 'O':'8','S':'16'},
#     'Base':{'H':'1','C':'2', 'N':'3', 'O':'4','S':'5'},
#     'Loc':{'A':'5', 'B':'10', 'G':'15','D':'20', 'E':'25', 'Z':'30', 'H':'35'}, # "α,β,γ,δ,ϵ,ζ,η
#     'Num':{'1':'35', '2':'70', '3':'110'}
#     }
# ATOM_MAP = {
#     'Backbone':{'H':'1', 'C':'2', 'N':'3', 'O':'4','S':'5'},
#     # 'Base':{'H':'1','C':'2', 'N':'3', 'O':'4','S':'5'},
#     'Loc':{'A':'1', 'B':'2', 'G':'3','D':'4', 'E':'5', 'Z':'6', 'H':'7',"X":"8"}, # "α,β,γ,δ,ϵ,ζ,η
#     'Num':{'1':1, '2':2, '3':3,'T':4}
    # }
AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
# HYDROPATHY = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9,
#                "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, 
#                "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
HYDROPATHY = {'#': 0.5, 'I': 1, 'V': 0.9667, 'L': 0.9222, 'F': 0.8111, 'C': 0.7778,
                'M': 0.7111, 'A': 0.7000, 'W': 0.4000, 'G': 0.4556, 'T': 0.4222, 'S': 0.4111,
                'Y': 0.3556, 'P': 0.3222, 'H': 0.1444, 'N': 0.1111, 'D': 0.1111, 'Q': 0.1111,
                'E': 0.1111, 'K': 0.0667, 'R': 0.0000}
# VOLUME = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, 
#           "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7,
#             "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
VOLUME = {'#': 0, 'G': 0.2638, 'A': 0.3889, 'S': 0.3907, 'C': 0.4763, 'D': 0.4877, 'P': 0.4947,
            'N': 0.5009, 'T': 0.5097, 'E': 0.6076, 'V': 0.6146, 'Q': 0.6313, 'H': 0.6725, 'M': 0.7151,
            'I': 0.7318, 'L': 0.7318, 'K': 0.7401, 'R': 0.7612, 'F': 0.8336, 'Y': 0.8499, 'W': 1.0000}

CHARGE = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
POLARITY = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
ACCEPTOR = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
DONOR = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
BACKBONE_NAMES = ["CA", "C", "N", "O"]
AA_NAME_NUMBER = {k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())}


class PDBProtein(object):
    def __init__(self, path):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError("Path not exit, please check the path")
    
        with open(path, 'r') as f:
            self.pdb = f.read()

        self.ptable = Chem.GetPeriodicTable()

        self.title = None
        self.atoms = []
        self.atom_lines = [] 
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.amino_acid_seq = []
        self.amino_acid_loc = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self.atoms_type = []
        self.residue_len = []
        self.hydropathy = []
        self.volume = []
        self.charge = []
        self.polarity = []
        self.acceptor = []
        self.donor = []
        
        self._get_atom_info()

    def _get_atom_info(self):

        residues_tmp = {}
        num = 0
        for line in self.pdb.splitlines():
            num += 1
            if line[0:6].strip() == 'HEADER':
                self.title = line[10:].strip().lower()
            elif line[0:6].strip() == 'ATOM':
                self.atoms.append(int(line[6:11]))  # atom
                self.atom_lines.append(line.strip())
                element_symb = line[76:78].strip().capitalize()  # C or N or O or other elements
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                atomic_number = self.ptable.GetAtomicNumber(element_symb)
                next_ptr = len(self.element)  # record atom location
                self.element.append(atomic_number)
                self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
                self.pos.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32))  # x y z
                self.atom_name.append(line[12:16].strip())  # atom_name
                self.is_backbone.append(line[12:16].strip() in BACKBONE_NAMES)
                self.atom_to_aa_type.append(AA_NAME_NUMBER[line[17:20].strip()])  # 'res_name'
                # self.amino_acid_loc.append(float(line[23:26]))
                # self.aa_name.append(line[17:20].strip())=

                chain_res_id = '%s_%d_%s' % (line[21:22].strip(), int(line[22:26]), line[26:27].strip())   # chain  segment  res_id  res_insert_id
                if chain_res_id not in residues_tmp:
                    residues_tmp[chain_res_id] = {
                        'name': line[17:20].strip(),  # res_name
                        'loc': float(line[23:26]),
                        'atoms': [next_ptr],    # 
                        'chain': line[21:22].strip(),   #  chain
                        'segment': line[72:76].strip(),  # segment
                    }
                else:
                    assert residues_tmp[chain_res_id]['name'] == line[17:20].strip()
                    assert residues_tmp[chain_res_id]['chain'] == line[21:22].strip() 
                    # print(next_ptr) 
                    residues_tmp[chain_res_id]['atoms'].append(next_ptr)
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.
        
        # print(residues_tmp)
        self._process_residues(residues_tmp)
        # print(residues_tmp)

    def _process_residues(self, residues_tmp):
        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]   # self.resides save value of residues_tmp  like {'name': 'LYS', 'atoms': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'chain': 'A', 'segment': 'A'}
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone
        for residue in self.residues:   # {'name': 'LYS', 'atoms': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'chain': 'A', 'segment': 'A'}
            self.amino_acid.append(AA_NAME_NUMBER[residue['name']])
            self.amino_acid_seq.append(AA_NAME_SYM[residue['name']])
            self.amino_acid_loc.append(residue['loc'])
            self.center_of_mass.append(residue['center_of_mass'])
            self.residue_len.append(len(residue['atoms']))
            self.hydropathy.append(HYDROPATHY[AA_NAME_SYM[residue['name']]])
            self.volume.append(VOLUME[AA_NAME_SYM[residue['name']]])
            self.charge.append(CHARGE[AA_NAME_SYM[residue['name']]])
            self.polarity.append(POLARITY[AA_NAME_SYM[residue['name']]])
            self.acceptor.append(ACCEPTOR[AA_NAME_SYM[residue['name']]])
            self.donor.append(DONOR[AA_NAME_SYM[residue['name']]])
            for name in BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self,  ligand_pos=None, distance=10.0):
        element = []
        pos = []
        is_backbone = []
        atom_name = []
        atom_to_aa_type = []
        if ligand_pos is not None:
            for residue in self.residues:
                if (((residue['center_of_mass'] - ligand_pos) ** 2).sum(-1) ** 0.5).min() < distance:
                    for atom_idx in residue['atoms']:
                        element.append(self.element[atom_idx])
                        pos.append(self.pos[atom_idx])
                        is_backbone.append(self.is_backbone[atom_idx])
                        atom_name.append(self.atom_name[atom_idx])
                        atom_to_aa_type.append(self.atom_to_aa_type[atom_idx])
            return {
                'element': np.array(element, dtype=int),
                'pos': np.array(pos, dtype=np.float32),
                'is_backbone': np.array(is_backbone, dtype=bool),
                'atom_name': self.atom_name,
                'atom_to_aa_type': np.array(atom_to_aa_type, dtype=int),
            }
        else:
            return {
                'element': np.array(self.element, dtype=int),
                'pos': np.array(self.pos, dtype=np.float32),
                'is_backbone': np.array(self.is_backbone, dtype=bool),
                'atom_name': self.atom_name,
                'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=int),
            }

    def to_dict_residue(self):
        attr = np.stack((self.hydropathy, self.volume, self.charge, self.polarity, self.acceptor, self.donor), axis=-1)
        # new_dict = {b: a for a, b in zip(self.amino_acid_seq, B)}
        return {
            'amino_acid': np.array(self.amino_acid, dtype=int),
            'amino_acid_seq': np.array(self.amino_acid_seq, dtype=str),
            'amino_acid_loc': np.array(self.amino_acid_loc, dtype=int),
            
            # 'atoms_type': np.array(self.atoms_type, dtype=int),
            # 'pos': np.array(self.pos, dtype=np.float32),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'len': np.array(self.residue_len, dtype=int),
            'attr':  np.array(attr, dtype=np.float32),
        }

    
    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        
        for residue in residues:
            
            for atom_idx in residue['atoms']:
                block += self.atom_lines[atom_idx] + "\n"
        block += "END\n"
        return block


if __name__ == '__main__':
    path = '/home/user/ydliu/drug_diffusion/data/crossdocked_pocket10/RDM1_ARATH_7_163_0/2q3t_A_rec_2q3t_cps_lig_tt_docked_47_pocket10.pdb'
    # parse_pdb_file(path)
    # for item in parse_pdb_file(path):
    #     print(item)
    residue = PDBProtein(path).to_dict_atom()
    # residue = PDBProtein(path).select_residue()
    print(residue)





