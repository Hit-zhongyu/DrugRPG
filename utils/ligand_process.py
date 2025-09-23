import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures, QED
from evaluation.sascorer import compute_sa_score

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}

def parse_sdf_file(path):
    
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False, sanitize=False)))  # 该方式读取的文件存成的是一个列表的形式，支持列表的操作。 removeHs=False表示不移除氢原子 
    rdmol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(rdmol)   #  获取最小环集合  这是一个地址
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=int)

    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1   # GetAtomIds() 获取与特定化学特征相关的原子的索引

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()  # 获取元素周期表的对象
    element, pos, charges = [], [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x: x.split(), sdf[4:4 + num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())  # capitalize() 将字符串的第一个字母变成大写   GetAtomicNumber将元素符号变为原子号
        # repalce Br as Cl
        if atomic_number == 35:
            atomic_number = 17  # 简化计算
        element.append(atomic_number)
        pos.append([x, y, z])

        atomic_weight = ptable.GetAtomicWeight(atomic_number)  # 获取原子量
        accum_pos += np.array([x, y, z]) * atomic_weight   
        accum_mass += atomic_weight  # 计算原子总量
    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)


    element = np.array(element, dtype=int)
    pos = np.array(pos, dtype=np.float32)
    # charges = np.array(charges, dtype=np.float32)
    # print(charges)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    # row, col, edge_type = [], [], []
    # for bond_line in sdf[4 + num_atoms:4 + num_atoms + num_bonds]:
    #     start, end = int(bond_line[0:3]) - 1, int(bond_line[3:6]) - 1
    #     row += [start, end]
    #     col += [end, start]
    #     edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    # edge_index = np.array([row, col], dtype=int)
    # edge_type = np.array(edge_type, dtype=int)

    # perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()  # argsort 返回np中从小到大的索引值
    # edge_index = edge_index[:, perm]
    # edge_type = edge_type[perm]

    bond_dict = {}
    for bond_line in sdf[4 + num_atoms : 4 + num_atoms + num_bonds]:
        start = int(bond_line[0:3]) - 1
        end = int(bond_line[3:6]) - 1
        a, b = (start, end) if start < end else (end, start)
        bond_type = bond_type_map[int(bond_line[6:9])]
        bond_dict[(a, b)] = bond_type

    row, col, edge_type = [], [], []
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            if i == j:
                continue  # 跳过自环
            # 查找无序键对
            a, b = (i, j) if i < j else (j, i)
            bond = bond_dict.get((a, b), 0)  # 不存在则返回0
            row.append(i)
            col.append(j)
            edge_type.append(bond)

    edge_index = np.array([row, col], dtype=int) 
    
    data = {
        'element': element,  # 原子类型
        'pos': pos,  # 坐标
        # 'charge': charges,  # 电荷
        'bond_index': edge_index,  # 边
        'bond_type': edge_type,   # 单键 双键或其他
        'center_of_mass': center_of_mass,   # 分子质量中心向量
        'atom_feature': feat_mat,   # 原子属于哪一类  如'Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable'
    }
    
    return data


if __name__ == '__main__':
    path = '/mnt/rna01/lzy/pocketdiff5/8pkm/8pkm.sdf'
    data = parse_sdf_file(path)
    print(data['bond_type'])


