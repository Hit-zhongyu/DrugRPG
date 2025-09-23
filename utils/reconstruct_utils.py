
import torch
import itertools
import copy
import networkx as nx
import numpy as np
from rdkit import Chem
from collections import defaultdict
from openbabel import openbabel as ob
import math

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    34: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
    (35, False): 13,
    (53, False): 14,
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}


def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol

PATTERNS_1 = [
        [Chem.MolFromSmarts('[#6,#7,#8]-[#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1'),
        Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')],
        [Chem.MolFromSmarts('[#6,#7,#8]-[#6]1(-[#6,#7,#8])~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1'),
        Chem.MolFromSmarts('[C,N,O]-[C]1(-[C,N,O])~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')]
        #Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C]~&@[C]~&@[N]~&@[C]~&@[C]-1'),
        #Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]-1'),
    ]
MAX_VALENCE = {'C':4, 'N':3}


def modify(mol, max_double_in_6ring=0):
    #atoms = mol.GetAtoms()
    mol_copy = copy.deepcopy(mol)
    mw = Chem.RWMol(mol)

    p1 = Chem.MolFromSmarts('[#6,#7]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1')
    p1_ = Chem.MolFromSmarts('[C,N]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    subs = set(list(mw.GetSubstructMatches(p1)) + list(mw.GetSubstructMatches(p1_)))
    subs_set_1 = [set(s) for s in subs]
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        #b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
        change_double = False
        r_b_double = 0
        b_list = []
        for ix,c in enumerate(comb):
            b = mw.GetBondBetweenAtoms(*c)
            if ix == 0:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRingSize(6)
                b_list.append((c, bt, is_r))
                continue
            if b is not None:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRing()
                b_list.append((c, bt, is_r))
                if is_r is True and bt == 'DOUBLE':
                    r_b_double += 1
                    if r_b_double > max_double_in_6ring:
                        change_double = True
        if change_double:
            for ix,b in enumerate(b_list):
                if ix == 0:
                    if b[-1] is False:
                        mw.RemoveBond(*b[0])
                        mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    else:
                        continue
                if b[1] == 'DOUBLE' and b[-1] is False:
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    break
    
    #p2 = Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    for p2 in PATTERNS_1:
        Chem.GetSSSR(mw)
        subs2 = set(list(mw.GetSubstructMatches(p2[0])) + list(mw.GetSubstructMatches(p2[1])))
        for sub in subs2:
            comb = itertools.combinations(sub, 2)
            b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
            for b in b_list:
                if b[-1].GetBondType().__str__() == 'DOUBLE':
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    Chem.GetSSSR(mw)
    p3 = Chem.MolFromSmarts('[#8]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1')
    p3_ = Chem.MolFromSmarts('[O]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    subs = set(list(mw.GetSubstructMatches(p3)) + list(mw.GetSubstructMatches(p3_)))
    subs_set_2 = [set(s) for s in subs]
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
        for b in b_list:
            if b[-1].GetBondType().__str__() == 'DOUBLE' and b[-1].IsInRing() is True:
                mw.RemoveBond(*b[0])
                mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    p = Chem.MolFromSmarts('[#6,#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1')
    p_ = Chem.MolFromSmarts('[C,N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')
    Chem.GetSSSR(mw)
    subs = set(list(mw.GetSubstructMatches(p)) + list(mw.GetSubstructMatches(p_)))
    subs_set_3 = [set(s) for s in subs]
    for sub in subs:
        pass_sub = False
        if subs_set_2:
            for s in subs_set_2:
                if len(s-set(sub)) == 1:
                    pass_sub = True
                    break
        if pass_sub:
            continue

        bond_list = [(i,sub[0]) if ix+1==len(sub) else (i, sub[ix+1]) for ix,i in enumerate(sub)]
        if len(bond_list) == 0:
            continue
        atoms = [mw.GetAtomWithIdx(i) for i in sub]
        for a in atoms:
            if a.GetExplicitValence()==MAX_VALENCE[a.GetSymbol()] and a.GetHybridization().__str__()=='SP3':
                break
        else:
            bond_type = [mw.GetBondBetweenAtoms(*b).GetBondType().__str__() for b in bond_list]
            if bond_type.count('DOUBLE') > max_double_in_6ring:
                for b in bond_list:
                    mw.RemoveBond(*b)
                    mw.AddBond(*b, Chem.rdchem.BondType.AROMATIC)
    
    # get new mol from modified mol
    conf = mw.GetConformer()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(mw.GetNumAtoms())
    for i, atom in enumerate(mw.GetAtoms()):
        rd_atom = Chem.Atom(atom.GetAtomicNum())
        rd_mol.AddAtom(rd_atom)
        rd_coords = conf.GetAtomPosition(i)
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    for i, bond in enumerate(mw.GetBonds()):
        bt = bond.GetBondType()
        node_i = bond.GetBeginAtomIdx()
        node_j = bond.GetEndAtomIdx()
        rd_mol.AddBond(node_i, node_j, bt)
    out_mol = rd_mol.GetMol()
    # check validility of the new mol
    mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(out_mol))
    if mol_check:
        try:
            Chem.Kekulize(out_mol)
            del mol_copy
            return out_mol
        except:
            del mol
            return None
    else:
        del mol
        return None
    # if mol_check:
    #     try:
    #         Chem.Kekulize(out_mol)
    #         del mol_copy
    #         return out_mol
    #     except:
    #         del mol
    #         return mol_copy
    # else:
    #     del mol
    #     return mol_copy


def remove_cycles(edge_index, pos, bond_type):
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        return edge_index, bond_type
    
    edge_to_cycles = defaultdict(list)
    original_edges_sorted = set(tuple(sorted(edge)) for edge in edge_index.t().tolist())
    
    for cycle in cycles:
        cycle_length = len(cycle)
        closed_cycle = cycle + [cycle[0]]
        for i in range(len(closed_cycle)-1):
            u, v = closed_cycle[i], closed_cycle[i+1]
            edge = tuple(sorted((u, v))) 
            edge_to_cycles[edge].append(cycle_length)

    edges_to_remove = set()
    
    for cycle in cycles:
        cycle_length = len(cycle)
        if cycle_length not in [3, 4]:
        # if cycle_length != 3:
            continue 
        
        closed_cycle = cycle + [cycle[0]]    
        cycle_edges = []
        for i in range(len(closed_cycle)-1):
            u, v = closed_cycle[i], closed_cycle[i+1]
            edge_sorted = tuple(sorted((u, v)))
            if edge_sorted in original_edges_sorted:
                cycle_edges.append(edge_sorted)
        
        if not cycle_edges:
            continue
        
        involved_in_other_cycles = False
        for edge in cycle_edges:
            if len(edge_to_cycles[edge]) > 1:
                involved_in_other_cycles = True
                break
        
        if not involved_in_other_cycles:
            continue

        edges_with_length = []
        for edge in cycle_edges:
            u, v = edge
            dist = torch.norm(pos[u] - pos[v], p=2).item()
            edges_with_length.append((dist, edge))
        edges_with_length.sort(reverse=True, key=lambda x: x[0]) 
        
        # selected_edge = None
        # for dist, edge_sorted in edges_with_length:
        #     associated_cycles = edge_to_cycles.get(edge_sorted, [])
        #     has_non_target = any(length not in [3,4] for length in associated_cycles)
        #     if not has_non_target:
        #         selected_edge = edge_sorted
        #         break
        selected_edge = None
        for dist, edge_sorted in edges_with_length:
            associated_cycles = edge_to_cycles.get(edge_sorted, [])
            is_only_in_target_cycles = all(length in [3,4] for length in associated_cycles)
            # is_only_in_target_cycles = all(length == 3 for length in associated_cycles)
            if is_only_in_target_cycles:
                selected_edge = edge_sorted
                break
        if selected_edge is None and len(edges_with_length) > 0:
            selected_edge = edges_with_length[0][1]
        
        if selected_edge is not None:
            edges_to_remove.add(selected_edge)
    
    original_edges = set(map(tuple, edge_index.t().tolist()))
    edges_to_remove_original = set()
    for edge_sorted in edges_to_remove:
        for edge in original_edges:
            if tuple(sorted(edge)) == edge_sorted:
                edges_to_remove_original.add(edge)
                break 

    original_edges_list = edge_index.t().tolist()
    mask = []
    for edge in original_edges_list:
        if tuple(edge) not in edges_to_remove_original:
            mask.append(True)
        else:
            mask.append(False)
    
    mask = torch.tensor(mask, dtype=torch.bool)
    new_edge_index = edge_index[:, mask]
    new_bond_type = bond_type[mask]
    
    return new_edge_index, new_bond_type

# def remove_cycles(edge_index, pos, bond_type):
#     G = nx.Graph()
#     edges = edge_index.t().tolist()
#     G.add_edges_from(edges)
    
#     try:
#         cycles = nx.cycle_basis(G)
#     except nx.NetworkXNoCycle:
#         return edge_index, bond_type
    
#     edge_to_cycles = defaultdict(list)
#     original_edges_sorted = set(tuple(sorted(edge)) for edge in edge_index.t().tolist())
    
#     # 记录每条边属于哪些环
#     for cycle_idx, cycle in enumerate(cycles):
#         cycle_length = len(cycle)
#         closed_cycle = cycle + [cycle[0]]
#         for i in range(len(closed_cycle)-1):
#             u, v = closed_cycle[i], closed_cycle[i+1]
#             edge = tuple(sorted((u, v))) 
#             edge_to_cycles[edge].append((cycle_idx, cycle_length))

#     edges_to_remove = set()
    
#     for cycle_idx, cycle in enumerate(cycles):
#         cycle_length = len(cycle)
        
#         # 跳过不是3或4元环的情况
#         if cycle_length not in [3, 4]:
#             continue 
        
#         closed_cycle = cycle + [cycle[0]]    
#         cycle_edges = []
#         for i in range(len(closed_cycle)-1):
#             u, v = closed_cycle[i], closed_cycle[i+1]
#             edge_sorted = tuple(sorted((u, v)))
#             if edge_sorted in original_edges_sorted:
#                 cycle_edges.append(edge_sorted)
        
#         if not cycle_edges:
#             continue
        
#         # 检查环是否与其他环共享边
#         shared_edges = set()
#         for edge in cycle_edges:
#             cycle_list = edge_to_cycles[edge]
#             if len(cycle_list) > 1:  # 这条边属于多个环
#                 shared_edges.add(edge)
        
#         # 判断是否为单独存在的环
#         is_isolated_cycle = (len(shared_edges) == 0)
        
#         # 如果是单独的三元环，跳过不处理
#         if is_isolated_cycle and cycle_length == 3:
#             continue
        
#         # 按边长排序
#         edges_with_length = []
#         for edge in cycle_edges:
#             u, v = edge
#             dist = torch.norm(pos[u] - pos[v], p=2).item()
#             edges_with_length.append((dist, edge))
#         edges_with_length.sort(reverse=True, key=lambda x: x[0])
        
#         selected_edge = None
        
#         if is_isolated_cycle:
#             # 单独的四元环：选择最长的边
#             selected_edge = edges_with_length[0][1]
#         else:
#             # 与其他环共享边的环（3元或4元）：选择非共享边中最长的
#             for dist, edge_sorted in edges_with_length:
#                 if edge_sorted not in shared_edges:
#                     selected_edge = edge_sorted
#                     break
            
#             # 如果所有边都是共享边，选择最长的
#             if selected_edge is None:
#                 selected_edge = edges_with_length[0][1]
        
#         if selected_edge is not None:
#             edges_to_remove.add(selected_edge)
    
#     # 后续代码保持不变
#     original_edges = set(map(tuple, edge_index.t().tolist()))
#     edges_to_remove_original = set()
#     for edge_sorted in edges_to_remove:
#         for edge in original_edges:
#             if tuple(sorted(edge)) == edge_sorted:
#                 edges_to_remove_original.add(edge)
#                 break 

#     original_edges_list = edge_index.t().tolist()
#     mask = []
#     for edge in original_edges_list:
#         if tuple(edge) not in edges_to_remove_original:
#             mask.append(True)
#         else:
#             mask.append(False)
    
#     mask = torch.tensor(mask, dtype=torch.bool)
#     new_edge_index = edge_index[:, mask]
#     new_bond_type = bond_type[mask]
    
#     return new_edge_index, new_bond_type

def remove_bond(edge_index, pos, bond_type, frag_bond_index=None, threshold=2.0):

    src_pos = pos[edge_index[0]]
    dst_pos = pos[edge_index[1]]
    distances = torch.norm(src_pos - dst_pos, dim=1, p=2)

    mask = distances <= threshold
    if frag_bond_index is not None: 
        frag_atom_set = set(frag_bond_index[0].tolist()) | set(frag_bond_index[1].tolist())
        src_in_frag = torch.tensor([src.item() in frag_atom_set for src in edge_index[0]])
        dst_in_frag = torch.tensor([dst.item() in frag_atom_set for dst in edge_index[1]])
        both_in_frag = src_in_frag & dst_in_frag
        mask = mask | both_in_frag

    new_edge_index = edge_index[:, mask]
    bond_type = bond_type[mask]
    return new_edge_index, bond_type

def get_max_valence(atom_type):
    """返回原子的最大价键数"""
    if atom_type == 'C':
        return 4
    elif atom_type == 'N':
        return 3
    elif atom_type == 'O':
        return 2
    else:
        # 其他原子类型默认不限制，返回一个大值
        return 100
    
def adjust_bond_types_by_valence(atom, edge_index, bond_type, frag_bond_index=None):

    edge_index = edge_index.t()
    # new_bond_type = bond_type.clone()
    # valence = defaultdict(int)
    # degree = defaultdict(int)
    
    # for i in range(edge_index.shape[1]):
    #     src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    #     k = int(bond_type[i])
        
    #     # 避免重复计算（无向图）
    #     if src < dst:
    #         valence[src] += k
    #         valence[dst] += k
        
    #     degree[src] += 1
    #     degree[dst] += 1
    
    # # 识别需要降级的键
    # for i in range(edge_index.shape[1]):
    #     src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    #     k = int(bond_type[i])
        
    #     if k not in {2, 3}:
    #         continue
            
    #     # 检查各种降级条件
    #     should_reduce = False
        
    #     # 价电子数超限
    #     max_src = get_max_valence(atom[src])
    #     max_dst = get_max_valence(atom[dst])
    #     if valence[src] > max_src or valence[dst] > max_dst:
    #         should_reduce = True
            
    #     # 端基碳规则
    #     if (atom[src] == "C" and degree[src] == 1) or (atom[dst] == "C" and degree[dst] == 1):
    #         should_reduce = True
            
    #     # C≡C 规则
    #     if atom[src] == "C" and atom[dst] == "C" and k == 3:
    #         should_reduce = True
            
    #     # N=N 或 N≡N 规则
    #     if atom[src] == "N" and atom[dst] == "N" and k in (2, 3):
    #         should_reduce = True
            
    #     if should_reduce:
    #         new_bond_type[i] = 1
    
    bond_types = []
    edge_indexs = []
    valence = defaultdict(int)
    for i in range(len(edge_index)):
        src, dst = edge_index[i]
        src = src.item()
        dst = dst.item()
        if src < dst:
            k = int(bond_type[i])
            valence[src] += k
            valence[dst] += k
            bond_types.append(k)
            edge_indexs.append([src, dst])

    degree = defaultdict(int)
    for src, dst in edge_indexs:
         degree[src] += 1
         degree[dst] += 1

    to_reduce = set()
    for i in range(len(edge_index)):
        src, dst = edge_indexs[i]
        k = int(bond_type[i])
        if k not in {2, 3}:
            continue  

        atom_src, atom_dst = atom[src], atom[dst]
        max_src = get_max_valence(atom_src)
        max_dst = get_max_valence(atom_dst)

        if valence[src] > max_src or valence[dst] > max_dst:
            to_reduce.add((src, dst))
        if (atom[src] == "C" and degree[src] == 1) or (atom[dst] == "C" and degree[dst] == 1):
            to_reduce.add((src, dst))
        if atom[src] == "C" and atom[dst] == "C" and k == 3:
            to_reduce.add((src, dst))
        if atom[src] == "N" and atom[dst] == "N" and k in (2, 3):
            to_reduce.add((src, dst))

    edge_index = edge_index.t()

    new_bond_type = bond_types
    if to_reduce is not None:
        for (src, dst) in to_reduce:
            idx = (edge_index[0] == src) & (edge_index[1] == dst) 
            new_bond_type[torch.where(idx)[0]] = 1

    return edge_index, new_bond_type


def process_independent_ring_bonds(mol):

    rw_mol = Chem.RWMol(mol)
    
    ring_info = rw_mol.GetRingInfo()
    rings = ring_info.AtomRings()

    for bond in rw_mol.GetBonds():
        if not bond.IsInRing():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_atom = rw_mol.GetAtomWithIdx(begin_idx)
            end_atom = rw_mol.GetAtomWithIdx(end_idx)

            if begin_atom.IsInRing() and end_atom.IsInRing():
                begin_rings = [set(r) for r in rings if begin_idx in r]
                end_rings = [set(r) for r in rings if end_idx in r]
                common_ring = False
                for br in begin_rings:
                    for er in end_rings:
                        if br & er:
                            common_ring = True
                            break
                    if common_ring:
                        break
                if not common_ring and bond.GetBondType() == Chem.BondType.DOUBLE:
                    bond.SetBondType(Chem.BondType.SINGLE)
            
            if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                if begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'C':
                    bond.SetBondType(Chem.BondType.SINGLE)
    return rw_mol


def check_alert_structures(mol, alert_smarts_list):
    Chem.GetSSSR(mol)
    patterns = [Chem.MolFromSmarts(sma) for sma in alert_smarts_list]
    for p in patterns:
        subs = mol.GetSubstructMatches(p)
        if len(subs) != 0:
            return True
    else:
        return False
