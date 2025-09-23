from rdkit import Chem

from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField

def count_rings_num(mol, sizes=[3, 4, 5, 6, 7, 8, 9]):
    """
    统计分子中指定大小环的数量

    参数：
        mol: RDKit 的 Mol 对象
        sizes: 要统计的环大小列表，默认统计3~9元环

    返回：
        一个字典，键为环的大小，值为该大小环的数量
    """
    # 获取分子的环信息
    ring_info = mol.GetRingInfo()
    # AtomRings() 返回一个包含每个环中原子索引元组的列表
    rings = ring_info.AtomRings()

    # 初始化计数字典
    ring_count = {size: 0 for size in sizes}
    ring_count['other'] = 0
    for ring in rings:
        ring_size = len(ring)
        if ring_size in ring_count:
            ring_count[ring_size] += 1
        elif ring_size > 9:
            ring_count['other'] += 1
    return ring_count

def calculate_top(g_dict, top_n, metric="qed"):

    sa_list, vina_list, qed_list, atom_num_list, lipinski_list, logp_list = [], [], [], [], [], []
    vina_score_list, vina_dock_list,  vina_mini_list, LE_list, vina_score_LE_list, vina_dock_LE_list, vina_mini_LE_list = [],[], [], [], [], [], []
    
    for lig_id, data in g_dict.items():
        metric_values = data[metric]
        qed = data["qed"]
        sa = data["sa"]
        vina = data["vina"]
        atom_num = data["atom_num"]
        lipinski = data['lipinski']
        logp = data['logp']
        vina_score = data['vina_score']
        vina_dock = data['vina_dock']
        vina_mini = data['vina_mini']
        LE = data['LE']
        vina_score_LE = data['vina_score_LE']
        vina_dock_LE = data['vina_dock_LE']
        vina_mini_LE = data['vina_mini_LE']

        if len(data["qed"]) < 50:
            continue
        
        reverse_order = metric not in ["LE", "vina", 'vina_score', 'vina_dock']
        # 对QED降序排序并取前top_n个（如果存在）
        sorted_indices = sorted(range(len(metric_values)), key=lambda i: metric_values[i], reverse=reverse_order)[:top_n]
        for idx in sorted_indices:
            sa_list.append(sa[idx])
            vina_list.append(vina[idx])
            qed_list.append(qed[idx])
            atom_num_list.append(atom_num[idx])
            lipinski_list.append(lipinski[idx])
            logp_list.append(logp[idx])
            vina_score_list.append(vina_score[idx])
            vina_dock_list.append(vina_dock[idx])
            vina_mini_list.append(vina_mini[idx])
            LE_list.append(LE[idx])
            vina_score_LE_list.append(vina_score_LE[idx])
            vina_dock_LE_list.append(vina_dock_LE[idx])
            vina_mini_LE_list.append(vina_mini_LE[idx])

    
    # 计算均值
    mean_sa = sum(sa_list) / len(sa_list) if sa_list else 0
    mean_vina = sum(vina_list) / len(vina_list) if vina_list else 0
    mean_qed = sum(qed_list) / len(qed_list) if qed_list else 0
    mean_atom_num = sum(atom_num_list) / len(atom_num_list) if atom_num_list else 0
    mean_lipinski = sum(lipinski_list) / len(lipinski_list) if lipinski_list else 0
    mean_logp = sum(logp_list) / len(logp_list) if logp_list else 0
    mean_vina_dock = sum(vina_dock_list) / len(vina_dock_list) if vina_dock_list else 0
    mean_vina_score = sum(vina_score_list) / len(vina_score_list) if vina_score_list else 0
    mean_vina_mini = sum(vina_mini_list) / len(vina_mini_list) if vina_mini_list else 0
    mean_LE = sum(LE_list) / len(LE_list) if LE_list else 0
    mean_vina_score_LE = sum(vina_score_LE_list) / len(vina_score_LE_list) if vina_score_LE_list else 0
    mean_vina_dock_LE = sum(vina_dock_LE_list) / len(vina_dock_LE_list) if vina_dock_LE_list else 0
    mean_vina_mini_LE = sum(vina_mini_LE_list) / len(vina_mini_LE_list) if vina_mini_LE_list else 0
    
    return mean_sa, mean_vina, mean_qed, mean_lipinski, mean_logp, mean_atom_num,\
          mean_vina_score, mean_vina_dock, mean_vina_mini, mean_LE, mean_vina_score_LE, mean_vina_dock_LE, mean_vina_mini_LE


FUSED_QUA_RING_PATTERN = [
    Chem.MolFromSmarts(i) for i in[
        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R](~&@[R]~&@1~&@4)~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]2~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]34~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R@@H](~&@[R@H]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R@H]~&@4~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@4~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@13~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1'
            ]
        ]
PATTERNS = [Chem.MolFromSmarts(i) for i in [
                        '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
                        '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
                        '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1']
           ]

def judge_fused_ring(mol):
    for pat in PATTERNS+FUSED_QUA_RING_PATTERN:
        if mol.HasSubstructMatch(pat):
            return True
    else:
        return False
    

PATTERNS_1 = [Chem.MolFromSmarts(i) for i in [
                        '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
                         '[R]1~&@[R]~&@[R]~&@12~&@[R]~&@[R]~&@2'
                        ]
           ]

def judge_unexpected_ring(mol):
    for pat in PATTERNS_1:
        subs = mol.GetSubstructMatches(pat)
        if len(subs) > 0:
            return True
    else:
        return False




def get_energy(mol, conf_id=-1):
    mol = Chem.AddHs(mol, addCoords=True)
    uff = UFFGetMoleculeForceField(mol, confId=conf_id)
    e_uff = uff.CalcEnergy()
    return e_uff


