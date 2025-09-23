from .geometry import eval_bond_length, eval_stability, eval_bond_angle


def eval_geom(mol):
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()

    # eval bond_length
    bond_dist = eval_bond_length.bond_distance_from_mol(mol)
    # all_bond_dist += bond_dist

    # eval bond_angle
    bond_angle = eval_bond_angle.bond_angle_from_mol(mol)
    # all_bond_angle += bond_angle

    # eval stability            
    # r_stable = eval_stability.check_stability(pos, atom_types)
    # all_mol_stable += r_stable[0]
    # all_atom_stable += r_stable[1]
    # all_n_atom += r_stable[2]

    geom_results = {'bond_dist': bond_dist, 'bond_angle': bond_angle}
                # 'mol_stable': r_stable[0], 'atom_stable_num': r_stable[1], 
                # 'atom_num': r_stable[2]}
    
    return geom_results