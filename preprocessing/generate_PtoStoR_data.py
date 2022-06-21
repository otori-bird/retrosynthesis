import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import multiprocessing

from rdkit import Chem
from tqdm import tqdm


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# Get the mapping numbers in a SMARTS.
def get_idx(smarts_item):
    item = re.findall('(?<=:)\d+', smarts_item)
    item = list(map(int, item))
    return item


#  Get the dict maps each atom index to the mapping number.
def get_atomidx2mapidx(mol):
    atomidx2mapidx = {}
    for atom in mol.GetAtoms():
        atomidx2mapidx[atom.GetIdx()] = atom.GetAtomMapNum()
    return atomidx2mapidx


#  Get the dict maps each mapping number to the atom index .
def get_mapidx2atomidx(mol):
    mapidx2atomidx = {}
    mapidx2atomidx[0] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            mapidx2atomidx[0].append(atom.GetIdx())
        else:
            mapidx2atomidx[atom.GetAtomMapNum()] = atom.GetIdx()
    return mapidx2atomidx


# Get the reactant atom index list in the order of product atom index.
def get_order(product_mol, patomidx2pmapidx, rmapidx2ratomidx):
    order = []
    for atom in product_mol.GetAtoms():
        order.append(rmapidx2ratomidx[patomidx2pmapidx[atom.GetIdx()]])
    return order


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


# Convert smarts to smiles by remove mapping numbers
# copy from RetroXpert
def old_smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)


def clear_map_canonical_smiles(smi, canonical=True, root=-1,sanitize=True):
    mol = Chem.MolFromSmiles(smi,sanitize=sanitize)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


# Split product smarts according to the target adjacent matrix
def get_smarts_pieces(mol, src_adj, target_adj, reacts, add_bond=False):
    m, n = src_adj.shape
    emol = Chem.RWMol(mol)
    for j in range(m):
        for k in range(j + 1, n):
            if target_adj[j][k] == src_adj[j][k]:
                continue
            # emol.RemoveBond(j,k)
            # if target_adj[j][k] > 0:
            #     bt = target_adj[j][k]
            #     if bt == 1:
            #         bt = Chem.rdchem.BondType.SINGLE
            #     elif bt == 2:
            #         bt = Chem.rdchem.BondType.DOUBLE
            #     elif bt == 3:
            #         bt = Chem.rdchem.BondType.TRIPLE
            #     elif bt == 4:
            #         bt = Chem.rdchem.BondType.AROMATIC
            #     emol.AddBond(j,k,bt)
            if 0 == target_adj[j][k]:
                bt = emol.GetBondBetweenAtoms(j,k).GetBondType()
                emol.RemoveBond(j, k)
                hs = 1
                if bt == Chem.BondType.DOUBLE:
                    hs = 2
                elif bt == Chem.BondType.TRIPLE:
                    hs = 3
                emol.GetAtomWithIdx(j).SetNumExplicitHs(emol.GetAtomWithIdx(j).GetNumExplicitHs()+hs)
                emol.GetAtomWithIdx(k).SetNumExplicitHs(emol.GetAtomWithIdx(k).GetNumExplicitHs()+hs)
            elif add_bond:
                emol.AddBond(j, k, Chem.rdchem.BondType.SINGLE)
    synthon_mol = emol.GetMol()
    synthon_smiles = Chem.MolToSmiles(synthon_mol, isomericSmiles=True)
    synthons = synthon_smiles.split('.')
    # Find the reactant with maximum common atoms for each synthon
    syn_idx_list = [get_idx(synthon) for synthon in synthons]
    react_idx_list = [get_idx(react) for react in reacts]

    react_max_common_synthon_index = []
    for react_idx in react_idx_list:
        react_common_idx_cnt = []
        for syn_idx in syn_idx_list:
            common_idx = list(set(syn_idx) & set(react_idx))
            react_common_idx_cnt.append(len(common_idx))
        max_cnt = max(react_common_idx_cnt)
        react_max_common_index = react_common_idx_cnt.index(max_cnt)
        react_max_common_synthon_index.append(react_max_common_index)
    react_synthon_index = np.argsort(react_max_common_synthon_index).tolist()
    reacts = [reacts[k] for k in react_synthon_index]

    return '.'.join(synthons), '.'.join(reacts)
    # return ' . '.join(synthons), ' . '.join(reacts)


def del_index(smarts):
    t = re.sub(':\d*', '', smarts)
    return t


def onehot_encoding(x, total):
    return np.eye(total)[x]


def collate(data):
    return map(list, zip(*data))


def get_cano_map_number(smi,root=-1, sanitize=True):
    atommap_mol = Chem.MolFromSmiles(smi,sanitize=sanitize)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi,root=root,sanitize=sanitize),sanitize=sanitize)
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for i,index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol,root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root
    # root = -1
    # for i, atom in enumerate(mol.GetAtoms()):
    #     if atom.GetAtomMapNum() == root_map_number:
    #         return i


"""single version"""
# def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None,root_aligned=True):
#     """
#     preprocess reaction data to extract graph adjacency matrix and features
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     total = 0
#     src_data = []
#     tgt_data = []
#     differ = 0
#     skip_dict = {
#         'invalid_p':0,
#         'invalid_r':0,
#         'small_p':0,
#         'small_r':0,
#         'error_mapping':0,
#         'error_mapping_p':0,
#         'empty_p':0,
#         'empty_r':0,
#     }
#     pt = re.compile(r':(\d+)]')
#     # cano_dis = []
#     # root_dis = []
#     pbar = tqdm(range(len(reactants)))
#     products_origin = products.copy()
#     reactants_origin = reactants.copy()
#     for index in pbar:
#         pbar.set_description(f"Data Size:{len(src_data)}")
#         product = products_origin[index]
#         reactant = reactants_origin[index]
#         pro_mol = Chem.MolFromSmiles(product)
#         rea_mol = Chem.MolFromSmiles(reactant)
#         """checking data quality"""
#         rids = sorted(re.findall(pt, reactant))
#         pids = sorted(re.findall(pt, product))
#         return_status = {
#             "status": 0,
#             "ptos_src_data": [],
#             "ptos_tgt_data": [],
#             "stor_src_data": [],
#             "stor_tgt_data": [],
#         }
#         if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
#             return_status["status"] = "error_mapping"
#         if len(set(rids)) != len(rids):  # mapping is not 1:1
#             return_status["status"] = "error_mapping"
#         if len(set(pids)) != len(pids):  # mapping is not 1:1
#             return_status["status"] = "error_mapping"
#         if "" == product:
#             return_status["status"] = "empty_p"
#         if "" == reactant:
#             return_status["status"] = "empty_r"
#         if rea_mol is None:
#             return_status["status"] = "invalid_r"
#         if len(rea_mol.GetAtoms()) < 5:
#             return_status["status"] = "small_r"
#         if pro_mol is None:
#             return_status["status"] = "invalid_p"
#         if len(pro_mol.GetAtoms()) == 1:
#             return_status["status"] = "small_p"
#         if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
#             return_status["status"] = "error_mapping_p"
#         """finishing checking data quality"""
#
#         if return_status['status'] == 0:
#             product_adj = Chem.rdmolops.GetAdjacencyMatrix(pro_mol)
#             product_adj = product_adj + np.eye(product_adj.shape[0])
#             product_adj = product_adj.astype(np.bool)
#             reactant_adj = Chem.rdmolops.GetAdjacencyMatrix(rea_mol)
#             reactant_adj = reactant_adj + np.eye(reactant_adj.shape[0])
#             reactant_adj = reactant_adj.astype(np.bool)
#             patomidx2pmapidx = get_atomidx2mapidx(pro_mol)
#             rmapidx2ratomidx = get_mapidx2atomidx(rea_mol)
#             order = get_order(pro_mol, patomidx2pmapidx, rmapidx2ratomidx)
#             target_adj = reactant_adj[order][:, order]
#             reactant = reactant.split(".")
#             synthons, ordered_reactants = get_smarts_pieces(pro_mol, product_adj,
#                                                             target_adj, reactant)
#
#             pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
#
#             # in fact these are synthons, lazy to change the variable names
#             synthons = synthons.split(".")
#             reactants = ordered_reactants.split(".")
#
#             if root_aligned:
#                 # reversable = len(reactant) > 1
#                 reversable = False  # no shuffle
#
#                 """ptos"""
#                 product_roots = [-1]
#                 max_times = len(pro_atom_map_numbers)
#                 times = min(augmentation, max_times)
#                 if times < augmentation:  # times = max_times
#                     product_roots.extend(pro_atom_map_numbers)
#                     product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
#                 else:  # times = augmentation
#                     while len(product_roots) < times:
#                         product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
#                         # pro_atom_map_numbers.remove(product_roots[-1])
#                         if product_roots[-1] in product_roots[:-1]:
#                             product_roots.pop()
#                 times = len(product_roots)
#                 syn_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", syn))) for syn in synthons]
#                 assert times == augmentation
#                 if reversable:
#                     times = int(times / 2)
#                 for k in range(times):
#                     pro_root_atom_map = product_roots[k]
#                     pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
#                     cano_atom_map = get_cano_map_number(product, root=pro_root)
#                     if cano_atom_map is None:
#                         return_status["status"] = "error_mapping"
#                         return return_status
#                     pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
#                     aligned_synthons = []
#                     aligned_synthons_order = []
#                     used_indices = []
#                     for i, syn_map_number in enumerate(syn_atom_map_numbers):
#                         for j, map_number in enumerate(cano_atom_map):
#                             # select mapping reactans
#                             if map_number in syn_map_number:
#                                 syn_root = get_root_id(Chem.MolFromSmiles(synthons[i], sanitize=False),
#                                                        root_map_number=map_number)
#                                 syn_smi = clear_map_canonical_smiles(synthons[i], canonical=True, root=syn_root,
#                                                                      sanitize=False)
#                                 aligned_synthons.append(syn_smi)
#                                 aligned_synthons_order.append(j)
#                                 used_indices.append(i)
#                                 break
#                     sorted_synthons = sorted(list(zip(aligned_synthons, aligned_synthons_order)), key=lambda x: x[1])
#                     aligned_synthons = [item[0] for item in sorted_synthons]
#                     syn_smi = ".".join(aligned_synthons)
#                     product_tokens = smi_tokenizer(pro_smi)
#                     synthon_tokens = smi_tokenizer(syn_smi)
#                     return_status['ptos_src_data'].append(product_tokens)
#                     return_status['ptos_tgt_data'].append(synthon_tokens)
#                     # if Chem.MolFromSmiles(pro_smi).GetAtomWithIdx(0).GetSymbol() != Chem.MolFromSmiles(reactant_smi).GetAtomWithIdx(0).GetSymbol():
#                     #     print(index)
#                     #     print(product_tokens)
#                     #     print(reactant_tokens)
#
#                     # cano_dis.append(Levenshtein.distance(pro_smi,clear_map_canonical_smiles(reactant_smi)))
#                     # root_dis.append(Levenshtein.distance(pro_smi,reactant_smi))
#                     if reversable:
#                         aligned_synthons.reverse()
#                         syn_smi = ".".join(aligned_synthons)
#                         product_tokens = smi_tokenizer(pro_smi)
#                         synthon_tokens = smi_tokenizer(syn_smi)
#                         return_status['ptos_src_data'].append(product_tokens)
#                         return_status['ptos_tgt_data'].append(synthon_tokens)
#                 """stor"""
#                 synthon_roots = []
#                 max_times = np.prod([len(map_numbers) for map_numbers in syn_atom_map_numbers])
#                 times = min(augmentation, max_times)
#                 j = 0
#                 while j < times:
#                     synthon_roots.append([random.sample(syn_atom_map_numbers[k], 1)[0] for k in range(len(synthons))])
#                     if synthon_roots[-1] in synthon_roots[:-1]:
#                         synthon_roots.pop()
#                     else:
#                         j += 1
#                 if j < augmentation:
#                     synthon_roots.extend(random.choices(synthon_roots, k=augmentation - times))
#                     times = augmentation
#                 if reversable:
#                     times = int(times / 2)
#                 for j in range(times):
#                     max_synthon_size = -1
#                     rooted_reactants, rooted_synthons = [], []
#                     pro_root_map_number = -1
#                     for k, (rea, syn) in enumerate(zip(reactants, synthons)):
#                         if j + 1 == times:
#                             syn_root_atom_map = -1
#                         else:
#                             syn_root_atom_map = synthon_roots[j][k]
#                         syn_mol = Chem.MolFromSmiles(syn, sanitize=False)
#                         syn_root = get_root_id(syn_mol, root_map_number=syn_root_atom_map)
#                         syn_size = len(syn_mol.GetAtoms())
#                         cano_atom_map = get_cano_map_number(syn, root=syn_root, sanitize=False)
#                         syn = clear_map_canonical_smiles(syn, root=syn_root, sanitize=False)
#                         root_map_number = cano_atom_map[0]
#                         if pro_root_map_number == -1 or max_synthon_size < syn_size:
#                             pro_root_map_number = root_map_number
#                             max_synthon_size = syn_size
#                         rea_mol = Chem.MolFromSmiles(rea)
#                         rea_root = get_root_id(rea_mol, root_map_number=root_map_number)
#                         rea = clear_map_canonical_smiles(rea, root=rea_root)
#                         rooted_reactants.append(rea)
#                         rooted_synthons.append(syn)
#
#                         # rooted_reactants.sort(key = lambda i:len(i), reverse=True)
#                         # rooted_synthons.sort(key = lambda i:len(i), reverse=True)
#                     if len(rooted_reactants) < len(reactants):
#                         for j in range(len(rooted_reactants), len(reactants)):
#                             rooted_reactants.append(clear_map_canonical_smiles(reactants[j]))
#                     if len(rooted_synthons) < len(synthons):
#                         for j in range(len(rooted_synthons), len(synthons)):
#                             rooted_synthons.append(clear_map_canonical_smiles(synthons[j]))
#                     assert len(rooted_reactants) == len(reactants)
#                     assert len(rooted_synthons) == len(synthons)
#                     reactant = ".".join(rooted_reactants)
#                     synthon = ".".join(rooted_synthons)
#                     pro_mol = Chem.MolFromSmiles(product)
#                     pro_root = get_root_id(pro_mol, pro_root_map_number)
#                     pro_smi = clear_map_canonical_smiles(product, root=pro_root)
#                     # product = clear_map_canonical_smiles(product)
#                     # product_tokens = smi_tokenizer(synthon) + " [PREDICT] " + smi_tokenizer(product)
#                     product_tokens = smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(synthon)
#                     reactant_tokens = smi_tokenizer(reactant)
#                     return_status['stor_src_data'].append(product_tokens)
#                     return_status['stor_tgt_data'].append(reactant_tokens)
#                     if reversable:
#                         rooted_reactants.reverse()
#                         rooted_synthons.reverse()
#                         reactant = ".".join(rooted_reactants)
#                         synthon = ".".join(rooted_synthons)
#                         product_tokens = smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(synthon)
#                         reactant_tokens = smi_tokenizer(reactant)
#                         return_status['stor_src_data'].append(product_tokens)
#                         return_status['stor_tgt_data'].append(reactant_tokens)
#             else:
#                 cano_product = clear_map_canonical_smiles(product)
#                 cano_synthons = ".".join([clear_map_canonical_smiles(syn, sanitize=False) for syn in synthons])
#                 cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactants])
#                 return_status['ptos_src_data'].append(smi_tokenizer(cano_product))
#                 return_status['ptos_tgt_data'].append(smi_tokenizer(cano_synthons))
#                 return_status['stor_src_data'].append(
#                     smi_tokenizer(cano_product) + " [PREDICT] " + smi_tokenizer(cano_synthons))
#                 return_status['stor_tgt_data'].append(smi_tokenizer(cano_reactanct))
#                 pro_mol = Chem.MolFromSmiles(cano_product)
#                 syn_mols = [Chem.MolFromSmiles(syn, sanitize=False) for syn in cano_synthons.split(".")]
#                 rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
#                 for i in range(int(augmentation - 1)):
#                     pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
#                     syn_smi = [Chem.MolToSmiles(syn_mol, doRandom=True) for syn_mol in syn_mols]
#                     rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
#                     syn_smi = ".".join(syn_smi)
#                     rea_smi = ".".join(rea_smi)
#                     return_status['ptos_src_data'].append(smi_tokenizer(pro_smi))
#                     return_status['ptos_tgt_data'].append(smi_tokenizer(syn_smi))
#                     return_status['stor_src_data'].append(
#                         smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(syn_smi))
#                     return_status['stor_tgt_data'].append(smi_tokenizer(rea_smi))
#
#     # print(f"Cano Mean {np.mean(cano_dis)} Median {np.median(cano_dis)}")
#     # print(f"Root Mean {np.mean(root_dis)} Median {np.median(root_dis)}")
#     print(differ/len(src_data))
#
#     print('size', len(src_data))
#     for key,value in skip_dict.items():
#         print(f"{key}:{value},{value/len(reactants)}")
#     with open(
#             os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
#         for src in src_data:
#             f.write('{}\n'.format(src))
#
#     with open(
#             os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
#         for tgt in tgt_data:
#             f.write('{}\n'.format(tgt))
#     return src_data,tgt_data

"""multiprocess"""
def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None,root_aligned=True,character=False,processes=-1):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """

    if not os.path.exists(os.path.join(save_dir,"P2S")):
        os.makedirs(os.path.join(save_dir,"P2S"))
    if not os.path.exists(os.path.join(save_dir,"S2R")):
        os.makedirs(os.path.join(save_dir,"S2R"))

    if not os.path.exists(os.path.join(save_dir,"P2S",set_name)):
        os.makedirs(os.path.join(save_dir,"P2S",set_name))
    if not os.path.exists(os.path.join(save_dir,"S2R",set_name)):
        os.makedirs(os.path.join(save_dir,"S2R",set_name))


    data = [ {
        "reactant":i,
        "product":j,
        "augmentation":augmentation,
        "root_aligned":root_aligned,
    }  for i,j in zip(reactants,products)]
    ptos_src_data = []
    ptos_tgt_data = []
    stor_src_data = []
    stor_tgt_data = []
    skip_dict = {
        'invalid_p':0,
        'invalid_r':0,
        'small_p':0,
        'small_r':0,
        'error_mapping':0,
        'error_mapping_p':0,
        'empty_p':0,
        'empty_r':0,
    }
    processes = multiprocessing.cpu_count() if processes < 0 else processes
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(func=multi_process,iterable=data)
    pool.close()
    pool.join()
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue

        if character:
            for i in range(len(result['ptos_src_data'])):
                result['ptos_src_data'][i] = " ".join([char for char in "".join(result['ptos_src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['ptos_src_data'][i] = " ".join([char for char in "".join(result['ptos_src_data'][i].split())])
            for i in range(len(result['stor_src_data'])):
                result['stor_src_data'][i] = " ".join([char for char in "".join(result['stor_src_data'][i].split())])
            for i in range(len(result['stor_tgt_data'])):
                result['stor_tgt_data'][i] = " ".join([char for char in "".join(result['stor_tgt_data'][i].split())])
        ptos_src_data.extend(result['ptos_src_data'])
        ptos_tgt_data.extend(result['ptos_tgt_data'])
        stor_src_data.extend(result['stor_src_data'])
        stor_tgt_data.extend(result['stor_tgt_data'])

    print('size', len(ptos_src_data))
    for key,value in skip_dict.items():
        print(f"{key}:{value},{value/len(reactants)}")
    with open(
            os.path.join(save_dir, "P2S", set_name, 'src-{}.txt'.format(set_name)), 'w') as f:
        for src in ptos_src_data:
            f.write('{}\n'.format(src))

    with open(
            os.path.join(save_dir, "P2S", set_name, 'tgt-{}.txt'.format(set_name)), 'w') as f:
        for tgt in ptos_tgt_data:
            f.write('{}\n'.format(tgt))

    with open(
            os.path.join(save_dir, "S2R", set_name, 'src-{}.txt'.format(set_name)), 'w') as f:
        for src in stor_src_data:
            f.write('{}\n'.format(src))

    with open(
            os.path.join(save_dir, "S2R", set_name, 'tgt-{}.txt'.format(set_name)), 'w') as f:
        for tgt in stor_tgt_data:
            f.write('{}\n'.format(tgt))
    # return src_data,tgt_data


def multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status":0,
        "ptos_src_data":[],
        "ptos_tgt_data":[],
        "stor_src_data":[],
        "stor_tgt_data":[],
    }
    if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
        return_status["status"] = "error_mapping"
    if len(set(rids)) != len(rids):  # mapping is not 1:1
        return_status["status"] = "error_mapping"
    if len(set(pids)) != len(pids):  # mapping is not 1:1
        return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        product_adj = Chem.rdmolops.GetAdjacencyMatrix(pro_mol)
        product_adj = product_adj + np.eye(product_adj.shape[0])
        product_adj = product_adj.astype(np.bool)
        reactant_adj = Chem.rdmolops.GetAdjacencyMatrix(rea_mol)
        reactant_adj = reactant_adj + np.eye(reactant_adj.shape[0])
        reactant_adj = reactant_adj.astype(np.bool)
        patomidx2pmapidx = get_atomidx2mapidx(pro_mol)
        rmapidx2ratomidx = get_mapidx2atomidx(rea_mol)
        order = get_order(pro_mol, patomidx2pmapidx, rmapidx2ratomidx)
        target_adj = reactant_adj[order][:, order]
        reactant = reactant.split(".")
        synthons, ordered_reactants = get_smarts_pieces(pro_mol, product_adj,
                                                        target_adj, reactant)
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        synthons = synthons.split(".")
        reactants = ordered_reactants.split(".")
        if data['root_aligned']:
            # reversable = len(reactant) > 1
            reversable = False  # no shuffle

            """ptos"""
            product_roots = [-1]
            max_times = len(pro_atom_map_numbers)
            times = min(augmentation, max_times)
            if times < augmentation:  # times = max_times
                product_roots.extend(pro_atom_map_numbers)
                product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
            else:  # times = augmentation
                while len(product_roots) < times:
                    product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                    # pro_atom_map_numbers.remove(product_roots[-1])
                    if product_roots[-1] in product_roots[:-1]:
                        product_roots.pop()
            times = len(product_roots)
            syn_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", syn))) for syn in synthons]
            assert times == augmentation
            if reversable:
                times = int(times / 2)
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_synthons = []
                aligned_synthons_order = []
                used_indices = []
                for i, syn_map_number in enumerate(syn_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in syn_map_number:
                            syn_root = get_root_id(Chem.MolFromSmiles(synthons[i],sanitize=False),  root_map_number=map_number)
                            syn_smi = clear_map_canonical_smiles(synthons[i], canonical=True, root=syn_root,sanitize=False)
                            aligned_synthons.append(syn_smi)
                            aligned_synthons_order.append(j)
                            used_indices.append(i)
                            break
                sorted_synthons = sorted(list(zip(aligned_synthons, aligned_synthons_order)), key=lambda x: x[1])
                aligned_synthons = [item[0] for item in sorted_synthons]
                syn_smi = ".".join(aligned_synthons)
                product_tokens = smi_tokenizer(pro_smi)
                synthon_tokens = smi_tokenizer(syn_smi)
                return_status['ptos_src_data'].append(product_tokens)
                return_status['ptos_tgt_data'].append(synthon_tokens)

                if reversable:
                    aligned_synthons.reverse()
                    syn_smi = ".".join(aligned_synthons)
                    product_tokens = smi_tokenizer(pro_smi)
                    synthon_tokens = smi_tokenizer(syn_smi)
                    return_status['ptos_src_data'].append(product_tokens)
                    return_status['ptos_tgt_data'].append(synthon_tokens)
            """stor"""
            synthon_roots = []
            max_times = np.prod([len(map_numbers) for map_numbers in syn_atom_map_numbers])
            times = min(augmentation, max_times)
            j = 0
            while j < times:
                synthon_roots.append([random.sample(syn_atom_map_numbers[k], 1)[0] for k in range(len(synthons))])
                if synthon_roots[-1] in synthon_roots[:-1]:
                    synthon_roots.pop()
                else:
                    j += 1
            if j < augmentation:
                synthon_roots.extend(random.choices(synthon_roots, k=augmentation - times))
                times = augmentation
            if reversable:
                times = int(times / 2)
            for j in range(times):
                max_synthon_size = -1
                rooted_reactants, rooted_synthons = [], []
                pro_root_map_number = -1
                for k, (rea, syn) in enumerate(zip(reactants, synthons)):
                    if j + 1 == times:
                        syn_root_atom_map = -1
                    else:
                        syn_root_atom_map = synthon_roots[j][k]
                    syn_mol = Chem.MolFromSmiles(syn,sanitize=False)
                    syn_root = get_root_id(syn_mol, root_map_number=syn_root_atom_map)
                    syn_size = len(syn_mol.GetAtoms())
                    cano_atom_map = get_cano_map_number(syn, root=syn_root, sanitize=False)
                    syn = clear_map_canonical_smiles(syn, root=syn_root, sanitize=False)
                    root_map_number = cano_atom_map[0]
                    if pro_root_map_number == -1 or max_synthon_size < syn_size:
                        pro_root_map_number = root_map_number
                        max_synthon_size = syn_size
                    rea_mol = Chem.MolFromSmiles(rea)
                    rea_root = get_root_id(rea_mol, root_map_number=root_map_number)
                    rea = clear_map_canonical_smiles(rea, root=rea_root)
                    rooted_reactants.append(rea)
                    rooted_synthons.append(syn)

                if len(rooted_reactants) < len(reactants):
                    for j in range(len(rooted_reactants), len(reactants)):
                        rooted_reactants.append(clear_map_canonical_smiles(reactants[j]))
                if len(rooted_synthons) < len(synthons):
                    for j in range(len(rooted_synthons), len(synthons)):
                        rooted_synthons.append(clear_map_canonical_smiles(synthons[j]))
                assert len(rooted_reactants) == len(reactants)
                assert len(rooted_synthons) == len(synthons)
                reactant = ".".join(rooted_reactants)
                synthon = ".".join(rooted_synthons)
                pro_mol = Chem.MolFromSmiles(product)
                pro_root = get_root_id(pro_mol, pro_root_map_number)
                pro_smi = clear_map_canonical_smiles(product, root=pro_root)
                # product = clear_map_canonical_smiles(product)
                product_tokens = smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(synthon)
                reactant_tokens = smi_tokenizer(reactant)
                return_status['stor_src_data'].append(product_tokens)
                return_status['stor_tgt_data'].append(reactant_tokens)
                if reversable:
                    rooted_reactants.reverse()
                    rooted_synthons.reverse()
                    reactant = ".".join(rooted_reactants)
                    synthon = ".".join(rooted_synthons)
                    product_tokens = smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(synthon)
                    reactant_tokens = smi_tokenizer(reactant)
                    return_status['stor_src_data'].append(product_tokens)
                    return_status['stor_tgt_data'].append(reactant_tokens)
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_synthons = ".".join([clear_map_canonical_smiles(syn,sanitize=False) for syn in synthons])
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactants])
            return_status['ptos_src_data'].append(smi_tokenizer(cano_product))
            return_status['ptos_tgt_data'].append(smi_tokenizer(cano_synthons))
            return_status['stor_src_data'].append(smi_tokenizer(cano_product) + " [PREDICT] " + smi_tokenizer(cano_synthons))
            return_status['stor_tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            syn_mols = [Chem.MolFromSmiles(syn,sanitize=False) for syn in cano_synthons.split(".")]
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol,doRandom=True)
                syn_smi = [Chem.MolToSmiles(syn_mol,doRandom=True) for syn_mol in syn_mols]
                rea_smi = [Chem.MolToSmiles(rea_mol,doRandom=True) for rea_mol in rea_mols]
                syn_smi = ".".join(syn_smi)
                rea_smi = ".".join(rea_smi)
                return_status['ptos_src_data'].append(smi_tokenizer(pro_smi))
                return_status['ptos_tgt_data'].append(smi_tokenizer(syn_smi))
                return_status['stor_src_data'].append(smi_tokenizer(pro_smi) + " [PREDICT] " + smi_tokenizer(syn_smi))
                return_status['stor_tgt_data'].append(smi_tokenizer(rea_smi))
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        # default='USPTO_full',
                        default='USPTO_50K',
                        help='dataset: USPTO_50K or USPTO-full')
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-seed",type=int,default=33)
    parser.add_argument("-processes",type=int,default=-1)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    parser.add_argument("-postfix",type=str,default="")
    args = parser.parse_args()
    print('preprocessing dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO_50K', 'USPTO_full','USPTO-MIT']
    print(args)
    if args.test_only:
        datasets = ['test']
    elif args.train_only:
        datasets = ['train']
    elif args.test_except:
        datasets = ['val', 'train']
    elif args.validastrain:
        datasets = ['test', 'val', 'train']
    else:
        datasets = ['test', 'val', 'train']
    random.seed(args.seed)
    if args.dataset == "USPTO-MIT":
        datadir = './dataset/{}'.format(args.dataset)
        savedir = './dataset/{}_P2S2R_aug{}'.format(args.dataset,args.augmentation)
        savedir += args.postfix
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i, data_set in enumerate(datasets):
            with open(os.path.join(datadir,f"{data_set}.txt"),"r") as f:
                reaction_list = f.readlines()
                if args.validastrain and data_set == "train":
                    with open(os.path.join(datadir, f"val.txt"), "r") as f:
                        reaction_list += f.readlines()

                reactant_smarts_list = list(
                    map(lambda x: x.split('>>')[0], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split('>>')[1], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split(' ')[0], product_smarts_list))

                multiple_product_indices = [i for i in range(len(product_smarts_list)) if "." in product_smarts_list[i]]
                for index in multiple_product_indices:
                    products = product_smarts_list[index].split(".")
                    for product in products:
                        reactant_smarts_list.append(reactant_smarts_list[index])
                        product_smarts_list.append(product)
                for index in multiple_product_indices[::-1]:
                    del reactant_smarts_list[index]
                    del product_smarts_list[index]

                preprocess(
                    savedir,
                    reactant_smarts_list,
                    product_smarts_list,
                    data_set,
                    args.augmentation,
                    reaction_types=None,
                    root_aligned=not args.canonical,
                    character=args.character,
                    processes=args.processes,
                )

    else:
        datadir = './dataset/{}'.format(args.dataset)
        savedir = './dataset/{}_triple_aug{}'.format(args.dataset, args.augmentation)


        savedir += args.postfix
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i, data_set in enumerate(datasets):
            csv_path = f"{datadir}/raw_{data_set}.csv"
            csv = pd.read_csv(csv_path)
            reaction_list = list(csv["reactants>reagents>production"])
            if args.validastrain and data_set == "train":
                csv_path = f"{datadir}/raw_val.csv"
                csv = pd.read_csv(csv_path)
                reaction_list += list(csv["reactants>reagents>production"])

            reactant_smarts_list = list(
                map(lambda x: x.split('>')[0], reaction_list))
            reactant_smarts_list = list(
                map(lambda x: x.split(' ')[0], reactant_smarts_list))
            reagent_smarts_list = list(
                map(lambda x: x.split('>')[1], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split('>')[2], reaction_list))
            product_smarts_list = list(
                map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
            print("Total Data Size", len(reaction_list))

            # reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
            sub_react_list = reactant_smarts_list
            sub_prod_list = product_smarts_list
            # duplicate multiple product reactions into multiple ones with one product each
            multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
            for index in multiple_product_indices:
                products = sub_prod_list[index].split(".")
                for product in products:
                    sub_react_list.append(sub_react_list[index])
                    sub_prod_list.append(product)
            for index in multiple_product_indices[::-1]:
                del sub_react_list[index]
                del sub_prod_list[index]
            preprocess(
                savedir,
                sub_react_list,
                sub_prod_list,
                data_set,
                args.augmentation,
                reaction_types=None,
                root_aligned=not args.canonical,
                character=args.character,
                processes=args.processes,
            )
