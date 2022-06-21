import numpy as np
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


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def smarts2smiles(smarts, canonical=True):
    mol = Chem.MolFromSmiles(smarts, sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=canonical)


def del_index(smarts):
    t = re.sub(':\d*', '', smarts)
    return t


def onehot_encoding(x, total):
    return np.eye(total)[x]


def collate(data):
    return map(list, zip(*data))


def get_cano_map_number(smi,root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi,root=root))
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
# def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None, reagent=False):
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
#     for index in pbar:
#         pbar.set_description(f"Data Size:{len(src_data)}")
#         product = products[index]
#         reactant = reactants[index]
#
#         if len(src_data) % augmentation != 0:
#             print(index)
#             exit(1)
#
#         # if index < 2281:
#         #     continue
#
#         pro_mol = Chem.MolFromSmiles(product)
#         rea_mol = Chem.MolFromSmiles(reactant)
#
#         """checking data quality"""
#         rids = sorted(re.findall(pt, reactant))
#         pids = sorted(re.findall(pt, product))
#         if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if len(set(rids)) != len(rids) :  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if len(set(pids)) != len(pids) :  # mapping is not 1:1
#             skip_dict['error_mapping'] += 1
#             continue
#         if "" == product:
#             skip_dict['empty_p'] += 1
#             continue
#         if "" == reactant:
#             skip_dict['empty_r'] += 1
#             continue
#         if rea_mol is None:
#             skip_dict['invalid_r'] += 1
#             continue
#         if len(rea_mol.GetAtoms()) < 5:
#             skip_dict['small_r'] += 1
#             continue
#         if pro_mol is None:
#             skip_dict['invalid_p'] += 1
#             continue
#         if len(pro_mol.GetAtoms()) == 1:
#             skip_dict['small_p'] += 1
#             continue
#         if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
#             skip_dict['error_mapping_p'] += 1
#             continue
#         """finishing checking data quality"""
#
#         pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
#         # if re.findall(r"(?<=:)\d+", product) != re.findall(pt,product):
#         #     print(index)
#         #     print(pro_atom_map_numbers)
#         #     print(re.findall(pt,product))
#         #     exit(1)
#         reactant = reactant.split(".")
#         product_roots = [-1]
#         # reversable = len(reactant) > 1
#         reversable = False # no shuffle
#
#         max_times = len(pro_atom_map_numbers)
#         times = min(augmentation, max_times)
#         if times < augmentation: # times = max_times
#             product_roots.extend(pro_atom_map_numbers)
#             product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
#         else: # times = augmentation
#             while len(product_roots) < times:
#                 product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
#                 # pro_atom_map_numbers.remove(product_roots[-1])
#                 if product_roots[-1] in product_roots[:-1]:
#                     product_roots.pop()
#         times = len(product_roots)
#         assert times == augmentation
#         if reversable:
#             times = int(times / 2)
#         for k in range(times):
#             pro_root_atom_map = product_roots[k]
#             pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
#             cano_atom_map = get_cano_map_number(product,root=pro_root)
#             if cano_atom_map is None:
#                 print(f"Product {index} Failed to find Canonical Mol with Atom MapNumber")
#                 continue
#             pro_smi = clear_map_canonical_smiles(product,canonical=True,root=pro_root)
#             # root_map_number = cano_atom_map[0]
#
#             # rea_mols = [Chem.MolFromSmiles(rea) for rea in reactant]
#             # rea_atom_number = [len(mol.GetAtoms()) for mol in rea_mols]
#             # sorted_reactants = sorted(list(zip(reactant, rea_mols,rea_atom_number)),key=lambda x:x[2],reverse=True)
#             # reactant = [i[0] for i in sorted_reactants]
#
#             aligned_reactants = []
#             aligned_reactants_order = []
#             rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
#             used_indices = []
#             for i, rea_map_number in enumerate(rea_atom_map_numbers):
#                 for j, map_number in enumerate(cano_atom_map):
#                     # select mapping reactans
#                     if map_number in rea_map_number:
#                         rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]),root_map_number=map_number)
#                         rea_smi = clear_map_canonical_smiles(reactant[i],canonical=True,root=rea_root)
#                         aligned_reactants.append(rea_smi)
#                         aligned_reactants_order.append(j)
#                         used_indices.append(i)
#                         break
#             sorted_reactants = sorted(list(zip(aligned_reactants,aligned_reactants_order)),key=lambda x:x[1])
#             aligned_reactants = [item[0] for item in sorted_reactants]
#             if reagent:
#                 for i in range(len(reactant)):
#                     if i not in used_indices:
#                         aligned_reactants.append(clear_map_canonical_smiles(reactant[i]))
#             # aligned_reactants_order = [item[1] for item in sorted_reactants]
#
#             # product_smi = clear_map_canonical_smiles(pro_smi)
#             reactant_smi = ".".join(aligned_reactants)
#             # if reactant_smi != clear_map_canonical_smiles(reactant):
#             #     differ += 1
#             # product_tokens = smi_tokenizer(product_smi)
#             product_tokens = smi_tokenizer(pro_smi)
#             reactant_tokens = smi_tokenizer(reactant_smi)
#             src_data.append(product_tokens)
#             tgt_data.append(reactant_tokens)
#             # if Chem.MolFromSmiles(pro_smi).GetAtomWithIdx(0).GetSymbol() != Chem.MolFromSmiles(reactant_smi).GetAtomWithIdx(0).GetSymbol():
#             #     print(index)
#             #     print(product_tokens)
#             #     print(reactant_tokens)
#
#             # cano_dis.append(Levenshtein.distance(pro_smi,clear_map_canonical_smiles(reactant_smi)))
#             # root_dis.append(Levenshtein.distance(pro_smi,reactant_smi))
#             if reversable:
#                 aligned_reactants.reverse()
#                 reactant_smi = ".".join(aligned_reactants)
#                 product_tokens = smi_tokenizer(pro_smi)
#                 reactant_tokens = smi_tokenizer(reactant_smi)
#                 src_data.append(product_tokens)
#                 tgt_data.append(reactant_tokens)
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
def preprocess(save_dir, src, tgt,set_name,mode,delunk,vocab):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = [ {
        "src":i,
        "tgt":j,
        "vocab":vocab,
        "mode":mode,
        "delunk":delunk,
    }  for i,j in zip(src,tgt)]
    src_data = []
    tgt_data = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(func=multi_process,iterable=data)
    pool.close()
    pool.join()
    for result in tqdm(results):
        src_data.append(result['src_data'])
        tgt_data.append(result['tgt_data'])

    print('size', len(src_data))
    with open(
            os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
        for src in src_data:
            f.write('{}\n'.format(src))

    with open(
            os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
        for tgt in tgt_data:
            f.write('{}\n'.format(tgt))
    return src_data,tgt_data


def multi_process(data):

    src = data['src']
    tgt = data['tgt']
    mode = data['mode']
    delunk = data['delunk']
    vocab = data['vocab']
    return_status = {
        'src_data':"",
        'tgt_data':""
    }
    return_status['tgt_data'] = tgt
    src = src.split()
    if mode == "token":
        for index in range(len(src)):
            is_mask = random.random()
            if is_mask < 0.15:
                prob = random.random()
                if prob < 0.8:
                    if delunk:
                        src[index] = ""
                    else:
                        src[index] = "<unk>"
                elif prob < 0.9:
                    src[index] = random.sample(vocab,1)[0]
    elif mode == "span":
        mask_pos = int(random.random() * len(src))
        mask_len = int(len(src) * 0.15)
        for index in range(mask_pos,mask_pos + mask_len):
            if index >= len(src):
                index -= len(src)
            prob = random.random()
            if prob < 0.8:
                if delunk:
                    src[index] = ""
                else:
                    src[index] = "<unk>"
            elif prob < 0.9:
                src[index] = random.sample(vocab, 1)[0]

    return_status['src_data'] = " ".join(src)
    # if Chem.MolFromSmiles(pro_smi).GetAtomWithIdx(0).GetSymbol() != Chem.MolFromSmiles(reactant_smi).GetAtomWithIdx(0).GetSymbol():
    #     print(index)
    #     print(product_tokens)
    #     print(reactant_tokens)

    # cano_dis.append(Levenshtein.distance(pro_smi,clear_map_canonical_smiles(reactant_smi)))
    # root_dis.append(Levenshtein.distance(pro_smi,reactant_smi))
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        default='USPTO_full',)
    parser.add_argument("-mode",type=str,default="token")
    parser.add_argument("-delunk",action="store_true")
    args = parser.parse_args()
    savedir = f"./dataset/{args.dataset}_masked_{args.mode}"
    if args.delunk:
        savedir += "_delunk"
    dataset = f"./dataset/{args.dataset}"
    print(args)
    random.seed(33)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    vocab = set()
    print("Building vocab from training data")
    src = os.path.join(dataset,"train","src-train.txt")
    with open(src,"r") as f:
        src = [line.strip() for line in f.readlines()]
    for i in tqdm(src):
        vocab.update(i.split())
    print("Vocab size:", len(vocab))
    vocab = list(vocab)

    for data_set in ['test','val','train']:
        src = os.path.join(dataset,data_set,f"src-{data_set}.txt")
        tgt = os.path.join(dataset,data_set,f"src-{data_set}.txt")
        with open(src,"r") as f:
            src = [line.strip() for line in f.readlines()]
        with open(tgt,"r") as f:
            tgt = [line.strip() for line in f.readlines()]
        preprocess(os.path.join(savedir,data_set),src,tgt,data_set,args.mode,args.delunk,vocab)