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
def preprocess(save_dir, products,set_name, augmentation=1, reaction_types=None, reagent=False,character=False):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = [ {
        "product":j,
        "augmentation":augmentation,
    }  for j in products]
    src_data = []
    tgt_data = []
    skip_dict = {
        'invalid_p':0,
        'small_p':0,
    }
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(func=multi_process,iterable=data)
    pool.close()
    pool.join()
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        if character:
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])

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
    pt = re.compile(r':(\d+)]')
    product = data['product']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    """checking data quality"""
    return_status = {
        "status":0,
        "src_data":[],
        "tgt_data":[],
    }
    if pro_mol is None or "" == product:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:

        # # produdct
        # pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        # product_roots = [-1]
        # # reversable = len(reactant) > 1
        # reversable = False  # no shuffle
        #
        # max_times = len(pro_atom_map_numbers)
        # times = min(augmentation, max_times)
        # if times < augmentation:  # times = max_times
        #     product_roots.extend(pro_atom_map_numbers)
        #     product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
        # else:  # times = augmentation
        #     while len(product_roots) < times:
        #         product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
        #         # pro_atom_map_numbers.remove(product_roots[-1])
        #         # if product_roots[-1] in product_roots[:-1]:
        #         #     product_roots.pop()
        # times = len(product_roots)
        # assert times == augmentation
        # for k in range(times):
        #     pro_root_atom_map = product_roots[k]
        #     pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
        #     pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
        #     product_tokens = smi_tokenizer(pro_smi)
        #     return_status['src_data'].append(product_tokens)
        #     return_status['tgt_data'].append(product_tokens)
        #
        # # reactant
        smi = product.split(".")
        smi = [clear_map_canonical_smiles(s) for s in smi]
        mols = [Chem.MolFromSmiles(s) for s in smi]
        for k in range(augmentation):
            smi = ".".join([Chem.MolToSmiles(mol,doRandom=True) for mol in mols])
            tokens = smi_tokenizer(smi)
            return_status['src_data'].append(tokens)
            return_status['tgt_data'].append(tokens)

    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-seed",type=int,default=33)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-character", action="store_true")
    parser.add_argument("-canonical", action="store_true")
    parser.add_argument("-mode", type=str, default="product")
    args = parser.parse_args()
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
    if args.mode == "reactant":
        postfix = "_reactant"
    elif args.mode == "product":
        postfix = "_product"
    else:
        postfix = ""
    assert args.mode in ['both','product','reactant']
    datadir = './dataset/USPTO_full'
    savedir = './dataset/USPTO_full_pretrain_aug{}'.format(args.augmentation)

    savedir += postfix
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

        random.shuffle(reaction_list)
        reactant_smarts_list = list(
            map(lambda x: x.split('>')[0], reaction_list))
        reactant_smarts_list = list(
            map(lambda x: x.split(' ')[0], reactant_smarts_list))
        product_smarts_list = list(
            map(lambda x: x.split('>')[2], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
        print("Total Data Size", len(reaction_list))

        save_dir = os.path.join(savedir, data_set)

        if args.mode == "reactant":
            sub_prod_list = reactant_smarts_list
        elif args.mode == "product":
            sub_prod_list = product_smarts_list
            # duplicate multiple product reactions into multiple ones with one product each
            multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
            for index in multiple_product_indices:
                products = sub_prod_list[index].split(".")
                for product in products:
                    sub_prod_list.append(product)
            for index in multiple_product_indices[::-1]:
                del sub_prod_list[index]
        else:
            sub_prod_list = product_smarts_list
            # duplicate multiple product reactions into multiple ones with one product each
            multiple_product_indices = [i for i in range(len(sub_prod_list)) if "." in sub_prod_list[i]]
            for index in multiple_product_indices:
                products = sub_prod_list[index].split(".")
                for product in products:
                    sub_prod_list.append(product)
            for index in multiple_product_indices[::-1]:
                del sub_prod_list[index]
            sub_prod_list += reactant_smarts_list

        src_data, tgt_data = preprocess(
            save_dir,
            sub_prod_list,
            data_set,
            args.augmentation,
            reaction_types=None,
            reagent=False,
            character=args.character
        )
