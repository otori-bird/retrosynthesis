import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import textdistance
import multiprocessing

from rdkit import Chem
from tqdm import tqdm


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def align_reactant_products(reactants,products):
    reactants = reactants.split(".")
    products = products.split(".")
    pattern = re.compile(r"(?<=:)\d+")
    reactants_number = [re.findall(pattern,reactant) for reactant in reactants]
    products_number = [re.findall(pattern,product) for product in products]
    new_reactants = []
    new_products = []
    selected_reactants = []
    selected_products = []
    for p,num_p in enumerate(products_number):
        overlap_percent = [len(list(set(num_r) & set(num_p))) / len(set(num_p)) for num_r in reactants_number]
        if np.max(overlap_percent) >= 0.5:
            r = np.argmax(overlap_percent)
            if r not in selected_reactants:
                selected_reactants.append(r)
                selected_products.append(p)
                new_reactants.append(reactants[r])
                new_products.append(products[p])

    for p in range(len(products)):
        if p not in selected_products:
            new_products.append(products[p])
    for r in range(len(reactants)):
        if r not in selected_reactants:
            new_reactants.append(reactants[r])

    assert len(new_reactants) == len(reactants)
    assert len(new_products) == len(products)
    return ".".join(new_reactants), ".".join(new_products)


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
# def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None, separated=False,root_aligned=True,character=False):
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
#         reactant = reactant.split(".")
#         product = product.split(".")
#         rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
#         max_times = np.prod([len(map_numbers) for map_numbers in rea_atom_map_numbers])
#         times = min(augmentation,max_times)
#         reactant_roots = [[-1 for _ in reactant]]
#         j = 0
#         while j < times:
#             reactant_roots.append([random.sample(rea_atom_map_numbers[k], 1)[0] for k in range(len(reactant))])
#             if reactant_roots[-1] in reactant_roots[:-1]:
#                 reactant_roots.pop()
#             else:
#                 j += 1
#         if j < augmentation:
#             reactant_roots.extend(random.choices(reactant_roots,k=augmentation-times))
#             times = augmentation
#         reversable = False # no reverse
#         assert times == augmentation
#         if reversable:
#             times = int(times / 2)
#
#         pro_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", pro))) for pro in product]
#         full_pro_atom_map_numbers = set(map(int, re.findall(r"(?<=:)\d+", ".".join(product))))
#         for k in range(times):
#             tmp = list(zip(reactant, reactant_roots[k], rea_atom_map_numbers))
#             random.shuffle(tmp)
#             reactant_k, reactant_roots_k, rea_atom_map_numbers_k = [i[0] for i in tmp], [i[1] for i in tmp], [i[2] for i in tmp]
#         # for k in range(times):
#         #     tmp = list(zip(reactant,reactant_roots[k]))
#         #     random.shuffle(tmp)
#         #     reactant_k, reactant_roots_k = [i[0] for i in tmp],[i[1] for i in tmp]
#             aligned_reactants = []
#             aligned_products = []
#             aligned_products_order = []
#             all_atom_map = []
#             for i,rea in enumerate(reactant_k):
#                 rea_root_atom_map = reactant_roots_k[i]
#                 rea_root = get_root_id(Chem.MolFromSmiles(rea),root_map_number=rea_root_atom_map)
#                 cano_atom_map = get_cano_map_number(rea,rea_root)
#                 if cano_atom_map is None:
#                     print(f"Reactant {index} Failed to find Canonical Mol with Atom MapNumber")
#                     continue
#                 rea_smi = clear_map_canonical_smiles(rea,canonical=True,root=rea_root)
#                 aligned_reactants.append(rea_smi)
#                 all_atom_map.extend(cano_atom_map)
#
#             for i, pro_map_number in enumerate(pro_atom_map_numbers):
#                 reactant_candidates = []
#                 selected_reactant = []
#                 for j, map_number in enumerate(all_atom_map):
#                     if map_number in pro_map_number:
#                         for rea_index, rea_atom_map_number in enumerate(rea_atom_map_numbers_k):
#                             if map_number in rea_atom_map_number and rea_index not in selected_reactant:
#                                 selected_reactant.append(rea_index)
#                                 reactant_candidates.append((map_number, j, len(rea_atom_map_number)))
#
#                 # select maximal reactant
#                 reactant_candidates.sort(key=lambda x: x[2], reverse=True)
#                 map_number = reactant_candidates[0][0]
#                 j = reactant_candidates[0][1]
#                 pro_root = get_root_id(Chem.MolFromSmiles(product[i]), root_map_number=map_number)
#                 pro_smi = clear_map_canonical_smiles(product[i], canonical=True, root=pro_root)
#                 aligned_products.append(pro_smi)
#                 aligned_products_order.append(j)
#
#             # for i, pro_map_number in enumerate(pro_atom_map_numbers):
#             #     for j, map_number in enumerate(all_atom_map):
#             #         # select mapping reactans
#             #         if map_number in pro_map_number:
#             #             pro_root = get_root_id(Chem.MolFromSmiles(product[i]),root_map_number=map_number)
#             #             pro_smi = clear_map_canonical_smiles(product[i],canonical=True,root=pro_root)
#             #             aligned_products.append(pro_smi)
#             #             aligned_products_order.append(j)
#             #             break
#             sorted_products = sorted(list(zip(aligned_products,aligned_products_order)),key=lambda x:x[1])
#             aligned_products = [item[0] for item in sorted_products]
#             pro_smi = ".".join(aligned_products)
#             rea_smi = ".".join(aligned_reactants)
#             product_tokens = smi_tokenizer(pro_smi)
#             reactant_tokens = smi_tokenizer(rea_smi)
#             src_data.append(reactant_tokens)
#             tgt_data.append(product_tokens)
#             if reversable:
#                 aligned_reactants.reverse()
#                 aligned_products.reverse()
#                 pro_smi = ".".join(aligned_products)
#                 rea_smi = ".".join(aligned_reactants)
#                 product_tokens = smi_tokenizer(pro_smi)
#                 reactant_tokens = smi_tokenizer(rea_smi)
#                 src_data.append(reactant_tokens)
#                 tgt_data.append(product_tokens)
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
def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None, separated=False,root_aligned=True,character=False,processes=-1):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = [ {
        "reactant":i,
        "product":j,
        "augmentation":augmentation,
        "root_aligned":root_aligned,
        "separated":separated,
    }  for i,j in zip(reactants,products)]
    src_data = []
    tgt_data = []
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
    edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        if character:
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
        edit_distances.append(result['edit_distance'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
    print(np.mean(edit_distances))
    print('size', len(src_data))
    for key,value in skip_dict.items():
        print(f"{key}:{value},{value/len(reactants)}")
    if augmentation != 999:
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
    reactant = data['reactant']
    augmentation = data['augmentation']
    separated = data['separated']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status":0,
        "src_data":[],
        "tgt_data":[],
        "edit_distance":0,
    }
    # if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(rids)) != len(rids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(pids)) != len(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
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
        reactant = reactant.split(".")
        if data['root_aligned']:
            product = product.split(".")
            rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
            max_times = np.prod([len(map_numbers) for map_numbers in rea_atom_map_numbers])
            times = min(augmentation, max_times)
            reactant_roots = [[-1 for _ in reactant]]
            j = 0
            while j < times:
                reactant_roots.append([random.sample(rea_atom_map_numbers[k], 1)[0] for k in range(len(reactant))])
                if reactant_roots[-1] in reactant_roots[:-1]:
                    reactant_roots.pop()
                else:
                    j += 1
            if j < augmentation:
                reactant_roots.extend(random.choices(reactant_roots, k=augmentation - times))
                times = augmentation
            reversable = False  # no reverse
            assert times == augmentation
            if reversable:
                times = int(times / 2)

            pro_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", pro))) for pro in product]
            full_pro_atom_map_numbers = set(map(int, re.findall(r"(?<=:)\d+", ".".join(product))))
            for k in range(times):
                tmp = list(zip(reactant, reactant_roots[k],rea_atom_map_numbers))
                random.shuffle(tmp)
                reactant_k, reactant_roots_k,rea_atom_map_numbers_k = [i[0] for i in tmp], [i[1] for i in tmp], [i[2] for i in tmp]
                aligned_reactants = []
                aligned_products = []
                aligned_products_order = []
                all_atom_map = []
                for i, rea in enumerate(reactant_k):
                    rea_root_atom_map = reactant_roots_k[i]
                    rea_root = get_root_id(Chem.MolFromSmiles(rea), root_map_number=rea_root_atom_map)
                    cano_atom_map = get_cano_map_number(rea, rea_root)
                    if cano_atom_map is None:
                        print(f"Reactant Failed to find Canonical Mol with Atom MapNumber")
                        continue
                    rea_smi = clear_map_canonical_smiles(rea, canonical=True, root=rea_root)
                    aligned_reactants.append(rea_smi)
                    all_atom_map.extend(cano_atom_map)

                for i, pro_map_number in enumerate(pro_atom_map_numbers):
                    reactant_candidates = []
                    selected_reactant = []
                    for j, map_number in enumerate(all_atom_map):
                        if map_number in pro_map_number:
                            for rea_index, rea_atom_map_number in enumerate(rea_atom_map_numbers_k):
                                if map_number in rea_atom_map_number and rea_index not in selected_reactant:
                                    selected_reactant.append(rea_index)
                                    reactant_candidates.append((map_number, j, len(rea_atom_map_number)))

                    # select maximal reactant
                    reactant_candidates.sort(key=lambda x: x[2], reverse=True)
                    map_number = reactant_candidates[0][0]
                    j = reactant_candidates[0][1]
                    pro_root = get_root_id(Chem.MolFromSmiles(product[i]), root_map_number=map_number)
                    pro_smi = clear_map_canonical_smiles(product[i], canonical=True, root=pro_root)
                    aligned_products.append(pro_smi)
                    aligned_products_order.append(j)

                sorted_products = sorted(list(zip(aligned_products, aligned_products_order)), key=lambda x: x[1])
                aligned_products = [item[0] for item in sorted_products]
                pro_smi = ".".join(aligned_products)
                if separated:
                    reactants = []
                    reagents = []
                    for i,cano_atom_map in enumerate(rea_atom_map_numbers_k):
                        if len(set(cano_atom_map) & full_pro_atom_map_numbers) > 0:
                            reactants.append(aligned_reactants[i])
                        else:
                            reagents.append(aligned_reactants[i])
                    rea_smi = ".".join(reactants)
                    reactant_tokens = smi_tokenizer(rea_smi)
                    if len(reagents) > 0 :
                        reactant_tokens += " <separated> " + smi_tokenizer(".".join(reagents))
                else:
                    rea_smi = ".".join(aligned_reactants)
                    reactant_tokens = smi_tokenizer(rea_smi)
                product_tokens = smi_tokenizer(pro_smi)
                return_status['src_data'].append(reactant_tokens)
                return_status['tgt_data'].append(product_tokens)
                if reversable:
                    aligned_reactants.reverse()
                    aligned_products.reverse()
                    pro_smi = ".".join(aligned_products)
                    rea_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(rea_smi)
                    return_status['src_data'].append(reactant_tokens)
                    return_status['tgt_data'].append(product_tokens)
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant])
            return_status['src_data'].append(smi_tokenizer(cano_reactanct))
            return_status['tgt_data'].append(smi_tokenizer(cano_product))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol,doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol,doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(rea_smi))
                return_status['tgt_data'].append(smi_tokenizer(pro_smi))
        edit_distances = []
        for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        default='USPTO_50K')
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-seed",type=int,default=33)
    parser.add_argument("-processes",type=int,default=-1)
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-train_only", action="store_true")
    parser.add_argument("-test_except", action="store_true")
    parser.add_argument("-validastrain", action="store_true")
    parser.add_argument("-separated", action="store_true")
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
        datasets = ['valid', 'train']
    elif args.validastrain:
        datasets = ['test', 'val', 'train']
    else:
        datasets = ['test', 'val', 'train']

    random.seed(args.seed)
    if args.dataset == "USPTO-MIT":
        datadir = './dataset/{}'.format(args.dataset)
        savedir = './dataset/{}_RtoP_aug{}'.format(args.dataset,args.augmentation)
        if args.separated:
            savedir += "_separated"
        savedir += args.postfix
        print(savedir)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for i, data_set in enumerate(datasets):
            with open(os.path.join(datadir,f"{data_set}.txt"),"r") as f:
                reaction_list = f.readlines()
                if args.validastrain and data_set == "train":
                    print("validastrain")
                    with open(os.path.join(datadir, f"val.txt"), "r") as f:
                        reaction_list += f.readlines()
                reactant_smarts_list = list(
                    map(lambda x: x.split('>>')[0], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split('>>')[1], reaction_list))
                product_smarts_list = list(
                    map(lambda x: x.split(' ')[0], product_smarts_list))

                save_dir = os.path.join(savedir, data_set)
                src_data, tgt_data = preprocess(
                    save_dir,
                    reactant_smarts_list,
                    product_smarts_list,
                    data_set,
                    args.augmentation,
                    reaction_types=None,
                    separated=args.separated,
                    root_aligned=not args.canonical,
                    character=args.character,
                    processes=args.processes
                )

    else:
        datadir = './dataset/{}'.format(args.dataset)
        savedir = './dataset/{}_RtoP_aug{}'.format(args.dataset, args.augmentation)
        if args.separated:
            savedir += "_separated"

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
            sub_react_list = reactant_smarts_list
            sub_prod_list = product_smarts_list
            save_dir = os.path.join(savedir, data_set)
            src_data, tgt_data = preprocess(
                save_dir,
                sub_react_list,
                sub_prod_list,
                data_set,
                args.augmentation,
                reaction_types=None,
                separated=args.separated,
                root_aligned=not args.canonical,
                character=args.character,
                processes=args.processes
            )
