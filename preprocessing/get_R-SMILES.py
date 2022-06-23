import numpy as np
import argparse
import re
import random
import textdistance

from rdkit import Chem


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


def get_forward_rsmiles(data):
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
    reactant = reactant.split(".")
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
    edit_distances = []
    for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
        edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
    return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


def get_retro_rsmiles(data):
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
        "src_data":[],
        "tgt_data":[],
        "edit_distance":0,
    }
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
    reactant = reactant.split(".")
    reversable = False  # no shuffle
    # augmentation = 100
    if augmentation == 999:
        product_roots = pro_atom_map_numbers
        times = len(product_roots)
    else:
        product_roots = [-1]
        # reversable = len(reactant) > 1

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
        assert times == augmentation
        if reversable:
            times = int(times / 2)
    # candidates = []
    for k in range(times):
        pro_root_atom_map = product_roots[k]
        pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
        cano_atom_map = get_cano_map_number(product, root=pro_root)
        if cano_atom_map is None:
            return_status["status"] = "error_mapping"
            return return_status
        pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
        aligned_reactants = []
        aligned_reactants_order = []
        rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
        used_indices = []
        for i, rea_map_number in enumerate(rea_atom_map_numbers):
            for j, map_number in enumerate(cano_atom_map):
                # select mapping reactans
                if map_number in rea_map_number:
                    rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                    rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                    aligned_reactants.append(rea_smi)
                    aligned_reactants_order.append(j)
                    used_indices.append(i)
                    break
        sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
        aligned_reactants = [item[0] for item in sorted_reactants]
        reactant_smi = ".".join(aligned_reactants)
        product_tokens = smi_tokenizer(pro_smi)
        reactant_tokens = smi_tokenizer(reactant_smi)

        return_status['src_data'].append(product_tokens)
        return_status['tgt_data'].append(reactant_tokens)

        if reversable:
            aligned_reactants.reverse()
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi)
            reactant_tokens = smi_tokenizer(reactant_smi)
            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)
    assert len(return_status['src_data']) == data['augmentation']
    edit_distances = []
    for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
        edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
    return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rxn',type=str,required=True)
    parser.add_argument('-mode',type=str,default="retro",)
    parser.add_argument('-forward_mode',type=str,default="separated",)
    parser.add_argument("-augmentation",type=int,default=1)
    parser.add_argument("-seed",type=int,default=33)
    args = parser.parse_args()
    print(args)
    reactant,reagent,product = args.rxn.split(">")
    pt = re.compile(r':(\d+)]')
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    if len(rids) == 0 or len(pids) == 0:
        print("No atom mapping found!")
        exit(1)
    if args.mode == "retro":
        args.input = product
        args.output = reactant
    else:
        args.input = reactant
        args.output = product

    print("Original input:", args.input)
    print("Original output:",args.output)
    src_smi = clear_map_canonical_smiles(args.input)
    tgt_smi = clear_map_canonical_smiles(args.output)
    if src_smi == "" or tgt_smi == "":
        print("Invalid SMILES!")
        exit(1)
    print("Canonical input:", src_smi)
    print("Canonical output:",tgt_smi)

    mapping_check = True
    if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
        mapping_check = False
    if len(set(rids)) != len(rids):  # mapping is not 1:1
        mapping_check = False
    if len(set(pids)) != len(pids):  # mapping is not 1:1
        mapping_check = False
    if not mapping_check:
        print("The quality of the atom mapping may not be good enough, which can affect the effect of root alignment.")
    data = {
        'product':product,
        'reactant':reactant,
        'augmentation':args.augmentation,
        'separated':args.forward_mode == "separated"
    }
    if args.mode == "retro":
        res = get_retro_rsmiles(data)
    else:
        res = get_forward_rsmiles(data)
    for index,(src,tgt) in enumerate(zip(res['src_data'], res['tgt_data'])):
        print(f"ID:{index}")
        print(f"R-SMILES input:{''.join(src.split())}")
        print(f"R-SMILES output:{''.join(tgt.split())}")
    print("Avg. edit distance:", res['edit_distance'])
