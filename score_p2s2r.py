from rdkit import Chem
import argparse
from tqdm import tqdm
import multiprocessing

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def canonicalize_smiles_clear_map(smiles,return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles,sanitize=not opt.synthon)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles,sanitize=not opt.synthon) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size,key=lambda x:x[1],reverse=True)[0][0],return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '',''
        else:
            return ''


def compute_mix_rank(prediction,beam_size,ptos_topk):
    valid_score = [[[k for k in range(beam_size)] for j in range(len(prediction[0]))] for i in range(ptos_topk)]
    invalid_rates = [0 for k in range(beam_size)]

    rank = {}
    max_frag_rank = {}
    for i in range(ptos_topk): # 3
        for j in range(len(prediction[i])):  # 20
            for k in range(len(prediction[i][j])): # 10
                if prediction[i][j][k][0] == "":
                    valid_score[i][j][k] = opt.beam_size + 1
                    invalid_rates[k] += 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[i][j], valid_score[i][j])), key=lambda x: x[1]) if i[0][0] != ""]
            prediction[i][j] = list(set(de_error))
            prediction[i][j].sort(key=de_error.index)
            for k, data in enumerate(prediction[i][j]):
                if data in rank:
                    rank[data] += 1 / (0.1 * k + 1 + 1.0 * i)
                else:
                    rank[data] = 1 / (0.1 * k + 1 + 1.0 * i)
    return rank, invalid_rates


def compute_rank(prediction):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    max_frag_rank = {}
    for j in range(len(prediction)):
        for k in range(len(prediction[j])):
            if prediction[j][k][0] == "":
                valid_score[j][k] = opt.beam_size + 1
                invalid_rates[k] += 1
        # error detection and deduplication
        de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if i[0][0] != ""]
        prediction[j] = list(set(de_error))
        prediction[j].sort(key=de_error.index)
        for k, data in enumerate(prediction[j]):
            if data in rank:
                rank[data] += 1 / (1.0 * k + 1)
            else:
                rank[data] = 1 / (1.0 * k + 1)
    return rank,invalid_rates


def main(opt):
    print('Reading predictions from file ...')
    with open(opt.predictions, 'r') as f:
        # lines = f.readlines()
        lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
        print(len(lines))
        data_size = len(lines) // (opt.ptos_topk * opt.augmentation * opt.beam_size) if opt.length == -1 else opt.length
        lines = lines[:data_size * (opt.ptos_topk * opt.augmentation * opt.beam_size)]
        print("Canonicalizing predictions using Process Number ",opt.process_number)
        pool = multiprocessing.Pool(processes=opt.process_number)
        lines = pool.map(func=canonicalize_smiles_clear_map,iterable=lines)
        pool.close()
        pool.join()

        predictions = [[[[] for j in range(opt.augmentation)] for k in range(opt.ptos_topk)] for i in range(data_size)]  # data_len x ptos_topk x augmentation x beam_size
        for i, line in enumerate(lines):
            data_idx = i // (opt.beam_size * opt.augmentation * opt.ptos_topk)
            ptos_topk = i % (opt.beam_size * opt.augmentation * opt.ptos_topk) // (opt.beam_size * opt.augmentation)
            augmentation_idx = i % (opt.beam_size * opt.augmentation * opt.ptos_topk) % (opt.augmentation * opt.beam_size) // opt.beam_size
            predictions[data_idx][ptos_topk][augmentation_idx].append(line)

    print('Reading targets from file ...')
    with open(opt.targets, 'r') as f:
        lines = f.readlines()
        print("Origin File Length", len(lines))
        targets = [canonicalize_smiles_clear_map(''.join(lines[i].strip().split(' '))) for i in tqdm(range(0,data_size * opt.augmentation * opt.ptos_topk,opt.augmentation * opt.ptos_topk))]

    ground_truth = targets
    print("Origin Target Lentgh, ", len(ground_truth))
    print("Cutted Length, ",data_size)

    accuracy = [0 for j in range(opt.n_best)]
    max_frag_accuracy = [0 for j in range(opt.n_best)]
    invalid_rates = [0 for j in range(opt.beam_size)]
    sorted_invalid_rates = [0 for j in range(opt.beam_size)]
    unique_rates = 0
    ranked_results = []
    for i in tqdm(range(len(predictions))):
        mix_rank = []
        mix_results = []
        rank,invalid_rate = compute_mix_rank(predictions[i],opt.beam_size,opt.ptos_topk,)
        rank = list(zip(rank.keys(),rank.values()))
        rank.sort(key=lambda x:x[1],reverse=True)
        rank = rank[:opt.n_best]
        for item in rank:
            mix_rank.append(item)
            mix_results.append(item[0][0])

        ranked_results.append(mix_results)
        for j, item in enumerate(mix_rank):
            if item[0][0] == ground_truth[i][0]:
                for k in range(j,opt.n_best):
                    accuracy[k] += 1
                break
        for j, item in enumerate(mix_rank):
            if item[0][1] == ground_truth[i][1]:
                for k in range(j,opt.n_best):
                    max_frag_accuracy[k] += 1
                break
        for j in range(len(mix_rank),opt.beam_size):
            sorted_invalid_rates[j] += 1
        unique_rates += len(mix_rank)

    for i in range(opt.n_best):
        print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i+1,accuracy[i] / data_size * 100,max_frag_accuracy[i] / data_size * 100),
              " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(invalid_rates[i] / data_size / opt.augmentation * 100,sorted_invalid_rates[i] / data_size / opt.augmentation * 100))
    print("Unique Rates:{:.3f}%".format(unique_rates / data_size / opt.beam_size * 100))

    if opt.save_file != "":
        with open(opt.save_file,"w") as f:
            for res in ranked_results:
                for smi in res:
                    f.write(smi)
                    f.write("\n")
                for i in range(len(res),opt.n_best):
                    f.write("")
                    f.write("\n")

"""Detailed Scoring Mode not completed"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_p2s2r.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=10,help='Beam size')
    parser.add_argument('-n_best', type=int, default=10,help='n best')
    parser.add_argument('-predictions', type=str, required=True,
                        help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, required=True, help="Path to file containing targets")
    parser.add_argument('-augmentation', type=int, default=1)
    parser.add_argument('-ptos_topk', type=int, default=3)
    parser.add_argument('-length', type=int, default=-1)
    parser.add_argument('-process_number', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-synthon', action="store_true", default=False)
    parser.add_argument('-save_file', type=str,default="")

    opt = parser.parse_args()
    print(opt)
    #    opt.beam_size = 10
    main(opt)
