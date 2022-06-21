from rdkit import Chem
import os
import argparse
import re

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-ptos_src', type=str,required=True,help="The path of products")
    parser.add_argument('-ptos_results', type=str,required=True,help="The path of P2S results")
    parser.add_argument('-ptos_beam_size', type=int,default=10)
    parser.add_argument('-ptos_topk', type=int,default=3,help="The number of topk P2S results that used for S2R predictions")
    parser.add_argument('-stor_targets', type=str,required=True,help="The path of ground truth / reactants")
    parser.add_argument('-augmentation', type=int,default=20)
    parser.add_argument('-savedir', type=str,required=True)

    opt = parser.parse_args()
    print(opt)
    shuffle = True
    with open(opt.ptos_results, 'r') as f:
        lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
        print(len(lines))
        data_size = len(lines) // opt.ptos_beam_size
        ptos_pred = [[] for i in range(data_size)]  # data_len x beam_size
        for i, line in enumerate(lines):
            ptos_pred[i // opt.ptos_beam_size].append(line)
    print('Reading ptos_src from file ...')
    with open(opt.ptos_src, 'r') as f:
        lines = f.readlines()
        print("Origin File Length", len(lines))
        sources = [''.join(lines[i].strip().split(' ')) for i in range(0, data_size * opt.augmentation, opt.augmentation)]
    print('Reading stor_targets from file ...')
    with open(opt.stor_targets, 'r') as f:
        lines = f.readlines()
        print("Origin File Length", len(lines))
        targets = [''.join(lines[i].strip().split(' ')) for i in range(0, data_size * opt.augmentation, opt.augmentation)]
    assert len(ptos_pred) == len(targets)
    with open(os.path.join(opt.savedir,"ptos-src-test.txt"),"w") as f:
        for i in range(len(ptos_pred)):
            pro = Chem.CanonSmiles(sources[i])
            for j in range(opt.ptos_topk):
                syn_mols = [Chem.MolFromSmiles(syn, sanitize=False) for syn in ptos_pred[i][j].split(".")]
                if shuffle:
                    for k in range(int(opt.augmentation / 2)):
                        if k == 0:
                            syn = smi_tokenizer(ptos_pred[i][j]).split(".")
                        else:
                            syn = smi_tokenizer(".".join([Chem.MolToSmiles(syn_mol,doRandom=True) for syn_mol in syn_mols])).split(" . ")
                        f.write(smi_tokenizer(pro)+ " [PREDICT] "  + " . ".join(syn))
                        f.write("\n")
                        syn.reverse()
                        f.write(smi_tokenizer(pro)+ " [PREDICT] "  + " . ".join(syn))
                        f.write("\n")
                else:
                    for k in range(opt.augmentation):
                        if k == 0:
                            f.write(smi_tokenizer(pro) + " [PREDICT] "  + smi_tokenizer(ptos_pred[i][j]))
                            f.write("\n")
                        else:
                            f.write(smi_tokenizer(pro) + " [PREDICT] " + smi_tokenizer(".".join([Chem.MolToSmiles(syn_mol,doRandom=True) for syn_mol in syn_mols])))
                            f.write("\n")
    with open(os.path.join(opt.savedir,"ptos-tgt-test.txt"),"w") as f:
        for i in range(len(targets)):
            for j in range(opt.ptos_topk):
                for k in range(opt.augmentation):
                    f.write(smi_tokenizer(targets[i]))
                    f.write("\n")
