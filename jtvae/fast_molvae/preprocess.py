import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle as pickle
sys.path.append("..")
from fast_jtnn import *
import rdkit
import datetime
i=0
def tensorize(smiles, assm=True):
    # global i
    # i += 1
    # if i % 1000 == 0:
    #     print(i, datetime.datetime.now())

    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol
    return mol_tree

if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    all_data = pool.map(tensorize, data)

    le = (len(all_data) + num_splits - 1) // num_splits
    print(f"le: {le}, splits: {num_splits}, numer: {(len(all_data) + num_splits - 1)}, le * splits: {le * num_splits}")
    if le * num_splits < (len(all_data) + num_splits - 1):
        print('incrementing le')
        le += 1
    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

# python preprocess.py --train ../data/zinc/train.txt --split 100 --jobs 20