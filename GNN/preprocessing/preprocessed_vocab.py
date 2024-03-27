import torch, os

from dig.ggraph.dataset import ZINC250k, ZINC800
from dig.ggraph.method.JTVAE import fast_jtnn
from dig.ggraph.method.JTVAE import jtvae

from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int)
parser.add_argument("--stop", type=int)
args = parser.parse_args()

print('Loading Zinc...')
dataset = ZINC250k(one_shot=False, root="../data")
smiles = torch.load(os.path.join(dataset.processed_dir, "data.pt"))[-1]
print('Loaded!')
processed_dir = dataset.processed_dir
del dataset
smiles = smiles[args.start:args.stop]

cset = set()
for smile in tqdm(smiles):
    mol = fast_jtnn.MolTree(smile)
    for c in mol.nodes:
        cset.add(c.smiles)
cset = list(cset)

torch.save(cset, os.path.join(processed_dir, "vocab", f"vocab{args.start}_{args.stop}.pt"))
print('Vocab built!')

jtvae = jtvae.JTVAE(list_smiles=smiles, build_vocab=False)
jtvae.vocab = cset

preprocessed = jtvae.preprocess(smiles)

torch.save(preprocessed, os.path.join(processed_dir, "preprocessed", f"preprocessed{args.start}_{args.stop}.pt"))