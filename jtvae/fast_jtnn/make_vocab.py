import os, pickle, sys
from tqdm import tqdm
os.getcwd()
os.chdir("jtvae")
os.listdir()

data_dir = "data/zinc/train"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

smiles = set()

for f in tqdm(files):
    with open(f, "rb") as file:
        mol_batch = pickle.load(file)
    smiles = smiles.union(*[{node.smiles for node in mol_tree.nodes} for mol_tree in mol_batch])

    
len(smiles)

new_vocab = "\n".join(list(smiles))

new_vocab_path = "/mnt/data/shared/jacob/CHEMIR/jtvae/data/zinc/new_vocab.txt"
with open(new_vocab_path, "w") as f:
    f.write(new_vocab)
new_vocab2 = [x.strip("\r\n ") for x in open(new_vocab_path)] 

vocab_path = "/mnt/data/shared/jacob/CHEMIR/jtvae/data/zinc/vocab.txt"
vocab = [x.strip("\r\n ") for x in open(vocab_path)] 

len(set(vocab).intersection(set(new_vocab2)))
len(set(vocab) - set(new_vocab2))
len(set(new_vocab2) - set(vocab))