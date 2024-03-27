import torch, argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()

path = args.path
# path = "/mnt/data/shared/jacob/CHEMIR/GNN/data/zinc250k_property/processed/dev"
vocab_files = os.listdir(os.path.join(path, "vocab"))
tree_files = os.listdir(os.path.join(path, "preprocessed"))

vocab_files = sorted(vocab_files, key=lambda file: int(file.split('vocab')[1].split('.')[0].split('_')[0]))
tree_files = sorted(tree_files, key=lambda file: int(file.split('preprocessed')[1].split('.')[0].split('_')[0]))
print(f"Vocab files: {vocab_files}\n\nTree files: {tree_files}")
vocab = set(sum([torch.load(os.path.join(path, 'vocab', f)) for f in vocab_files], []))
trees = sum([torch.load(os.path.join(path, 'preprocessed', f)) for f in tree_files], [])
print(f"Vocab len: {len(vocab)}; Tree len: {len(trees)}")

torch.save(vocab, os.path.join(path, "vocab.pt"))
torch.save(trees, os.path.join(path, "trees.pt"))