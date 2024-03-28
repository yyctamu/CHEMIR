
import os, pickle, sys
from tqdm import tqdm
os.getcwd()
# os.chdir("jtvae")
os.chdir("..")
sys.path.append(".")
from fast_jtnn import *
os.listdir()

processed_trees = []

vocab_path = "data/zinc/new_vocab.txt"
vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
vocab = Vocab(vocab)

data_folder = "data/zinc/train"
data_files = [fn for fn in os.listdir(data_folder) if fn != 'processed']
print(data_files)

save_dir = os.path.join(data_folder, "processed")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for split_id in tqdm(range(len(data_files))):
    fn = data_files[split_id]
    fn = os.path.join(data_folder, fn)
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    dataset = MolTreeDataset([data], vocab, True)[0]
    with open(os.path.join(save_dir, 'tensors-%d.pkl' % split_id), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

