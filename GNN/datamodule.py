import torch, os, random
from pytorch_lightning import LightningDataModule
from dig.ggraph.method.JTVAE.fast_jtnn import PairTreeFolder, MolTreeDataset
# from dig.ggraph.dataset import ZINC250k, ZINC800
from torch.utils.data import Dataset, DataLoader

class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        pin_memory: bool,
        num_workers: int,
        limit: int,
    ):
        super().__init__()
        print('Loading data...')
        # self.dataset = ZINC250k(one_shot=False, root=data_dir)
        # self.smiles = torch.load(os.path.join(self.dataset.processed_dir, "data.pt"))[-1]
        self.vocab = list(torch.load(os.path.join(data_dir, "vocab.pt")))
        self.trees = torch.load(os.path.join(data_dir, "dev/preprocessed.pt"))
        if limit > 0 or limit is not None:
            # self.dataset.slices = {k:v[:limit] for k,v in self.dataset.slices.items()}
            self.smiles = self.smiles[:limit]
            self.trees = self.trees[:limit]
        # assert len(self.trees) == len(self.dataset)
        # assert self.dataset[0].smile == self.smiles[0]
        # assert self.dataset[len(self.dataset) // 2].smile == self.smiles[len(self.dataset) // 2]
        # assert self.dataset[len(self.dataset)].smile == self.smiles[len(self.dataset)]
        # print(f'Loaded data ({len(self.dataset)}), smiles ({len(self.smiles)}), vocab ({len(self.vocab)}), and trees ({len(self.trees)})')
        print(f'Loaded vocab ({len(self.vocab)}) and trees ({len(self.trees)})')

        self.save_hyperparameters(logger=False)

    def train_dataloader(self):
        return MolTreeFolder(preprocessed_data=self.trees,
                             vocab=self.vocab,
                             batch_size=self.hparams.batch_size,
                             num_workers=self.hparams.num_workers,
                             shuffle=True)

    # def val_dataloader(self):

    # def test_dataloader(self):

class MolTreeFolder(object):

    def __init__(self, preprocessed_data, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.preprocessed_data = preprocessed_data
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        #         for fn in self.data_files:

        if self.shuffle:
            random.shuffle(self.preprocessed_data)  # shuffle data before batch

        batches = [self.preprocessed_data[i: i + self.batch_size]
                   for i in range(0, len(self.preprocessed_data), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()

        dataset = MolTreeDataset(batches, self.vocab, self.assm)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=self.num_workers, collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        del batches, dataset, dataloader