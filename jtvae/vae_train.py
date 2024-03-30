import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys, os
import numpy as np
import argparse
from collections import deque
import pickle as pickle

from fast_jtnn import *
import rdkit

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

# https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

writer = SummaryWriter(args.save_dir)

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
print(model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
# scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
epoch_step = 0
beta = args.beta
meters = np.zeros(5)
epoch_meters = np.zeros(5)

writer.add_scalar("step/LR", scheduler.get_lr()[0], total_step)
writer.add_scalar("step/beta", beta, total_step)

nOOM = 0

for epoch in tqdm(range(args.epoch)):
    loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=args.num_workers)
    progress = tqdm(loader)
    for batch in progress:
        try:
            total_step += 1
            epoch_step += 1
            # try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, beta)
            # del batch
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            # except Exception as e:
            #     print(e)
            #     continue

            losses = np.array([loss.item(), kl_div, wacc * 100, tacc * 100, sacc * 100])
            meters = meters + losses
            epoch_meters = epoch_meters + losses

            if total_step % args.print_iter == 0:
                torch.cuda.empty_cache()
                meters /= args.print_iter
                writer.add_scalar("step/loss", meters[0], total_step)
                writer.add_scalar("step/KL", meters[1], total_step)
                writer.add_scalar("step/word", meters[2], total_step)
                writer.add_scalar("step/topo", meters[3], total_step)
                writer.add_scalar("step/assm", meters[4], total_step)
                writer.add_scalar("step/param norm", param_norm(model), total_step)
                writer.add_scalar("step/grad norm", grad_norm(model), total_step)
                # print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                # sys.stdout.flush()
                meters *= 0

            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                writer.add_scalar("step/LR", scheduler.get_lr()[0], total_step)
                # print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)
                writer.add_scalar("step/beta", beta, total_step)

            loss_str = f'Loss: {losses[0]:.3e} - KL: {losses[1]:.3e} - word: {losses[2]:.3e} - topo: {losses[3]:.3e} - assm: {losses[4]:.3e}'
            progress.set_postfix_str(loss_str)

            # del loss, batch
            # torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            nOOM += 1
            writer.add_scalar("step/OOM", nOOM, total_step)
            torch.cuda.empty_cache()


    epoch_meters /= epoch_step
    epoch_step = 0
    writer.add_scalar("epoch/loss", epoch_meters[0], epoch)
    writer.add_scalar("epoch/KL", epoch_meters[1], epoch)
    writer.add_scalar("epoch/word", epoch_meters[2], epoch)
    writer.add_scalar("epoch/topo", epoch_meters[3], epoch)
    writer.add_scalar("epoch/assm", epoch_meters[4], epoch)
    epoch_meters *= 0

torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))
writer.flush()

# CUDA_VISIBLE_DEVICES=0 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0 --num_workers=16 --max_beta=0
# CUDA_VISIBLE_DEVICES=1 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0.01 --num_workers=8 --max_beta=0.01
# CUDA_VISIBLE_DEVICES=2 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0.1 --num_workers=4 --max_beta=0.1

# CUDA_VISIBLE_DEVICES=3 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0.01_v2 --num_workers=8 --max_beta=0.01
# CUDA_VISIBLE_DEVICES=4 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0.1_v2 --num_workers=4 --max_beta=0.1
# CUDA_VISIBLE_DEVICES=1 python vae_train.py --train=data/zinc/train --vocab=data/zinc/new_vocab.txt --save_dir=outputs/max_beta0_v2 --num_workers=8 --max_beta=0
