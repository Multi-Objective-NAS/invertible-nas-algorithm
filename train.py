import functions as f
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import constants
import sys
import time
import psutil

# device = torch.device('cuda:0')
DEVICE = torch.device('cpu')

train_dataloader, test_dataloader, val_dataloader = f.train_val_test_split(batch_size=constants.BATCH_SIZE, validation_ratio=0.1, test_ratio=0.1)
print("Completed data generation")

input_size = 9 * constants.EMBED_SIZE
feature_layers = [f.ODEBlock(f.ODEfunc(9))]
fc_layers = [f.norm(9), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), f.Flatten(), nn.Linear(9, 2)]

model = nn.Sequential(*feature_layers, *fc_layers).to(DEVICE)

criterion = nn.MSELoss().to(DEVICE)

batches_per_epoch = (train_dataloader.end - train_dataloader.start) // constants.BATCH_SIZE

lr_fn = f.learning_rate_with_decay(
        constants.BATCH_SIZE, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
)

optimizer = torch.optim.SGD(model.parameters(), lr=constants.LR, momentum=0.9)

best_acc = 0
batch_time_meter = f.RunningAverageMeter()
f_nfe_meter = f.RunningAverageMeter()
b_nfe_meter = f.RunningAverageMeter()
end = time.time()

print("Training Start")

for itr in range(constants.NEPOCHS):
    for batch_turn in range(batches_per_epoch):
        x, y = train_dataloader.next(constants.BATCH_SIZE)

        print(psutil.cpu_percent())
        print("memory use:", psutil.virtual_memory()[2])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr * batches_per_epoch + batch_turn)
        optimizer.zero_grad()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        result = logits.cpu().detach().numpy()
        loss = criterion(logits, y)
        print("[%d] Completed loss" % batch_turn)
        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()
        print("[%d] Completed backward" % batch_turn)
        nfe_backward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        batch_time_meter.update(time.time() - end)
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()
        print("[%d] Completed one batch" % batch_turn)

    with torch.no_grad():
        train_acc = f.accuracy(model, test_dataloader)
        val_acc = f.accuracy(model, val_dataloader)
        '''if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_acc = val_acc
            '''
        print(
                "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | Train Acc {:.4f} | Test Acc {:.4f}".format( itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg, b_nfe_meter.avg, train_acc, val_acc )
            )