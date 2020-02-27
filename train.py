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
from torchdiffeq import odeint

# device = torch.device('cuda:0')
DEVICE = torch.device('cpu')

train_dataset, test_dataset, train_eval_dataset = f.generate_data(constants.BATCH_SIZE)
print("Completed data generation")
print("Size is", len(train_dataset))

input_size = 9 * constants.EMBED_SIZE
feature_layers = [f.ODEBlock(f.ODEfunc(9))]
fc_layers = [f.Flatten(), nn.Linear(input_size, 2)]
model = nn.Sequential(*feature_layers, *fc_layers).to(DEVICE)

criterion = nn.MSELoss().to(DEVICE)

batches_per_epoch = len(train_dataset) // constants.BATCH_SIZE

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

print('Number of parameters: {}'.format(f.count_parameters(model)))
print("Training Start")

for itr in range(constants.NEPOCHS):
    for batch_turn in range(batches_per_epoch):
        x, y = f.data_loader(dataset=train_dataset, batch_turn=batch_turn, size=constants.BATCH_SIZE)
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

    random.shuffle(train_dataset)
    with torch.no_grad():
        train_acc = f.accuracy(model, train_eval_dataset)
        val_acc = f.accuracy(model, test_dataset)
        '''if val_acc > best_acc:
                torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_acc = val_acc
            '''
        print(
                "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | Train Acc {:.4f} | Test Acc {:.4f}".format( itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg, b_nfe_meter.avg, train_acc, val_acc )
            )

