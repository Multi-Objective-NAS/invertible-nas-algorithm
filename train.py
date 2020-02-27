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
from nasbench import api
import sys
import time
import psutil

# device = torch.device('cuda:0')
DEVICE = torch.device('cpu')

train_dataloader, test_dataloader, val_dataloader = f.train_val_test_split(batch_size=constants.BATCH_SIZE,
                                                                           validation_ratio=0.1, test_ratio=0.1)
print("Completed data generation")

input_size = 9 * constants.EMBED_SIZE
ODEf = f.ODEfunc(9)
ODEb = f.ODEBlock(ODEf)
feature_layers = [ODEb]
fc_layers = [f.norm(9), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), f.Flatten(), nn.Linear(9, 2)]

model = nn.Sequential(*feature_layers, *fc_layers).to(DEVICE)

criterion = nn.MSELoss().to(DEVICE)
train_size = train_dataloader.end - train_dataloader.start
batches_per_epoch = train_size // constants.BATCH_SIZE

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
            "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | Train Acc {:.4f} | Test Acc {:.4f}".format(
                itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg, b_nfe_meter.avg,
                train_acc, val_acc)
        )

search_size = 1
for itr in range(min(train_size, search_size)):
    x, y = train_dataloader.next(1)
    y = np.transpose(y.numpy())

    # Linear
    weight = model[5].weight
    bias = model[5].bias.cpu().detach().numpy()
    bias = bias.reshape((2, 1))
    z_tn = np.subtract(y, bias)
    inverse_weight = np.linalg.pinv(weight.cpu().detach().numpy())
    z_tn = np.matmul(inverse_weight, z_tn)

    # AdaptiveAvgPool1d(1)
    z_tn = np.repeat(z_tn, constants.EMBED_SIZE, axis=1)
    z_tn = np.expand_dims(z_tn, axis=0)

    # ode back
    z_tn = torch.from_numpy(z_tn).to(DEVICE)
    z_t0 = f.odeint(ODEf, z_tn, torch.tensor([1, 0]).float().type_as(z_tn), rtol=constants.RTOL, atol=constants.ATOL,
                    method='adams')

    assert z_t0.shape[0] == (1, 9, 18)

    encoded = f.max_index(z_t0)
    mat, op = f.decode(encoded)
    print("y:", y)


    def __run_experiment(self, mat, op):
        model_spec = api.ModelSpec(matrix=mat, ops=op)
        q = f.nasbench_api.query(model_spec)
        return q['validation_accuracy'], q['training_time']


    print("result", __run_experiment(mat, op))
