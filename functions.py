import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import constants
from nasbench import api
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint

# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench_api = api.NASBench('.data/nasbench_only108.tfrecord')
nasbench_hashkeys = nasbench_api.hash_iterator()

'''
Encode & Decode
'''


def encode(mat, ops):
    def _sanity_check(idx1, opidx, idx2):
        if idx1 not in range(7):
            return False
        if idx2 not in range(7):
            return False
        if opidx not in range(4):
            return False
        return True

    encoded = []
    for inbound, outbound in zip(*np.array(mat).nonzero()):
        op = constants.ALLOWED_OPS.index(ops[outbound])
        assert _sanity_check(inbound, op, outbound)

        embed = [0] * constants.EMBED_SIZE
        embed[inbound] = 1
        embed[7 + op] = 1
        embed[7 + len(constants.ALLOWED_OPS) + outbound] = 1

        encoded.append(embed)

    assert np.array(encoded).shape == (9, constants.EMBED_SIZE)
    return encoded


def max_index(output):

    assert output.shape == (1, 9, 18)

    encoded = []
    outputs = [output_ for output_ in output[0]]

    for output in outputs:
        embed = [0 for _ in range(constants.EMBED_SIZE)]
        inbound, op, outbound = np.argmax(output[:7]), np.argmax(output[7:-7]), np.argmax(output[-7:])
        embed[inbound] = 1
        embed[7 + op] = 1
        embed[7 + len(constants.ALLOWED_OPS) + outbound] = 1
        assert np.sum(np.array(embed)) == 3
        encoded.append(embed)

    assert np.array(encoded).shape == (9, constants.EMBED_SIZE)
    return encoded


def decode(encoded):
    mat = np.zeros((7, 7))
    ops = [constants.INPUT] + [constants.CONV3X3 for _ in range(5)] + [constants.OUTPUT]

    for embed in encoded:
        inbound, op, outbound = np.nonzero(np.array(embed))[0]
        op -= 7
        outbound -= (7 + len(constants.ALLOWED_OPS))
        ops[outbound] = constants.ALLOWED_OPS[op]
        if inbound > outbound:
            (inbound, outbound) = (outbound, inbound)
        elif inbound == outbound:
            continue
        mat[inbound][outbound] = 1

    return mat.tolist(), ops


'''
Generate Data
'''


class DataLoader(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.cur = start

    def __run_experiment(self, mat, op):
        model_spec = api.ModelSpec(matrix=mat, ops=op)
        q = nasbench_api.query(model_spec)
        return q['validation_accuracy'], q['training_time']

    def next(self, batchsize):
        global nasbench_hashkeys

        xset = []
        yset = []
        if self.cur == self.end:
            self.cur = self.end
        for idx in range(self.cur, min(self.cur + batchsize, self.end)):
            hash_val = nasbench_hashkeys[idx]
            fixed_stat, _ = nasbench_api.get_metrics_from_hash(hash_val)
            mat, op = fixed_stat['module_adjacency'], fixed_stat['module_operations']
            x = encode(mat, op)
            y = self.__run_experiment(mat, op)
            xset.append(x)
            yset.append(y)
        xset = torch.FloatTensor(xset)
        yset = torch.FloatTensor(yset)
        return xset, yset


def train_val_test_split(batch_size=100, validation_ratio=0.1, test_ratio=0.1):
    global nasbench_hashkeys

    revised = []
    for hash_val in nasbench_hashkeys:
        fixed_stat, _ = nasbench_api.get_metrics_from_hash(hash_val)
        mat, op = fixed_stat['module_adjacency'], fixed_stat['module_operations']
        if np.sum(mat) == constants.EDGE_NUM:
            revised.append(hash_val)

    total_size = (len(revised) // batch_size) * batch_size
    nasbench_hashkeys = revised[: total_size]
    val_size = int(total_size * validation_ratio)
    test_size = int(total_size * test_ratio)

    return DataLoader(val_size+test_size, total_size), DataLoader(0, val_size), DataLoader(val_size, val_size+test_size)


'''
Model
'''


class ConcatConv1d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConcatConv1d, self).__init__()
        self._layer = nn.Conv1d(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


def norm(dim):
    return nn.GroupNorm(dim, dim)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=constants.RTOL, atol=constants.ATOL, method = 'adams')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        ret = x.view(-1, shape)
        return ret


'''
For training
'''


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = constants.LR * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def accuracy(model, dataloader):
    datasize = dataloader.end - dataloader.start
    x, y = dataloader.next(datasize)
    x = torch.FloatTensor(x)
    x = x.to(constants.DEVICE)
    y = y.numpy()
    result = model(x).cpu().detach().numpy()
    total_correct = np.sum(np.square(result - y))
    return total_correct / datasize


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)