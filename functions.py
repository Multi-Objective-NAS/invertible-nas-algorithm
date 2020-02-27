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
from torchdiffeq import odeint


# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench_api = api.NASBench('nasbench_only108.tfrecord')


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
    def find_max_index(ls):
        return ls.index(max(ls))

    encoded = []
    outputs = [output[i: i + constants.EMBED_SIZE] for i in range(0, len(output), constants.EMBED_SIZE)]
    for output in outputs:
        embed = [0 for _ in range(constants.EMBED_SIZE)]
        inbound, op, outbound = find_max_index(output[:7]), find_max_index(output[7:-7]), find_max_index(output[-7:])
        embed[inbound] = 1
        embed[7 + op] = 1
        embed[7 + len(constants.ALLOWED_OPS) + outbound] = 1
        encoded += embed

    assert np.sum(np.array(encoded)) == 3
    return encoded

'''
def decode(encoded):
    mat = np.zeros((7, 7))
    op = [constants.INPUT] + [constants.CONV3X3 for _ in range(5)] + [constants.OUTPUT]

    outputs = [output[i: i + constants.EMBED_SIZE] for i in range(0, len(output), constants.EMBED_SIZE)]
    for embed in outputs:
        inbound, op, outbound = np.nonzero(np.array(embed))[0]
        op -= 7
        outbound -= (7 + len(constants.ALLOWED_OPS))
        mat[inbound][outbound] = 1
        op[outbound] = constants.ALLOWED_OPS[op]

    return mat.tolist(), op
'''

'''
Generate Data
'''


def generate_data(batch_size=100, validation_ratio=0.1, test_ratio=0.1):
    def run_experiment(mat, op):
        model_spec = api.ModelSpec(matrix=mat, ops=op)
        q = nasbench_api.query(model_spec)

        return q['validation_accuracy'], q['training_time']

    def train_val_test_split(dataset):
        random.shuffle(dataset)
        total_size = (len(dataset) // batch_size) * batch_size
        dataset = dataset[: total_size]
        val_size = int(total_size * validation_ratio)
        test_size = int(total_size * test_ratio)

        val, dataset = dataset[:val_size], dataset[val_size:]
        test, train = dataset[:test_size], dataset[test_size:]

        return train, val, test

    def is_target_graph(mat, op):
        return np.sum(mat) == constants.EDGE_NUM

    # Find every possible graph
    dataset = []
    for hash_val in nasbench_api.hash_iterator():
        fixed_stat, _ = nasbench_api.get_metrics_from_hash(hash_val)
        mat, op = fixed_stat['module_adjacency'], fixed_stat['module_operations']
        if is_target_graph(mat, op):
            x = encode(mat, op)
            y = run_experiment(mat, op)
            dataset.append((x, y))

    return train_val_test_split(dataset)


def data_loader(dataset, batch_turn, size):
    xset = []
    yset = []
    for x, y in dataset[size * batch_turn: size * batch_turn + size]:
        xset.append(x)
        yset.append(y)
    xset = torch.FloatTensor(xset)
    yset = torch.FloatTensor(yset)
    return xset, yset


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
        #batch_t = T_series[:x.shape[0]]
        out = odeint(self.odefunc, x, self.integration_time , rtol=constants.RTOL, atol=constants.ATOL, method = 'adams')
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


def accuracy(model, dataset):
    x, y = data_loader(dataset=dataset, batch_turn=0, size=len(dataset))
    x = torch.FloatTensor(x)
    x = x.to(constants.DEVICE)
    y = y.numpy()
    result = model(x).cpu().detach().numpy()
    total_correct = np.sum(np.square(result - y))
    return total_correct / len(dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)