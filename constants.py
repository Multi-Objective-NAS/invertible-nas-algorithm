import torch
'''
Main Argument
'''
# device = torch.device('cuda:0')
DEVICE = torch.device('cpu')
BATCH_SIZE = 50
LR = 0.1
NEPOCHS = 1
RTOL = 1e-7
ATOL = 1e-9
EDGE_NUM = 9

'''
Data Set
'''
# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3, OUTPUT]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix
EMBED_SIZE = 7 + 7 + len(ALLOWED_OPS)