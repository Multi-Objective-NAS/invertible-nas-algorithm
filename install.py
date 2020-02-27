import subprocess
import os

subprocess.call('pip install git+https://github.com/rtqichen/torchdiffeq')
subprocess.call('git clone https://github.com/rtqichen/torchdiffeq')

subprocess.call('git clone https://github.com/google-research/nasbench', shell=True )
current = os.getcwd()
os.chdir(current + '/.data')
subprocess.call('curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord', shell=True)
#subprocess.call('curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord', shell = True)