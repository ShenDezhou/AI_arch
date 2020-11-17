import argparse
import json
import time
from types import SimpleNamespace

import cupy
import numpy
from data import Data
from model import APSP, APSPPowerLawBound

#1. apsp_config : 117.64s
#2. apsp_gpu_config: 32.19
#3. dpmm_config: 145.31
#4. dpmm_gpu_config: 35.67
#5. naive_apsp_config: 142.92
#6. naive_apsp_gpu_config: 36.28
MODEL_MAP = {
    'dpmm': APSP,
    'powerlaw':  APSPPowerLawBound
}

def main(config_file):
    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

    data = Data(config.train_file_path)
    mat = data.load_file()

    t = time.process_time()
    apsp = MODEL_MAP[config.model_type](mat, config)
    mr = apsp.apsp(g_diameter=config.diameter)
    te = time.process_time()
    print('time:',(te-t))
    print(mr.shape)
    if config.device=='cpu':
        numpy.save(mr, config.experiment_name+'.npz')
    else:
        cupy.save(mr, config.experiment_name+'.npz')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file', default='config/naive_apsp_config.json',
        help='model config file')

    args = parser.parse_args()

    main(args.config_file)