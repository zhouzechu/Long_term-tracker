import argparse
import os.path as osp

from loguru import logger

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task

import torch
import time
from videoanalyst.model import builder as model_builder
def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser

if __name__ == "__main__":
    # test the running speed
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    use_gpu = True
    model = model_builder.build("track", task_cfg.model)
    x = torch.randn(1, 3, 303, 303)
    z_1 = torch.randn(1, 256, 3, 3)
    z_2 = torch.randn(1, 256, 3, 3)
    # model = onnx.load("test.onnx")
    if use_gpu:
        model = model.cuda()
        x = x.cuda()
        z_1 = z_1.cuda()
        z_2 = z_2.cuda()
    # oup = model(x, z)
    # x = x.numpy()
    # z = z.numpy()
    #
    # torch.onnx.export(model, (x, z), 'test.onnx', input_names=['x', 'z'], output_names=['output'])

    T_w = 20  # warmup
    T_t = 200  # test
    with torch.no_grad():
        for i in range(T_w):
            oup = model(x, z_1, z_2)
        t_s = time.time()
        for i in range(T_t):
            oup = model(x, z_1, z_2)
        t_e = time.time()

    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))

