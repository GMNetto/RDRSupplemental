import socket
import numpy as np
import yaml
import argparse
import os, sys
import os.path as osp

from utils.easydict import EasyDict



def postprocess(args):
    # -------------------- miscellaneous --------------------
    args.allow_less_points = hasattr(args, 'allow_less_points') and args.allow_less_points

    # -------------------- learning --------------------
    if not args.evaluate:
        # -------------------- init --------------------
        if not hasattr(args, 'init'):
            args.init = 'xavier'
        if not hasattr(args, 'gain'):
            args.gain = 1.

        # -------------------- custom lr --------------------
        if hasattr(args, 'custom_lr') and args.custom_lr:
            args.lrs = [float(item) for item in args.lrs.split(',')][::-1]
            args.lr_switch_epochs = [int(item) for item in args.lr_switch_epochs.split(',')][::-1]
            assert (len(args.lrs) == len(args.lr_switch_epochs))

            diffs = [second - first for first, second in zip(args.lr_switch_epochs, args.lr_switch_epochs[1:])]
            assert (np.all(np.array(diffs) < 0))

            args.lr = args.lrs[-1]

    # -------------------- resume --------------------
    if args.evaluate:
        assert (hasattr(args, 'resume'))
        assert args.resume is not False

    return args


parser = argparse.ArgumentParser()
parser.add_argument('--pc1', action='store')
parser.add_argument('--pc2', action='store')
parser.add_argument('--out', action='store')
parser.add_argument('--out_proj', action='store')
parser.add_argument('--out_match', action='store')
parser.add_argument('--out_target', action='store')
parser.add_argument('--out_target_match', action='store')
parser.add_argument('--noise', action='store')
parser.add_argument('--vis', action='store_true')

def parse_args_from_cmd(args):
    args2, _ = parser.parse_known_args()
    if args2.pc1 is not None:
        args.pc1 = args2.pc1
    if args2.pc2 is not None:
        args.pc2 = args2.pc2
    if args2.out is not None:
        args.out = args2.out
    if args2.out_proj is not None:
        args.out_proj = args2.out_proj
    if args2.out_match is not None:
        args.out_match = args2.out_match
    if args2.out_target is not None:
        args.out_target = args2.out_target
    if args2.out_target_match is not None:
        args.out_target_match = args2.out_target_match
    if args2.noise is not None:
        args.DATASET.NOISE_TYPE = args2.noise
    if args2.vis is not None and args2.vis:
        args.vis = True
    else:
        args.vis = False
    return args

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
        args = postprocess(args)
    return args

