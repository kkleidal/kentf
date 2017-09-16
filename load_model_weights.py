#!/usr/bin/env python3

from model_weights import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See what variables are stored in model.npy file')
    parser.add_argument('load_from', metavar='LOAD_FROM', type=str,
                        help='the model.npy file')
    args = parser.parse_args()
    if args.load_from == "-":
        args.load_from = "/dev/stdin"
    m = Model(args.load_from)
    for var, val in m.variables.items():
        print(var, val.shape)
