#!/usr/bin/env python3

from model_weights import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export tensorflow variables from checkpoint to numpy file')
    parser.add_argument('logdir', metavar='TF_LOGDIR', type=str,
                        help='the tensorflow log directory to import the latest checkpoint')
    parser.add_argument('save_to', metavar='SAVE_TO', type=str,
                        help='the location to save the .npy file to')
    args = parser.parse_args()
    if args.save_to == "-":
        args.save_to = "/dev/stdout"
    export_model_weights(args.logdir, args.save_to)
