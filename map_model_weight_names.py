#!/usr/bin/env python3

from model_weights import *
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='See what variables are stored in model.npy file')
    parser.add_argument('-c', '--clean', dest='remove_unmapped', action='store_true',
                                help='remove variables not mentioned in the mapping file')
    parser.add_argument('load_from', metavar='LOAD_FROM', type=str,
                        help='the model.npy file to load from')
    parser.add_argument('mapping_file', metavar='MAPPING_YAML', type=str,
                        help='the YAML file containing the mapping from the old variable name to the new variable name')
    parser.add_argument('save_to', metavar='SAVE_TO', type=str,
                        help='the model.npy file to write to')
    args = parser.parse_args()
    if args.load_from == "-":
        args.load_from = "/dev/stdin"
    if args.mapping_file == "-":
        args.mapping_file = "/dev/stdin"
    if args.save_to == "-":
        args.save_to = "/dev/stdout"
    with open(args.mapping_file, "r") as f:
        mapping = yaml.load(f)
    m = Model(args.load_from)
    m.map_variable_names(mapping, remove_unmentioned=args.remove_unmapped)
    m.save(args.save_to)
