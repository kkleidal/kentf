import tensorflow as tf
import os
import re
import attrdict
import yaml
import numpy as np
import sys
import attrdict

DEFAULT_CKPT_REGEX = r"^model\.ckpt-(\d+)\.meta$"

def load_model(sess, path, ckpt_regex=DEFAULT_CKPT_REGEX):
    best = None
    best_val = 0
    for fname in os.listdir(path):
        match = re.match(ckpt_regex, fname)
        if match is not None and (best is None or int(match.group(1)) > best_val):
            best = fname
            best_val = int(match.group(1))
    meta_file = os.path.join(path, best)
    ckpt_file = meta_file[:-len(".meta")]
    new_saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    new_saver.restore(sess, ckpt_file)

def remove_number(name):
    name = re.sub(r":\d+$", "", name, count=1)
    return name

def get_model_weights(logdir, config=None, ckpt_regex=DEFAULT_CKPT_REGEX):
    g = tf.Graph()
    with g.as_default():
        yamlf = os.path.join(logdir, "config.yaml")
        if config is None:
            if os.path.exists(yamlf):
                with open(yamlf, "r") as f:
                    config = attrdict.AttrDict(yaml.load(f))
            else:
                config = {}
        npy = {"config": config, "variables": {}}
        with tf.Session() as sess:
            load_model(sess, logdir, ckpt_regex=ckpt_regex)
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                name = remove_number(var.name)
                npy["variables"][name] = sess.run(var)
        return npy


def export_model_weights(logdir, save_to, config=None, ckpt_regex=DEFAULT_CKPT_REGEX):
    npy = get_model_weights(logdir, config=config, ckpt_regex=ckpt_regex)
    np.save(save_to, npy)

class Model:
    def __init__(self, filename):
        model = np.load(filename).tolist()
        self.variables = model["variables"]
        self.config = attrdict.AttrDict(model["config"])

    def map_variables(self, mapfn):
        new = {}
        for key, value in self.variables.items():
            res = mapfn(key, value)
            if res is not None:
                new_key, new_value = res
                new[new_key] = new_value
        self.variables = new

    def map_variable_names(self, name_mapping_dict, remove_unmentioned=False):
        if remove_unmentioned:
            return self.map_variables(lambda key, val: \
                    None if key not in name_mapping_dict else (name_mapping_dict[key], val)) 
        else:
            return self.map_variables(lambda key, val: \
                    (key, val) if key not in name_mapping_dict else (name_mapping_dict[key], val)) 

    def load_into_tensorflow(self, create_vars=True, log_warnings=True, log_matches=False, name=None):
        with tf.name_scope(name or "load-model-into-tf"):
            tfvars = {}
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                tfvars[remove_number(var.name)] = var
            ops = []
            for var, val in self.variables.items():
                if var in tfvars:
                    if log_matches:
                        print("Found var %s" % var, file=sys.stderr)
                    tfvar = tfvars[var]
                elif create_vars:
                    if log_matches:
                        print("Creating var %s" % var, file=sys.stderr)
                    tfvar = tf.get_variable(var, initializer=val)
                elif log_warnings:
                    print("kentf.model_weights WARNING: Could not make loader for %s" % var,
                            file=sys.stderr)
                    continue
                else:
                    continue
                ops.append(tfvar.assign(val))
            return tf.group(*ops)

    def save(self, to_file):
        np.save(to_file, {"variables": self.variables, "config": self.config.__dict__})

