#!/usr/bin/env python3

import sys
import argparse
import os
import argparse
import yaml
import json
from attrdict import AttrDict

class ConfigKWArg:
    def __init__(self, name, example, type=str, default=None, required=False, short_name=None, nargs=1, help=None):
        self.name = name
        self.type = type
        self.default = default
        if nargs != 1:
            self.default = []
        self.required = required
        self.short_name = short_name
        self.nargs = nargs
        self.help = help
        self.example = example

    def add_argument_to(self, parser):
        args = []
        kwargs = {}
        if self.short_name is not None:
            args.append("-%s" % self.short_name)
        args.append("--%s" % self.name)
        kwargs["dest"] = self.name
        if isinstance(self.type, str) and self.type.startswith("file"):
            kwargs["type"] = str
        else:
            kwargs["type"] = self.type
        if self.nargs != 1:
            kwargs["nargs"] = self.nargs
        if self.help is not None:
            kwargs["help"] = self.help
        kwargs["default"] = None
        parser.add_argument(*args, **kwargs)

    def get_example(self):
        if self.nargs != 1 and not isinstance(self.example, list):
            return [self.example]
        else:
            return self.example

    def update_config(self, config, config_from_file):
        if config.get(self.name, None) is None:
            config[self.name] = config_from_file.get(self.name, None)
        elif self.nargs != 1:
            config[self.name] = config.get(self.name, []) + config_from_file.get(self.name, [])

    def postprocess_config(self, config):
        if config.get(self.name, None) is None:
            if self.required:
                raise MissingArgumentError(self.name)
            config[self.name] = self.default
            return
        if isinstance(self.type, str) and self.type.startswith("file"):
            if self.type == "filew" or self.type == "filer":
                filenames = resolve_filenames(config[self.name], mode=self.type[-1])
                if len(filenames) > 1:
                    raise MultipleFilesError()
                if len(filenames) == 0:
                    raise NoSuchFileError()
                config[self.name] == filenames[0]
            elif self.type == "filesw" or self.type == "filesr":
                if isinstance(config[self.name], str):
                    filenames = resolve_filenames(config[self.name], mode=self.type[-1])
                else:
                    filenames = []
                    for name in config[self.name]:
                        filenames.extend(resolve_filenames(name, mode=self.type[-1]))
                config[self.name] = filenames


class ConfigHelper:
    def __init__(self, desc):
        self.desc = desc
        self.args = []
        self.reserved_long = {'help', 'yaml-config', 'json-config', 'yaml-stub', 'json-stub', 'save-json-config', 'save-yaml-config'}
        self.reserved_short = {'help', 'yc', 'jc'}
        self.args_long_hash = {}
        self.args_short_hash = {}

    def add_argument(self, arg):
        self.args.append(arg)
        if arg.name in self.reserved_long:
            raise DuplicateArgumentError(arg.name)
        else:
            self.args_long_hash[arg.name] = arg
            self.reserved_long.add(arg.name)
        if arg.short_name is not None:
            if arg.short_name in self.reserved_short:
                raise DuplicateArgumentError(arg.short_name)
            else:
                self.args_short_hash[arg.short_name] = arg
                self.reserved_short.add(arg.short_name)

    def _make_stub(self, fileobj, dump, config=None):
        obj = {arg.name: arg.get_example() for arg in self.args}
        if config is not None:
            obj = {arg.name: config[arg.name] for arg in self.args if arg.name in config}
        dump(obj, fileobj)

    def make_yaml_stub(self, fileobj, config=None):
        self._make_stub(fileobj, lambda x, f: yaml.dump(x, f, indent=2, default_flow_style=False), config)

    def make_json_stub(self, fileobj, config=None):
        self._make_stub(fileobj, lambda x, f: json.dump(x, f, indent=2, separators=(',', ': '), sort_keys=True), config)

    def parse_args(self):
        parser = argparse.ArgumentParser(self.desc)
        for arg in self.args:
            arg.add_argument_to(parser)
        parser.add_argument('-yc', '--yaml-config', dest="yaml_config", type=str, default=None,
            help="YAML configuration file to load to specify default args")
        parser.add_argument('-jc', '--json-config', dest="json_config", type=str, default=None,
            help="JSON configuration file to load to specify default args")
        parser.add_argument('--save-yaml-config', dest="save_yaml_config", type=str, default=None,
            help="Save configuration to this YAML file")
        parser.add_argument('--save-json-config', dest="save_json_config", type=str, default=None,
            help="Save configuration to this JSON file")
        parser.add_argument('--yaml-stub', dest="yaml_stub", action='store_true',
            help="Make stub YAML config file with example arguments")
        parser.add_argument('--json-stub', dest="json_stub", action='store_true',
            help="Make stub JSON config file with example arguments")
        config = AttrDict(vars(parser.parse_args()))
        if config.yaml_stub:
            with FileOpener(sys.stdout, "w") as f:
                self.make_yaml_stub(f)
            sys.exit(0)
        if config.json_stub:
            with FileOpener(sys.stdout, "w") as f:
                self.make_json_stub(f)
            sys.exit(0)
        if config.yaml_config is not None or config.json_config is not None:
            if config.yaml_config is not None and config.json_config is not None:
                raise RuntimeError("YAML config and JSON config files specified. You can only choose 1.")
            elif config.yaml_config is not None:
                with FileOpener(config.yaml_config, "r") as f:
                    config_file = yaml.load(f)
            elif config.json_config is not None:
                with FileOpener(config.json_config, "r") as f:
                    config_file = json.load(f)
            for arg in self.args:
                arg.update_config(config, config_file)
        for arg in self.args:
            arg.postprocess_config(config)
        if config.save_yaml_config:
            with FileOpener(config.save_yaml_config, "w") as f:
                self.make_yaml_stub(f, config=config)
        if config.save_json_config:
            with FileOpener(config.save_json_config, "w") as f:
                self.make_json_stub(f, config=config)
        del config["yaml_config"]
        del config["json_config"]
        del config["save_yaml_config"]
        del config["save_json_config"]
        del config["yaml_stub"]
        del config["json_stub"]
        return config

def resolve_filenames(filename, mode="w"):
    if filename == "-":
        if len(mode) == 0 or mode[0] == "r":
            return ["/dev/stdin"]
        else:
            return ["/dev/stdout"]
    elif filename == "&0":
        return ["/dev/stdin"]
    elif filename == "&1":
        return ["/dev/stdout"]
    elif filename == "&2":
        return ["/dev/stderr"]
    else:
        return [filename] 

class NoSuchFileError(RuntimeError):
    pass

class MultipleFilesError(RuntimeError):
    pass

class ArgumentError(RuntimeError):
    def __init__(self, arg, message=None):
        super(ArgumentError, self).__init__(message or "Argument '%s' was invalid." % arg)
        self.argument = arg

class MissingArgumentError(ArgumentError):
    def __init__(self, arg):
        super(MissingArgumentError, self).__init__(arg, "Argument '%s' was missing." % arg)

class DuplicateArgumentError(ArgumentError):
    def __init__(self, arg):
        super(DuplicateArgumentError, self).__init__(arg, "Argument '%s' occurred more than once." % arg)

class FileOpener:
    def __init__(self, fileobj, mode='r'):
        self.closeobj = False
        if isinstance(fileobj, str):
            filenames = resolve_filenames(fileobj, mode=mode)
            if len(filenames) == 0:
                raise NoSuchFileError()
            elif len(filenames) > 1:
                raise MultipleFilesError()
            self.fileobj = open(filenames[0], mode)
            self.closeobj = True
        else:
            self.fileobj = fileobj

    def write(self, *args, **kwargs):
        self.fileobj.write(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.fileobj.read(*args, **kwargs)

    def close(self):
        if self.closeobj:
            self.fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        if self.closeobj:
            self.close()
