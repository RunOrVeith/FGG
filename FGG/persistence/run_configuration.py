import abc
import atexit
import contextlib
import sys
from collections import defaultdict
import datetime
import argparse
from functools import partial
import warnings
import git


class RunSaver(object):
    _registered_hooks = []
    _termination_reason = None

    @staticmethod
    def exit_hook(*args, **kwargs):

        def _call_on_exit(func):
            to_call = partial(func, *args, **kwargs)
            RunSaver._registered_hooks.append(to_call)
            return func

        return _call_on_exit

    @staticmethod
    def call_all_registered_hooks():
        for hook in RunSaver._registered_hooks:
            try:
                hook()
            except Exception as e:
                warnings.warn(f"Could not call {hook.func.__name__} on exit: {e}")

    @staticmethod
    def run_on_exit():
        atexit.register(RunSaver.run_hooks_if_not_already)
        old_sysexcept = sys.excepthook

        def run_before_sys_except(type=None, *args, **kwargs):
            RunSaver.run_hooks_if_not_already()
            return old_sysexcept(type, *args, **kwargs)

        sys.excepthook = run_before_sys_except

    @staticmethod
    def run_hooks_if_not_already(termination_reason=...):
        if RunSaver._termination_reason is not None:
            # There was an exception that already caused serialization
            return
        else:
            RunSaver._termination_reason = termination_reason
            RunSaver.call_all_registered_hooks()


def enable_auto_run_save():
    RunSaver.run_on_exit()


def _make_class_action(key):
    class ClassAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            modules = values.split(".")
            module = sys.modules[".".join(modules[:-1])]
            cls = getattr(module, modules[-1])
            setattr(namespace, key, cls)

    return ClassAction


class AutoConfig(object, metaclass=abc.ABCMeta):

    def __new__(cls, *args, **kwargs):

        old_init = cls.__init__

        def _instantiate_after_init(self, *_args, **_kwargs):
            old_init(self, *_args, **_kwargs)
            if len(sys.argv) > 1:
                self.auto_argparse()

            self._serialize = RunSaver.exit_hook()(self._serialize)

        cls.__init__ = _instantiate_after_init
        return super().__new__(cls, )  # this is just object()

    @abc.abstractmethod
    def __init__(self):
        self._data_groups = defaultdict(set)
        self._create_cmd_arguments_for = defaultdict(lambda: True)

        with self.argument_group("code-state", create_cmd_arguments=False):
            repo = git.Repo(search_parent_directories=True)
            self.git_dirty_files = list(str(file).split("\n")[0] for file in repo.index.diff(None))
            self.git_hexsha = repo.head.object.hexsha

        with self.argument_group("execution", create_cmd_arguments=False):
            self.start_time = datetime.datetime.now()
            self.end_time = None
            self.timestamp = self.start_time.timestamp()

    def auto_argparse(self):

        def _get_additional_arguments(key, default):
            arguments = {}
            if isinstance(default, bool):
                arguments["action"] = f"store_{not default}".lower()
            elif isinstance(default, type):
                arguments["action"] = _make_class_action(key)
            elif isinstance(default, (float, int)):
                arguments["type"] = type(default)
            return arguments

        parser = argparse.ArgumentParser(
            description=f"Automatically generated argument parser for {self.__class__.__name__}"
        )
        for group, keys in self._data_groups.items():
            for key in keys:
                if self._create_cmd_arguments_for[key]:
                    parse_key = key.replace("_", "-")
                    default_value = getattr(self, key)
                    parser.add_argument(f"--{parse_key}", default=default_value,
                                        help=f"""Automatically generated flag for '{key}' in argument group '{group}'.
                                         Defaults to {default_value}.""",
                                        **_get_additional_arguments(key=key, default=default_value))
        # noinspection PyTypeChecker
        parser.parse_args(namespace=self)

    @contextlib.contextmanager
    def argument_group(self, name, create_cmd_arguments=True):
        old_keys = set(vars(self).keys())
        yield
        variables = vars(self)
        new_keys = set(variables.keys())
        added_keys = new_keys.difference(old_keys)
        self._data_groups[name] = self._data_groups[name].union(added_keys)
        if not create_cmd_arguments:
            for key in added_keys:
                self._create_cmd_arguments_for[key] = False

    def _serialize(self):
        self.end_time = datetime.datetime.now()
        data = {group_key: {k: str(v) for k, v in vars(self).items() if k in keys} for group_key, keys in
                self._data_groups.items()}
        self.serialize(data=data)

    @abc.abstractmethod
    def serialize(self, data):
        pass

    def finalize(self):
        pass
