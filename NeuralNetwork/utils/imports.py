#-*- coding:utf-8 -*-

import torch


# torch._six.PY3 表示 python3 
# 如果是 python3 则运行下面程序, 如果是 python2 运行 else 部分的程序
if torch._six.PY3:
    import importlib
    import importlib.util
    import sys

    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module
else:
    import imp

    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
