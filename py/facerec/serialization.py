#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import dill

def save_model(filename, model):
    output = open(filename, 'wb')
    dill.dump(model, output)
    output.close()

def load_model(filename):
    pkl_file = open(filename, 'rb')
    res = dill.load(pkl_file)
    pkl_file.close()
    return res
