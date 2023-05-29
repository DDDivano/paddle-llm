#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import numpy as np

def generate_array(shape, dtype, value_range):
    if isinstance(value_range, tuple) and len(value_range) == 2:
        min_val, max_val = value_range
    else:
        raise ValueError("Invalid value_range. Expected a tuple (min_val, max_val).")

    return np.random.uniform(min_val, max_val, shape).astype(dtype)
