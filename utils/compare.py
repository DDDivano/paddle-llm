#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import numpy as np


class Compare(object):
    def __init__(self, a, b, atol=0, rtol=0, mode="allclose"):
        self.a = a
        self.b = b
        self.atol = atol
        self.rtol = rtol
        if mode == "equal":
            self.equal()
        elif mode == "allclose":
            self.compare()

    def compare(self):
        if isinstance(self.a, list) and isinstance(self.b, list):
            for i in range(len(self.a)):
                np.testing.assert_allclose(self.a[i], self.b[i], atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(self.a, self.b, atol=self.atol, rtol=self.rtol)
    def equal(self):
        np.testing.assert_equal(self.a, self.b)
