#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from utils.tools import generate_array
import  numpy as np
import paddle
import torch

input_data = generate_array(shape=[2, 3, 3, 3], dtype=np.float32, value_range=(-1, 1))


def paddle_dynamic():
    x = paddle.to_tensor(input_data)
    x.stop_gradient = False
    result = paddle.abs(x)
    grad = paddle.grad(result, x)
    print(result)
    print(grad)
    return result.numpy(), grad[0].numpy()

def torch_dynamic():
    x = torch.from_numpy(input_data)
    x.requires_grad = True
    result = torch.abs(x)
    grad = torch.autograd.grad(result.sum(), x)
    print(result)
    print(grad)
    return result.detach().numpy(), grad[0].detach().numpy()


def paddle_static():
    x = paddle.to_tensor(input_data)
    x.stop_gradient = False
    result = paddle.jit.to_static(paddle.abs)(x)
    grad = paddle.grad(result, x)
    print(result)
    print(grad)
    return result.numpy(), grad[0].numpy()



def test_paddle_dynamic_vs_torch_fp32():
    """
    paddle dynamic vs torch fp32
    :return:
    """
    pass

def test_paddle_static_vs_torch_fp32():
    """
    paddle static vs torch fp32
    :return:
    """
    pass

def test_paddle_dynamic_stability_fp32():
    """
    paddle dynamic stability fp32
    :return:
    """
    pass

def test_paddle_static_stability_fp32():
    """
    paddle staic stability fp32
    :return:
    """
    pass


def test_paddle_dynamic_vs_torch_fp16():
    """
    paddle dynamic vs torch fp16
    :return:
    """
    pass

def test_paddle_static_vs_torch_fp16():
    """
    paddle static vs torch fp16
    :return:
    """
    pass

def test_paddle_dynamic_stability_fp16():
    """
    paddle dynamic stability fp16
    :return:
    """
    pass

def test_paddle_static_stability_fp16():
    """
    paddle staic stability fp16
    :return:
    """
    pass


def test_paddle_dynamic_vs_torch_bf16():
    """
    paddle dynamic vs torch bf16
    :return:
    """
    pass

def test_paddle_static_vs_torch_bf16():
    """
    paddle static vs torch bf16
    :return:
    """
    pass

def test_paddle_dynamic_stability_bf16():
    """
    paddle dynamic stability bf16
    :return:
    """
    pass

def test_paddle_static_stability_bf16():
    """
    paddle staic stability bf16
    :return:
    """
    pass
if __name__ == '__main__':
    # paddle_dynamic()
    # torch_dynamic()
    paddle_static()