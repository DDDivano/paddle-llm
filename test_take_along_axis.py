#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from utils.tools import generate_array
from utils.compare import Compare
import  numpy as np
import paddle
import torch

np.random.seed(33)
input_data = generate_array(shape=[1, 4096, 1001], dtype=np.float32, value_range=(-1, 1))
indices = generate_array(shape=[1, 4096, 1], dtype=np.int64, value_range=(1, 1000))


def paddle_dynamic(dtype=np.float32):
    if dtype == np.float32:
        input = input_data.astype(np.float32)
    elif dtype == np.float16:
        input = input_data.astype(np.float16)
    else:
        input = input_data
    x = paddle.to_tensor(input)
    x.stop_gradient = False
    result = paddle.take_along_axis(x, paddle.to_tensor(indices), axis=-1)
    grad = paddle.grad(result, x)
    return result.numpy(), grad[0].numpy()

def torch_dynamic(dtype=np.float32):
    if dtype == torch.float32:
        input = input_data.astype(np.float32)
    elif dtype == torch.float16:
        input = input_data.astype(np.float16)
    else:
        input = input_data
    x = torch.tensor(input)
    x.requires_grad = True
    result = torch.take_along_dim(x, torch.tensor(indices), dim=-1)
    result.retain_grad()
    result_sum = result.sum()
    result_sum.backward()
    grad = x.grad
    return result.detach().numpy(), grad.detach().numpy()


def paddle_static(dtype=np.float32):
    if dtype == np.float32:
        input = input_data.astype(np.float32)
    elif dtype == np.float16:
        input = input_data.astype(np.float16)
    else:
        input = input_data
    x = paddle.to_tensor(input)
    x.stop_gradient = False
    result = paddle.jit.to_static(paddle.take_along_axis)(x, paddle.to_tensor(indices), axis=-1)
    grad = paddle.grad(result, x)
    return result.numpy(), grad[0].numpy()



def test_paddle_dynamic_vs_torch_fp32():
    """
    paddle dynamic vs torch fp32
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float32)
    torch_res, torch_grad = torch_dynamic(np.float32)
    Compare(paddle_res, torch_res, rtol=1e-6, atol=1e-6)
    Compare(paddle_grad, torch_grad, rtol=1e-6, atol=1e-6)

def test_paddle_static_vs_torch_fp32():
    """
    paddle static vs torch fp32
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float32)
    torch_res, torch_grad = torch_dynamic(np.float32)
    Compare(paddle_res, torch_res, rtol=1e-6, atol=1e-6)
    Compare(paddle_grad, torch_grad, rtol=1e-6, atol=1e-6)

def test_paddle_dynamic_stability_fp32():
    """
    paddle dynamic stability fp32
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float32)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_dynamic(np.float32)
        Compare(paddle_res, paddle_stability_res, rtol=1e-6, atol=1e-6)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-6, atol=1e-6)


def test_paddle_static_stability_fp32():
    """
    paddle staic stability fp32
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float32)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_dynamic(np.float32)
        Compare(paddle_res, paddle_stability_res, rtol=1e-6, atol=1e-6)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-6, atol=1e-6)


def test_paddle_dynamic_vs_torch_fp16():
    """
    paddle dynamic vs torch fp16
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float16)
    torch_res, torch_grad = torch_dynamic(np.float16)
    Compare(paddle_res, torch_res, rtol=1e-3, atol=1e-3)
    Compare(paddle_grad, torch_grad, rtol=1e-3, atol=1e-3)

def test_paddle_static_vs_torch_fp16():
    """
    paddle static vs torch fp16
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float16)
    torch_res, torch_grad = torch_dynamic(np.float16)
    Compare(paddle_res, torch_res, rtol=1e-3, atol=1e-3)
    Compare(paddle_grad, torch_grad, rtol=1e-3, atol=1e-3)

def test_paddle_dynamic_stability_fp16():
    """
    paddle dynamic stability fp16
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float16)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_dynamic(np.float16)
        Compare(paddle_res, paddle_stability_res, rtol=1e-3, atol=1e-3)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-3, atol=1e-3)

def test_paddle_static_stability_fp16():
    """
    paddle staic stability fp16
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float16)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_dynamic(np.float16)
        Compare(paddle_res, paddle_stability_res, rtol=1e-3, atol=1e-3)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-3, atol=1e-3)


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
    # paddle_static()
    pass