#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from utils.tools import generate_array
from utils.compare import Compare
import  numpy as np
import paddle
import torch
from paddle.utils import map_structure
import pytest

np.random.seed(33)
input_data = generate_array(shape=[1, 1], dtype=np.float32, value_range=(-1, 1))
input_data_gradout = generate_array(shape=[1, 1], dtype=np.float32, value_range=(-1, 1))
input_datay = generate_array(shape=[1, 1], dtype=np.float32, value_range=(-1, 1))
# input_datay_gradout = generate_array(shape=[1, 1], dtype=np.float32, value_range=(-1, 1))
condition = generate_array(shape=[1, 1], dtype=np.bool_, value_range=(-1, 1))


def paddle_dynamic(dtype=np.float32, bf16=False):
    if dtype == np.float32:
        input = input_data.astype(np.float32)
        input_gradout = input_data_gradout.astype(np.float32)
        input1 = input_datay.astype(np.float32)
        # input1_gradout = input_datay_gradout.astype(np.float32)
    elif dtype == np.float16:
        input = input_data.astype(np.float16)
        input_gradout = input_data_gradout.astype(np.float16)
        input1 = input_datay.astype(np.float16)
        # input1_gradout = input_datay_gradout.astype(np.float16)
    else:
        input = input_data
        input_gradout = input_data_gradout
        input1 = input_datay
        # input1_gradout = input_datay_gradout
    if bf16:
        x = paddle.to_tensor(input)
        x = paddle.cast(x, dtype="uint16")
        x_gradout = paddle.to_tensor(input_gradout)
        x_gradout = paddle.cast(x_gradout, dtype="uint16")
        y = paddle.to_tensor(input1)
        y = paddle.cast(y, dtype="uint16")
        # y_gradout = paddle.to_tensor(input1_gradout)
        # y_gradout = paddle.cast(y_gradout, dtype="uint16")
    else:
        x = paddle.to_tensor(input)
        x_gradout = paddle.to_tensor(input_gradout)
        y = paddle.to_tensor(input1)
        # y_gradout = paddle.to_tensor(input1_gradout)
    condition_ = paddle.to_tensor(condition)
    x.stop_gradient = False
    y.stop_gradient = False
    result = paddle.where(condition_, x, y)
    grad = paddle.grad(result, x, grad_outputs=x_gradout)
    # y_grad = paddle.grad(result, y, grad_outputs=y_gradout)
    if bf16:
        result = paddle.cast(result, dtype="float32")
        grad = map_structure(lambda x: paddle.cast(x, dtype="float32"), grad)
        # y_grad = map_structure(lambda y: paddle.cast(y, dtype="float32"), y_grad)
    return result.numpy(), grad[0].numpy()
    # return result.numpy(), grad[0].numpy(), y_grad[0].numpy()

def torch_dynamic(dtype=np.float32, bf16=False):
    if dtype == np.float32:
        input = input_data.astype(np.float32)
        input_gradout = input_data_gradout.astype(np.float32)
        input1 = input_datay.astype(np.float32)
        # input1_gradout = input_datay_gradout.astype(np.float32)
    elif dtype == np.float16:
        input = input_data.astype(np.float16)
        input_gradout = input_data_gradout.astype(np.float16)
        input1 = input_datay.astype(np.float16)
        # input1_gradout = input_datay_gradout.astype(np.float16)
    else:
        input = input_data
        input_gradout = input_data_gradout
        input1 = input_datay
        # input1_gradout = input_datay_gradout
    if bf16:
        x = torch.tensor(input)
        x = x.to(dtype=torch.bfloat16)
        x_gradout = torch.tensor(input_gradout)
        x_gradout = x_gradout.to(dtype=torch.bfloat16)
        y = torch.tensor(input1)
        y = y.to(dtype=torch.bfloat16)
        # y_gradout = torch.tensor(input1_gradout)
        # y_gradout = y_gradout.to(dtype=torch.bfloat16)
    else:
        x = torch.tensor(input)
        x_gradout = torch.tensor(input_gradout)
        y = torch.tensor(input1)
        # y_gradout = torch.tensor(input1_gradout)
    condition_ = torch.tensor(condition)
    x.requires_grad = True
    y.requires_grad = True
    result = torch.where(condition_, x, y)
    grad = torch.autograd.grad(result, x, grad_outputs=x_gradout)
    # y_grad = torch.autograd.grad(result, y, grad_outputs=y_gradout)
    if bf16:
        result = result.to(dtype=torch.float32)
        grad = map_structure(lambda x: x.to(dtype=torch.float32), grad)
        # y_grad = map_structure(lambda y: y.to(dtype=torch.float32), y_grad)
    return result.detach().numpy(), grad[0].detach().numpy()
    # return result.detach().numpy(), grad[0].detach().numpy(), y_grad[0].detach().numpy()


def paddle_static(dtype=np.float32, bf16=False):
    if dtype == np.float32:
        input = input_data.astype(np.float32)
        input_gradout = input_data_gradout.astype(np.float32)
        input1 = input_datay.astype(np.float32)
        # input1_gradout = input_datay_gradout.astype(np.float32)
    elif dtype == np.float16:
        input = input_data.astype(np.float16)
        input_gradout = input_data_gradout.astype(np.float16)
        input1 = input_datay.astype(np.float16)
        # input1_gradout = input_datay_gradout.astype(np.float16)
    else:
        input = input_data
        input_gradout = input_data_gradout
        input1 = input_datay
        # input1_gradout = input_datay_gradout
    if bf16:
        x = paddle.to_tensor(input)
        x = paddle.cast(x, dtype="uint16")
        x_gradout = paddle.to_tensor(input_gradout)
        x_gradout = paddle.cast(x_gradout, dtype="uint16")
        y = paddle.to_tensor(input1)
        y = paddle.cast(y, dtype="uint16")
        # y_gradout = paddle.to_tensor(input1_gradout)
        # y_gradout = paddle.cast(y_gradout, dtype="uint16")
    else:
        x = paddle.to_tensor(input)
        x_gradout = paddle.to_tensor(input_gradout)
        y = paddle.to_tensor(input1)
        # y_gradout = paddle.to_tensor(input1_gradout)
    condition_ = paddle.to_tensor(condition)
    x.stop_gradient = False
    y.stop_gradient = False
    result = paddle.jit.to_static(paddle.where)(condition_, x, y)
    grad = paddle.grad(result, x, grad_outputs=x_gradout)
    # y_grad = paddle.grad(result, y, grad_outputs=y_gradout)
    if bf16:
        result = paddle.cast(result, dtype="float32")
        grad = map_structure(lambda x: paddle.cast(x, dtype="float32"), grad)
        # y_grad = map_structure(lambda y: paddle.cast(y, dtype="float32"), y_grad)
    return result.numpy(), grad[0].numpy()
    # return result.numpy(), grad[0].numpy(), y_grad[0].numpy()


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
        paddle_stability_res, paddle_stability_grad = paddle_static(np.float32)
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
        paddle_stability_res, paddle_stability_grad = paddle_static(np.float16)
        Compare(paddle_res, paddle_stability_res, rtol=1e-3, atol=1e-3)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-3, atol=1e-3)

# @pytest.mark.skip(reason="not support bf16")
def test_paddle_dynamic_vs_torch_bf16():
    """
    paddle dynamic vs torch bf16
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float32, True)
    torch_res, torch_grad = torch_dynamic(np.float32, True)
    Compare(paddle_res, torch_res, rtol=1e-2, atol=1e-2)
    Compare(paddle_grad, torch_grad, rtol=1e-2, atol=1e-2)

# @pytest.mark.skip(reason="not support bf16")
def test_paddle_static_vs_torch_bf16():
    """
    paddle static vs torch bf16
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float32, True)
    torch_res, torch_grad = torch_dynamic(np.float32, True)
    Compare(paddle_res, torch_res, rtol=1e-2, atol=1e-2)
    Compare(paddle_grad, torch_grad, rtol=1e-2, atol=1e-2)

# @pytest.mark.skip(reason="not support bf16")
def test_paddle_dynamic_stability_bf16():
    """
    paddle dynamic stability bf16
    :return:
    """
    paddle_res, paddle_grad = paddle_dynamic(np.float32, True)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_dynamic(np.float32, True)
        Compare(paddle_res, paddle_stability_res, rtol=1e-3, atol=1e-3)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-3, atol=1e-3)

# @pytest.mark.skip(reason="not support bf16")
def test_paddle_static_stability_bf16():
    """
    paddle staic stability bf16
    :return:
    """
    paddle_res, paddle_grad = paddle_static(np.float32, True)
    for i in range(5):
        paddle_stability_res, paddle_stability_grad = paddle_static(np.float32, True)
        Compare(paddle_res, paddle_stability_res, rtol=1e-3, atol=1e-3)
        Compare(paddle_grad, paddle_stability_grad, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    # print(paddle_dynamic())
    # print(torch_dynamic())
    # print(paddle_static())
    pass