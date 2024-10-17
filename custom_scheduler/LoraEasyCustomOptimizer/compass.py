# Copied from Lodestone and slightly modified, still should function the same, just added an extra check
# to turn off the stochastic rounding
# repo: https://github.com/lodestone-rock/compass_optimizer/blob/main/experimental/compass_experimental_sr_bf16.py
# Defaults tuned for lora training based on testing

import torch
from torch.optim import Optimizer
from .utils import copy_stochastic_, quantize, dequantize
import math

from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise


class Compass(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 7e-5)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.98, 0.999)).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.001).
        weight_decouple (bool): the optimizer uses decoupled weight decay as in AdamW. (default: true)
        fixed_decay (bool): fix weight decay (default: false).
        clip (float):
            Clip gradient to this value (default: 0.0).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        weight_decay=0.001, #Original default 0
        weight_decouple=True,
        fixed_decay=False,
        clip=0.0,
        amp_fac=2,
        eps=1e-8,
        centralization=0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay = weight_decay,
            weight_decouple = weight_decouple,
            fixed_decay = fixed_decay,
            clip=clip,
            amp_fac=amp_fac,
            eps=eps,
            centralization=centralization,
        )
        super(Compass, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass'
    
    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data)

                p_fp32 = p

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)
                    ema = state["ema"].to(torch.float32)
                    ema_squared = state["ema_squared"].to(torch.float32)
                else:
                    ema, ema_squared = state["ema"], state["ema_squared"]

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                weight_decouple = group["weight_decouple"],
                fixed_decay = group["weight_decouple"]
                centralization = group["centralization"]
                eps = group["eps"]
                clip = group["clip"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                debiased_lr = lr / bias_correction

                # Clip the gradient 
                if clip > 0.0:
                    grad.div_((self.get_rms(grad).add_(eps) / clip).clamp_(min=1.0))

                # Decay the first and second moment running average coefficient
                # ema = ema + (1 - beta1) * grad
                ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)
                # ema_squared = ema + (1 - beta2) * grad ** 2
                ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(eps)

                if weight_decouple:
                    # Perform stepweight decay
                    p_fp32.data.mul_(1.0 - (1.0 if fixed_decay else debiased_lr) * weight_decay)
                elif weight_decay > 0.0 and grad is not None:
                    grad.add_(p_fp32, alpha=weight_decay)

                # p = p - lr * grad / denom
                p_fp32.data.addcdiv_(grad, denom, value=-debiased_lr)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)

        return loss

class Compass8Bit(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        quantization_group_size (int):
            number of quant group (default: 8).
        quantization_factor (float):
            non linear quantization using x^f (default: 3.2)
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        amp_fac=2,
        eps=1e-8,
        weight_decay=0.001, #Original default 0
        centralization=0,
        quantization_group_size=8,
        quantization_factor=3.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            group_size=quantization_group_size,
            factor=quantization_factor,
        )
        super(Compass8Bit, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass8Bit'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass8Bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize(
                        torch.zeros_like(p.data),
                        group_size=group["group_size"],
                        factor=group["factor"],
                    )

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                ema = dequantize(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = dequantize(*state["ema_squared"]) + (1 - beta2) * grad**2
                state["ema"] = quantize(
                    ema, group_size=group["group_size"], factor=group["factor"]
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize(
                    ema_squared, group_size=group["group_size"], factor=group["factor"]
                )

                if weight_decay != 0:
                    # Perform stepweight decay
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32.data.mul_(1 - step_size * weight_decay)
                    else:
                        p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.addcdiv_(grad, denom, value=-step_size)
                else:
                    p.data.addcdiv_(grad, denom, value=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss

class Compass8BitBNB(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        quantization_group_size (int):
            number of quant group (default: 64).
    """

    def __init__(
        self,
        params,
        lr=1e-4, #Original default 1e-3
        betas=(0.975, 0.999), #Original default 0.99, 0.999
        amp_fac=2,
        eps=1e-8,
        weight_decay=0.001, #Original default 0
        centralization=0,
        quantization_group_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            group_size=quantization_group_size,
        )
        super(Compass8BitBNB, self).__init__(params, defaults)

    def __str__(self) -> str:
        return 'Compass8BitBNB'

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass8BitBNB does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["ema"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = quantize_blockwise(
                        torch.zeros_like(p.data),
                        blocksize=group["group_size"],
                    )

                # unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                # Decay the first and second moment running average coefficient
                ema = dequantize_blockwise(*state["ema"]) + (1 - beta1) * grad
                # ema.mul_(beta1).add_(grad, alpha=1 - beta1)
                # grad = grad + ema * amplification_factor
                grad.add_(ema, alpha=amplification_factor)

                ema_squared = (
                    dequantize_blockwise(*state["ema_squared"]) + (1 - beta2) * grad**2
                )
                state["ema"] = quantize_blockwise(
                    ema,
                    blocksize=group["group_size"],
                )

                # ema_squared.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # lr scaler + eps to prevent zero division
                # denom = exp_avg_sq.sqrt() + group['eps']
                denom = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])
                state["ema_squared"] = quantize_blockwise(
                    ema_squared,
                    blocksize=group["group_size"],
                )

                if weight_decay != 0:
                    # Perform stepweight decay
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        p_fp32.data.mul_(1 - step_size * weight_decay)
                    else:
                        p.data.mul_(1 - step_size * weight_decay)

                # p = p - lr * grad / denom
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_fp32.data.addcdiv_(grad, denom, value=-step_size)
                else:
                    p.data.addcdiv_(grad, denom, value=-step_size)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(p, p_fp32)

        return loss