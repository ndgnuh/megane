import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR
from megane.registry import Registry

lr_schedulers = Registry()


def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations, min_pct):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (
            total_iterations - warmup_iterations
        )
        multiplier = multiplier * (1 - min_pct)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier


def _dbnet_schedule(step, total_steps, power, warmup=0):
    if step < warmup:
        return step / warmup
    else:
        return (1 - (step - warmup) / (total_steps - warmup)) ** power


@lr_schedulers.register("dbnet")
def DBNetScheduler(optimizer, total_steps, power: float = 0.9, warmup=15):
    lambda_lr = partial(
        _dbnet_schedule,
        total_steps=total_steps,
        power=power,
        warmup=warmup,
    )
    return LambdaLR(optimizer, lambda_lr)


@lr_schedulers.register("cosine")
def ConsineDecayWithWarmup(optimizer, total_steps, min_pct=0.05, num_warmup_steps=30):
    schedule = partial(
        _cosine_decay_warmup,
        warmup_iterations=num_warmup_steps,
        total_iterations=total_steps,
        min_pct=min_pct,
    )
    return LambdaLR(optimizer, schedule)
