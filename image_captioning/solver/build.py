import torch

from image_captioning.solver import registry

from .lr_schedulers import WarmupMultiStepLR, SGDRCosineLR


@registry.OPTIMIZERS.register("Adam")
def make_adam(cfg, params):
    optimizer = torch.optim.Adam(
        params, cfg.SOLVER.BASE_LR,
        betas=cfg.SOLVER.BETAS
    )
    return optimizer


@registry.OPTIMIZERS.register("SGD")
def make_sgd(cfg, params):
    optimizer = torch.optim.SGD(
        params, cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM
    )
    return optimizer


@registry.SCHEDULERS.register('WarmupMultiStepLR')
def make_warmupmultisteplr(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )


@registry.SCHEDULERS.register('StepLR')
def make_step_lr(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg.SOLVER.STEP_SIZE,
        cfg.SOLVER.GAMMA
    )


@registry.SCHEDULERS.register("SGDR")
def make_sgdr(cfg, optimizer):
    return SGDRCosineLR(
        optimizer,
        cfg.SOLVER.T_MAX,
        cfg.SOLVER.T_MULTI,
        cfg.SOLVER.ETA_MIN,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )


def make_optimizer(cfg, model):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if 'bias' in name:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [param], "lr": lr, "weight_decay": weight_decay}]
    assert cfg.SOLVER.OPTIMIZER in registry.OPTIMIZERS, \
        "cfg.SOLVER.OPTIMIZER:{} are not registered in registry".format(
            cfg.SOLVER.OPTIMIZER
        )
    optimizer = registry.OPTIMIZERS[cfg.SOLVER.OPTIMIZER](cfg, params)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    assert cfg.SOLVER.SCHEDULER in registry.SCHEDULERS, \
        "cfg.SOLVER.SCHEDULER:{} are not registered in registry".format(
            cfg.SOLVER.SCHEDULER
        )
    return registry.SCHEDULERS[cfg.SOLVER.SCHEDULER](cfg, optimizer)
