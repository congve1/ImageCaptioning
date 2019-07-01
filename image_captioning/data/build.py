import logging

import torch.utils.data

from image_captioning.utils.imports import import_file
from image_captioning.utils.comm import get_world_size

from . import datasets as D
from . import samplers
from . import collate_batch as collate_fn


def build_dataset(cfg, dataset_name, dataset_catalog, split, seq_per_img, seq_max_len):
    """
    return a dataset with specified dataset name
    :param seq_max_len:
    :param split:
    :param cfg:
    :param dataset_name:
    :param dataset_catalog:
    :param seq_per_img:
    :return:
    """
    data = dataset_catalog.get(dataset_name)

    factory = getattr(D, data['factory'])

    args = data['args']
    args['seq_per_img'] = seq_per_img
    args['seq_max_len'] = seq_max_len
    args['split'] = split
    args['dataset_name'] = dataset_name
    args['att_feature_shape'] = (1, cfg.MODEL.ENCODER.FEATURE_SIZE,
                                 cfg.MODEL.ENCODER.ATT_SIZE, cfg.MODEL.ENCODER.ATT_SIZE)
    args['fc_feature_shape'] = (1, cfg.MODEL.ENCODER.FEATURE_SIZE)

    dataset = factory(**args)

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(
    sampler, imags_per_batch, num_iters = None, start_iter = 0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, imags_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, split='train', start_iter=0, is_distributed=False, is_create=False):
    num_gpus = get_world_size()
    if split == 'train':
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
        if is_create:
            num_iters = None
            start_iter = 0
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    paths_catalog = import_file(
        'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = {
        'train': cfg.DATASET.TRAIN,
        'val': cfg.DATASET.VAL,
        'test': cfg.DATASET.TEST
    }
    dataset_name = dataset_names[split]
    dataset = build_dataset(cfg, dataset_name, DatasetCatalog, split,
                            cfg.DATASET.SEQ_PER_IMG, cfg.DATASET.SEQ_MAX_LEN)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    num_workers = cfg.DATALOADER.NUM_WORKERS
    collator = DatasetCatalog.get(dataset_name)['collator']
    collator = getattr(collate_fn, collator)()
    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return dataloader
