import argparse
import os
import logging
import json
import datetime

import torch

from image_captioning.config import cfg
from image_captioning.data.build import make_data_loader
from image_captioning.solver.build import make_optimizer
from image_captioning.solver.build import make_lr_scheduler
from image_captioning.engine.trainer import do_train
from image_captioning.engine.inference import inference
from image_captioning.modeling.decoder.build import build_decoder
from image_captioning.layers.cross_entropy_loss import LanguageModelCriterion
from image_captioning.data.vocab.get_vocab import get_vocab
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.misc import mkdir
from image_captioning.utils.rewards import init_scorer
from image_captioning.utils.imports import import_file
from image_captioning.utils.comm import get_rank, synchronize, is_main_process


def train(cfg, local_rank, distributed):
    vocab = get_vocab(cfg.DATASET.VOCAB_PATH)
    device = cfg.MODEL.DEVICE
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    log_period = cfg.SOLVER.LOG_PERIOD
    val_period = cfg.SOLVER.VAL_PERIOD
    beam_size = cfg.MODEL.DECODER.BEAM_SIZE
    scst_iter = cfg.SOLVER.SCST_AFTER
    seq_per_img = cfg.DATASET.SEQ_PER_IMG
    grad_clip = cfg.SOLVER.GRAD_CLIP
    metric_logger_name = cfg.SOLVER.METRIC_LOGGER_NAME
    paths_catalog = import_file(
            'image_captioning.config.paths_catalog', cfg.PATHS_CATALOG, True
        )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset = DatasetCatalog.get(cfg.DATASET.TRAIN)
    if scst_iter != -1:
        cached_tokens = os.path.join(dataset['args']['root'], cfg.DATASET.TRAIN+"_words.pkl")
        init_scorer(cached_tokens)
    model = build_decoder(cfg)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False
        )
    arguments = dict()
    arguments['iteration'] = 0
    arguments['best_cider_score'] = -10000
    arguments['device'] = device
    arguments['checkpoint_period'] = checkpoint_period
    arguments['log_period'] = log_period
    arguments['val_period'] = val_period
    arguments['beam_size'] = beam_size
    arguments['scst_iter'] = scst_iter
    arguments['seq_per_img'] = seq_per_img
    arguments['grad_clip'] = grad_clip
    arguments['metric_logger_name'] = metric_logger_name

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = ModelCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    train_data_loader = make_data_loader(
        cfg,
        start_iter=arguments['iteration'],
        is_distributed=distributed
    )
    val_data_loader = make_data_loader(
        cfg,
        split='val',
        is_distributed=distributed
    )

    do_train(
        model,
        train_data_loader,
        val_data_loader,
        optimizer,
        scheduler,
        checkpointer,
        vocab,
        arguments,
    )
    return model


def test(cfg, model, verbose=False, distributed=False):
    if distributed:
        model = model.module
    logger = logging.getLogger("image_captioning.test")
    device = torch.device(cfg.MODEL.DEVICE)
    vocab_path = cfg.DATASET.VOCAB_PATH
    vocab = get_vocab(vocab_path)
    criterion = LanguageModelCriterion()
    test_data_loader = make_data_loader(
        cfg,
        split='test',
        is_distributed=distributed
    )
    beam_size = cfg.TEST.BEAM_SIZE
    test_loss, predictions, scores = inference(
        model,
        criterion,
        test_data_loader,
        vocab,
        beam_size,
        device
    )
    if is_main_process():
        if verbose:
            for prediction in predictions:
                image_id = prediction['image_id']
                caption = prediction['caption']
                logger.info(
                    "image_id:{}\nsent:{}".format(
                        image_id, caption
                    )
                )
        now = datetime.datetime.now()
        file_name = os.path.join(cfg.OUTPUT_DIR, "test(train)-" + now.strftime("%Y%m%d-%H%M%S") + ".json")
        json.dump(predictions, open(file_name, 'w'))
        logger.info("save results to {}".format(file_name))
        for metric, score in scores.items():
            logger.info(
                "metirc {}, score: {:.4f}".format(
                    metric, score
                )
            )


def main():
    parser = argparse.ArgumentParser(description='Pytorch image captioning trainging')
    parser.add_argument(
        '--config-file',
        default='',
        metavar="FILE",
        help='path to config file',
        type=str,
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help="Do not test the final model",
        action='store_true',
    )
    parser.add_argument(
        '--verbose',
        help="show test results",
        action="store_true"
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger('image_captioning', output_dir, get_rank(), "training_log.txt")
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    model = train(cfg, args.local_rank, args.distributed)
    if not args.skip_test:
        test(cfg, model, args.verbose, args.distributed)


if __name__ == '__main__':
    main()
