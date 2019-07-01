import json
import logging
import argparse
import os
from tqdm import tqdm
import shutil
import math

import numpy as np
import torch
import lmdb

from image_captioning.utils.imports import import_file
from image_captioning.config import cfg
from image_captioning.data.vocab.get_vocab import get_vocab
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.modeling.encoder.build import build_encoder
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.misc import encode_caption
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.comm import get_rank, synchronize, is_main_process, all_gather
from image_captioning.data.build import make_data_loader


def write_cache(env, cocoids, features):
    out_of_memory = False
    for idx, cocoid in enumerate(cocoids):
        try:
            with env.begin(write=True) as txn:
                txn.put("{:8d}".format(cocoid).encode(), features[idx].tobytes())
        except lmdb.MapFullError:
            out_of_memory = True
            break
    if out_of_memory:
        new_map_size = env.info()['map_size'] * 2
        if is_main_process():
            env.set_mapsize(new_map_size)
        write_cache(env, cocoids, features)


def _accumulate_data_on_multi_gpu(data):
    all_data = all_gather(data)
    if not is_main_process():
        return
    results = []
    for data in all_data:
        results.extend(data)
    return results


def generate_dataset(
        dataset,
        encoder,
        data_loader,
        device,
        seq_max_len,
        vocab,
):
    page_size = 4096
    logger = logging.getLogger("image_captioning.generate_dataset")
    root = dataset['args']['root']
    att_features_lmdb_path = dataset['args']['att_features_lmdb']
    if os.path.exists(att_features_lmdb_path):
        shutil.rmtree(att_features_lmdb_path)
    fc_features_lmdb_path = dataset['args']['fc_features_lmdb']
    if os.path.exists(fc_features_lmdb_path):
        shutil.rmtree(fc_features_lmdb_path)
    att_features_lmdb = None
    fc_features_lmdb = None
    encoded_captions_file = dataset['args']['encoded_captions_file']
    encoded_captions_lens_file = dataset['args']['encoded_captions_lens_file']
    cocoids_file = dataset['args']['cocoids_file']
    enc_captions = []
    coco_ids = []
    cap_lens = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, ncols=100, ascii=True, desc="processing")):
            img_t, img_captions, temp_ids, temp_paths = data
            fc_features, att_features = encoder(img_t.to(device))
            if not att_features_lmdb:
                att_tmp_np = att_features[0].detach().cpu().numpy()
                nbytes = att_tmp_np.nbytes
                num_pages = math.ceil(nbytes / page_size)
                map_size = (num_pages * page_size + 8 + 8) * (len(data_loader) + 3) + 16
                att_features_lmdb = lmdb.open(att_features_lmdb_path,
                                              map_size=map_size)
            if not fc_features_lmdb:
                fc_tmp_np = fc_features[0].detach().cpu().numpy()
                nbytes = fc_tmp_np.nbytes
                num_pages = math.ceil(nbytes/page_size)
                map_size = (num_pages*page_size + 8 + 8) * (len(data_loader) + 3) + 16
                fc_features_lmdb = lmdb.open(fc_features_lmdb_path,
                                             map_size=map_size)
            att_numpy = att_features.detach().cpu().numpy()
            fc_numpy = fc_features.detach().cpu().numpy()
            write_cache(att_features_lmdb, temp_ids, att_numpy)
            synchronize()
            write_cache(fc_features_lmdb, temp_ids, fc_numpy)
            synchronize()
            for j in range(fc_features.size(0)):
                for cap in img_captions[j]:
                    c_len = min(len(cap), seq_max_len)  # length without <start> <end> token
                    tmp_cap = ['<start>'] + cap + ['<end>'] + \
                        ['<pad>' for _ in range(seq_max_len-c_len)]
                    enc_captions.append(encode_caption(vocab, tmp_cap))
                    cap_lens.append(c_len)
            coco_ids.extend(temp_ids)
        synchronize()
        cap_lens = _accumulate_data_on_multi_gpu(cap_lens)
        synchronize()
        enc_captions = _accumulate_data_on_multi_gpu(enc_captions)
        synchronize()
        coco_ids = _accumulate_data_on_multi_gpu(coco_ids)
        synchronize()
        if is_main_process():
            enc_captions = torch.tensor(enc_captions, dtype=torch.long)
            cap_lens = torch.tensor(cap_lens, dtype=torch.long)
            logger.info("writing coocids to fie {}"
                        .format(cocoids_file))
            with open(cocoids_file, 'w') as f:
                json.dump(coco_ids, f)
            logger.info("writing encoded captions to file {}"
                        .format(encoded_captions_file))
            # Save encoded captions and their lengths to JSON files
            torch.save(enc_captions, encoded_captions_file)
            logger.info("writing captions lengths to file {}"
                        .format(encoded_captions_lens_file))
            torch.save(cap_lens, encoded_captions_lens_file)


def main():
    parser = argparse.ArgumentParser(description='Pytorch image captioning validating')
    parser.add_argument(
        '--config-file',
        default='',
        metavar="FILE",
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--alg-name',
        default='clw',
        help='your algorithm name',
        type=str
    )
    parser.add_argument(
        '--verbose',
        help='show val results',
        action='store_true'
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

    logger = setup_logger('image_captioning', cfg.OUTPUT_DIR, get_rank(), "generate_normal_dataset_log.txt")
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = '\n' + cf.read()
            logger.info(config_str)
    device = cfg.MODEL.DEVICE
    paths_catalog = import_file(
        "image_captioning.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    ResNetCatalog = paths_catalog.ResNetCatalog
    seq_max_len = cfg.DATASET.SEQ_MAX_LEN
    att_size = cfg.MODEL.ENCODER.ATT_SIZE
    att_dim = cfg.MODEL.ENCODER.FEATURE_SIZE
    # build encoder
    encoder = build_encoder(cfg)
    encoder_loader = ModelCheckpointer(cfg, encoder)
    url = ResNetCatalog.get(cfg.MODEL.ENCODER.CONV_BODY)
    encoder_loader.load(url)
    encoder = encoder.to(device)
    # build decoder
    # set to eval mode
    encoder.eval()
    vocab = get_vocab(cfg.DATASET.VOCAB_PATH)
    for split, dataset_name in zip(['train', 'val', 'test'],
                                   [cfg.DATASET.TRAIN, cfg.DATASET.VAL, cfg.DATASET.TEST]):
        if not dataset_name:
            continue
        logger.info('load dataset {}'.format(dataset_name))
        data_loader = make_data_loader(cfg, split=split,
                                       is_distributed=args.distributed,
                                       is_create=True)
        dataset = DatasetCatalog.get(dataset_name)
        generate_dataset(
            dataset, encoder,
            data_loader, device,
            seq_max_len, vocab,
        )


if __name__ == '__main__':
    main()
