import argparse
import os
import json
import logging

import torch
from tqdm import tqdm

from image_captioning.config import cfg
from image_captioning.utils.imports import import_file
from image_captioning.utils.misc import decode_sequence
from image_captioning.utils.misc import encode_caption
from image_captioning.utils.misc import mkdir
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.comm import get_rank, synchronize, is_main_process, all_gather
from image_captioning.modeling.encoder.build import build_encoder
from image_captioning.data.build import make_data_loader
from image_captioning.data.vocab.get_vocab import get_vocab


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
        vocab
):
    logger = logging.getLogger("image_captioning.generate_dataset")
    root = dataset['args']['root']
    att_features_folder = os.path.abspath(os.path.join(root, repr(data_loader.dataset) + "_att_features"))
    fc_features_folder = os.path.abspath(os.path.join(root, repr(data_loader.dataset) + '_fc_features'))
    if not os.path.exists(att_features_folder):
        mkdir(att_features_folder)
    if not os.path.exists(fc_features_folder):
        mkdir(fc_features_folder)
    att_features_paths_file = dataset['args']['att_features_paths_file']
    fc_features_paths_file = dataset['args']['fc_features_paths_file']
    encoded_captions_file = dataset['args']['encoded_captions_file']
    encoded_captions_lens_file = dataset['args']['encoded_captions_lens_file']
    cocoids_file = dataset['args']['cocoids_file']
    enc_captions = []
    coco_ids = []
    cap_lens = []
    att_features_paths = []
    fc_features_paths = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, ncols=100, ascii=True, desc="processing")):
            img_t, img_captions, temp_ids, temp_paths = data
            fc_features, att_features = encoder(img_t.to(device))
            for j in range(fc_features.size(0)):
                fc_feature_path = os.path.abspath(
                    os.path.join(fc_features_folder, str(temp_ids[j])+"_fc_feature.pt")
                )
                att_feature_path = os.path.abspath(
                    os.path.join(att_features_folder, str(temp_ids[j])+"_att_feature.pt")
                )
                att_feature = att_features[j].clone()
                fc_feature = fc_features[j].clone()
                torch.save(att_feature, att_feature_path)
                torch.save(fc_feature, fc_feature_path)
                att_features_paths.append(att_feature_path)
                fc_features_paths.append(fc_feature_path)
                for cap in img_captions[j]:
                    c_len = min(len(cap), seq_max_len)  # length without <start> <end> token
                    tmp_cap = ['<start>'] + cap + ['<end>'] + \
                      ['<pad>' for _ in range(seq_max_len-c_len)]
                    enc_captions.append(encode_caption(vocab, tmp_cap))
                    cap_lens.append(c_len)
            coco_ids.extend(temp_ids)
        synchronize()
        att_features_paths = _accumulate_data_on_multi_gpu(att_features_paths)
        synchronize()
        fc_features_paths = _accumulate_data_on_multi_gpu(fc_features_paths)
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
            logger.info("writing att feature paths to file {}"
                        .format(fc_features_paths_file))
            with open(att_features_paths_file, 'w') as f:
                json.dump(att_features_paths, f)

            logger.info("writing fc feature paths to file {}"
                        .format(att_features_paths_file))
            with open(fc_features_paths_file, 'w') as f:
                json.dump(fc_features_paths, f)
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
            data_loader, device, seq_max_len, vocab
        )


if __name__ == '__main__':
    main()
