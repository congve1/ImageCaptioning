import argparse
import os
import json
import logging

import torch
from tqdm import tqdm

from image_captioning.config import cfg
from image_captioning.utils.imports import import_file
from image_captioning.utils.misc import decode_sequence
from image_captioning.utils.checkpoint import ModelCheckpointer
from image_captioning.utils.collect_env import collect_env_info
from image_captioning.utils.logger import setup_logger
from image_captioning.utils.comm import get_rank, synchronize, is_main_process, all_gather
from image_captioning.modeling.encoder.build import build_encoder
from image_captioning.modeling.decoder.build import build_decoder
from image_captioning.data.build import make_data_loader
from image_captioning.data.vocab.get_vocab import get_vocab


def _accumulate_descriptions_from_multi_gpu(descriptions):
    all_descriptions = all_gather(descriptions)
    if not is_main_process():
        return
    results = []
    for descriptions in all_descriptions:
        results.extend(descriptions)
    return descriptions


def generate_online_results(
        encoder, decoder, data_loader, vocab,
        device, beam_size, output_json
):
    logger = logging.getLogger('image_captioning.generate_online_results')
    descriptions = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, ncols=100, ascii=True, desc="generating")):
            img_t, img_ids = data
            img_t = img_t.to(device)
            fc_features, att_features = encoder(img_t)
            seq, probs, weights = decoder.decode_search(
                fc_features, att_features, beam_size
            )
            #logger.info("processed: {}/{} images".format(img_t.size(0), len(data_loader)))
            sentences = decode_sequence(vocab, seq)
            for idx, sentence in enumerate(sentences):
                entry = {'image_id': img_ids[idx],
                         'caption': sentence}
                descriptions.append(entry)
        synchronize()
        descriptions = _accumulate_descriptions_from_multi_gpu(descriptions)
        if is_main_process():
            logger.info("saving results file: {}".format(output_json))
            with open(output_json, 'w') as f:
                json.dump(descriptions, f)
            logger.info("Done")


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

    logger = setup_logger('image_captioning', cfg.OUTPUT_DIR, get_rank(), "generate_online_results_log.txt")
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
    beam_size = cfg.TEST.BEAM_SIZE
    paths_catalog = import_file(
        "image_captioning.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    ResNetCatalog = paths_catalog.ResNetCatalog
    vocab = get_vocab(cfg.DATASET.VOCAB_PATH)
    # build encoder
    encoder = build_encoder(cfg)
    encoder_loader = ModelCheckpointer(cfg, encoder)
    url = ResNetCatalog.get(cfg.MODEL.ENCODER.CONV_BODY)
    encoder_loader.load(url)
    encoder = encoder.to(device)
    # build decoder
    decoder = build_decoder(cfg)
    decoder_loader = ModelCheckpointer(cfg, decoder)
    decoder_loader.load(cfg.MODEL.WEIGHT)
    decoder = decoder.to(device)
    # set to eval mode
    encoder.eval()
    decoder.eval()
    for split, dataset in zip(['val', 'test'], [cfg.DATASET.VAL, cfg.DATASET.TEST]):
        if not dataset:
            continue
        logger.info('load dataset {}'.format(dataset))
        data_loader = make_data_loader(cfg, split=split)
        generate_online_results(
            encoder,
            decoder,
            data_loader,
            vocab,
            device,
            beam_size,
            os.path.join(
                cfg.OUTPUT_DIR, 'captions_' + split + "2014_" + args.alg_name + '_results.json'
            ),
        )


if __name__ == '__main__':
    main()
