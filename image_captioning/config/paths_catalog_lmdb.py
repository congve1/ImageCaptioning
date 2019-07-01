"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = "G:/PyProjects/image_captioning_universal/datasets"
    DATASETS = {
        "coco_2014": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_lmdb',
            'encoded_captions_file': 'coco_2014_captions.pt',
            'encoded_captions_lens_file': 'co/coco_2014_captions_lens.pt',
            "cocoids_file": "coco_2014_cocoids.json",
        },
        "coco_2014_train": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_train_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_train_lmdb',
            'encoded_captions_file': 'coco_2014_captions_train.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_train.pt',
            "cocoids_file": "coco_2014_cocoids_train.json",
        },
        "coco_2014_val": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_val_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_val_lmdb',
            'encoded_captions_file': 'coco_2014_captions_val.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_val.pt',
            "cocoids_file": "coco_2014_cocoids_val.json",
        },
        "coco_2014_test": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_test_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_test_lmdb',
            'encoded_captions_file': 'coco_2014_captions_test.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_test.pt',
            "cocoids_file": "coco_2014_cocoids_test.json",
        },
        "coco_2014_train_simple": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_train_simple_lmdb",
            'fc_features_lmdb' : 'coco_2014_fc_features_train_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_train_simpe.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_train_simple.pt',
            "cocoids_file": "coco_2014_cocoids_train_simple.json",
        },
        "coco_2014_val_simple": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_val_simple_lmdb",
            'fc_features_lmdb' : 'coco_2014_fc_features_val_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_val_simple.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_val_simple.pt',
            "cocoids_file": "coco_2014_cocoids_val_simple.json",
        },
        "coco_2014_test_simple": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_test_simple_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_test_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_test_simple.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_test_simple.pt',
            "cocoids_file": "coco_2014_cocoids_test_simple.json",
        },
        "coco_2014_train_simple_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_train_simple_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_train_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_train_simpe.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_train_simple.pt',
            "cocoids_file": "coco_2014_cocoids_train_simple.json",
        },
        "coco_2014_val_simple_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_val_simple_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_val_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_val_simple.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_val_simple.pt',
            "cocoids_file": "coco_2014_cocoids_val_simple.json",
        },
        "coco_2014_test_simple_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco_simple.json",
            'att_features_lmdb': "coco_2014_att_features_test_simple_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_test_simple_lmdb',
            'encoded_captions_file': 'coco_2014_captions_test_simple.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_test_simple.pt',
            "cocoids_file": "coco_2014_cocoids_test_simple.json",
        },
        "coco_2014_test_online": {
            "img_dir": "coco/test2014",
            "ann_file": "coco/annotations/image_info_test2014.json"
        },
        "coco_2014_val_online": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/captions_val2014.json"
        },
        "coco_2014_train_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_train_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_train_lmdb',
            'encoded_captions_file': 'coco_2014_captions_train.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_train.pt',
            "cocoids_file": "coco_2014_cocoids_train.json",
        },
        "coco_2014_val_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_val_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_val_lmdb',
            'encoded_captions_file': 'coco_2014_captions_val.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_val.pt',
            "cocoids_file": "coco_2014_cocoids_val.json",
        },
        "coco_2014_test_create": {
            "img_dir": "coco",
            "ann_file": "annotations/dataset_coco.json",
            'att_features_lmdb': "coco_2014_att_features_test_lmdb",
            'fc_features_lmdb': 'coco_2014_fc_features_test_lmdb',
            'encoded_captions_file': 'coco_2014_captions_test.pt',
            'encoded_captions_lens_file': 'coco_2014_captions_lens_test.pt',
            "cocoids_file": "coco_2014_cocoids_test.json",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        img_dir =attrs['img_dir']
        if 'online' in name:
            args = dict(
                root=os.path.abspath(os.path.join(data_dir, attrs['img_dir'])),
                ann_file=os.path.abspath(os.path.join(data_dir, attrs['ann_file'])),
            )
            return dict(
                factory="COCODatasetOnline",
                collator="BatchCollatorOnline",
                vocab_file=os.path.abspath(os.path.join(data_dir, 'coco/vocab.pkl')),
                args=args
            )
        if 'create' in name:
            args = dict(
                root=os.path.abspath(os.path.join(data_dir, attrs['img_dir'])),
                ann_file=os.path.abspath(os.path.join(data_dir, img_dir, attrs['ann_file'])),
                att_features_lmdb=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['att_features_lmdb'])
                ),
                fc_features_lmdb=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['fc_features_lmdb'])
                ),
                encoded_captions_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['encoded_captions_file'])
                ),
                encoded_captions_lens_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['encoded_captions_lens_file'])
                ),
                cocoids_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['cocoids_file'])
                ),
            )
            return dict(
                factory="COCODatasetCreate",
                collator="BatchCollatorCreate",
                vocab_file=os.path.abspath(os.path.join(data_dir, 'coco/vocab.pkl')),
                args=args
            )
        if 'coco' in name:

            args = dict(
                root=os.path.abspath(os.path.join(data_dir, attrs['img_dir'])),
                ann_file=os.path.abspath(os.path.join(data_dir, img_dir, attrs['ann_file'])),
                att_features_lmdb=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['att_features_lmdb'])
                ),
                fc_features_lmdb=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['fc_features_lmdb'])
                ),
                encoded_captions_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['encoded_captions_file'])
                ),
                encoded_captions_lens_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['encoded_captions_lens_file'])
                ),
                cocoids_file=os.path.abspath(
                    os.path.join(data_dir, img_dir, attrs['cocoids_file'])
                ),
            )
            return dict(
                factory="COCODatasetLMDB",
                collator="BatchCollator",
                vocab_file=os.path.abspath(os.path.join(data_dir, 'coco/vocab.pkl')),
                args=args
            )


class ResNetCatalog(object):
    @staticmethod
    def get(name):
        model_urls = {
            'R-50-C4': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'R-50-C5': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'R-101-C5': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'R-101-C4': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'R-152-C5': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'R-154-C4': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'R-50-FPN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth',
            'R-101-FPN': 'https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_101_FPN_1x.pth'
        }
        return model_urls[name]
