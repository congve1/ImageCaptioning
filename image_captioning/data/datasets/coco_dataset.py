import pickle
import json
import os
from random import seed, choice, sample

from PIL import Image
import torch
import lmdb
import numpy as np
import logging
import time

from ..transforms.build import build_transforms


class COCODataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        root,
        att_features_paths_file,
        fc_features_paths_file,
        encoded_captions_file,
        encoded_captions_lens_file,
        cocoids_file,
        seq_per_img,
        **kwargs
    ):
        self.root = root
        self.seq_per_img = seq_per_img
        self.dataset_name = kwargs.get('dataset_name', 'Ooops')
        with open(att_features_paths_file, 'r') as f:
            self.att_features_paths = json.load(f)
        with open(fc_features_paths_file, 'r') as f:
            self.fc_features_paths = json.load(f)
        with open(cocoids_file, 'r') as f:
            self.cocoids = json.load(f)
        self.encoded_captions = torch.load(encoded_captions_file,
                                           map_location='cpu')
        self.encoded_captions_lens = torch.load(encoded_captions_lens_file,
                                                map_location='cpu')

    def __getitem__(self, index):

        att_feature = torch.load(
            self.att_features_paths[index//self.seq_per_img],
            map_location='cpu'
        )
        fc_feature = torch.load(
            self.fc_features_paths[index//self.seq_per_img],
            map_location='cpu'
        )
        cap_len = self.encoded_captions_lens[index]
        caption = self.encoded_captions[index]
        all_captions = self.encoded_captions[
            (index//self.seq_per_img)*self.seq_per_img:
            ((index//self.seq_per_img)+1)*self.seq_per_img
        ]
        cocoid = self.cocoids[index//self.seq_per_img]
        data = dict()
        data['att_feature'] = att_feature.unsqueeze(0)
        data['fc_feature'] = fc_feature.unsqueeze(0)
        data['cap_len'] = cap_len
        data['caption'] = caption
        data['all_captions'] = all_captions
        data['cocoid'] = cocoid
        return att_feature.unsqueeze(0), fc_feature.unsqueeze(0), caption, cap_len, all_captions, cocoid

    def __len__(self):
        return len(self.encoded_captions_lens)

    def __repr__(self):
        return self.dataset_name


class COCODatasetLMDB(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        root,
        att_features_lmdb,
        fc_features_lmdb,
        encoded_captions_file,
        encoded_captions_lens_file,
        cocoids_file,
        seq_per_img,
        att_feature_shape,
        fc_feature_shape,
        **kwargs,
    ):
        self.root = root
        self.seq_per_img = seq_per_img
        self.dataset_name = kwargs.get('dataset_name', 'Ooops')
        self.att_feature_shape = att_feature_shape
        self.fc_feature_shape = fc_feature_shape
        with open(cocoids_file, 'r') as f:
            self.cocoids = json.load(f)
        self.encoded_captions = torch.load(
            encoded_captions_file,
            map_location='cpu'
        )
        self.encoded_captions_lens = torch.load(
            encoded_captions_lens_file,
            map_location='cpu'
        )
        self.att_features_lmdb = lmdb.open(
            att_features_lmdb, readonly=True, max_readers=1,
            lock=False, readahead=False, meminit=False
        )
        self.fc_features_lmdb = lmdb.open(
            fc_features_lmdb, readonly=True, max_readers=1,
            lock=False, readahead=False, meminit=False
        )

    def __getitem__(self, index):
        att_features_lmdb = self.att_features_lmdb
        fc_features_lmdb = self.fc_features_lmdb
        cocoid = self.cocoids[index//self.seq_per_img]
        cocoid_enc = "{:8d}".format(cocoid).encode()
        with att_features_lmdb.begin(write=False) as txn:
            att_feature = txn.get(cocoid_enc)
        att_feature = np.frombuffer(att_feature, dtype=np.float32)
        att_feature = att_feature.reshape(self.att_feature_shape)
        att_feature = torch.from_numpy(att_feature)
        with fc_features_lmdb.begin(write=False) as txn:
            fc_feature = txn.get(cocoid_enc)
        fc_feature = np.frombuffer(fc_feature, dtype=np.float32)
        fc_feature = fc_feature.reshape(self.fc_feature_shape)
        fc_feature = torch.from_numpy(fc_feature)

        caption = self.encoded_captions[index]
        caption_len = self.encoded_captions_lens[index]
        all_captions = self.encoded_captions[
            (index//self.seq_per_img)*self.seq_per_img:
            ((index//self.seq_per_img)+1)*self.seq_per_img
        ]
        return att_feature, fc_feature, caption, caption_len, all_captions, cocoid

    def __len__(self):
        return len(self.encoded_captions_lens)

    def __repr__(self):
        return self.dataset_name


class COCODatasetOnline(torch.utils.data.dataset.Dataset):
    def __init__(self, ann_file, root, **kwargs):
        self.transforms = build_transforms()
        with open(ann_file, 'rb') as f:
            annotations = json.load(f, encoding='utf-8')
        self.images = annotations['images']
        self.root = root
        self.dataset_name = kwargs.get('dataset_name', 'ooops')

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images[index]['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_t = self.transforms(img)
        img_id = self.images[index]['id']
        return img_t, img_id

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return self.dataset_name


class COCODatasetCreate(torch.utils.data.dataset.Dataset):
    def __init__(self, ann_file, root,  seq_per_img, seq_max_len, split,**kwargs):
        self.dataset_name = kwargs.get('dataset_name', "Ooops")
        self.transforms = build_transforms()
        self.seq_per_img = seq_per_img
        self.seq_max_len = seq_max_len
        with open(ann_file, 'rb') as f:
            annotations = json.load(f, encoding='utf-8')
        images = annotations['images']
        self.img_paths = []
        self.captions = []
        self.coco_ids = []
        for idx, image in enumerate(images):
            # getting all the captions tokens associate with the image
            img_captions = []
            for sentence in image['sentences']:
                if len(sentence['tokens']) <= seq_max_len:
                    img_captions.append(sentence['tokens'])
                else:
                    img_captions.append(sentence['tokens'][:seq_max_len])
            path = os.path.abspath(os.path.join(root,
                                                image['filepath'],
                                                image['filename']))
            if split == 'train' and image['split'] in ['train', 'restval']:
                self.img_paths.append(path)
                self.captions.append(img_captions)
                self.coco_ids.append(image['cocoid'])
            elif split == image['split']:
                self.img_paths.append(path)
                self.captions.append(img_captions)
                self.coco_ids.append(image['cocoid'])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img_t = self.transforms(img)
        img_captions = self.captions[index]
        if len(img_captions) < self.seq_per_img:
            target_captions = (
                            img_captions
                            + [choice(img_captions) for _ in range(self.seq_per_img - len(img_captions))]
                    )
        else:
            target_captions = sample(
                img_captions, k=self.seq_per_img
            )
        cocoid = self.coco_ids[index]

        return img_t, target_captions, cocoid, img_path

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self):
        return self.dataset_name
