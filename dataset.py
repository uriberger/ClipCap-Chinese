import torch
from torch.utils.data import Dataset
import os
import pickle
import json
from typing import Tuple
from os.path import join
from loguru import logger
import glob
import skimage.io as io
from PIL import Image


class ClipCapDataset(Dataset):
    def __init__(self, clip_data_path, prefix_len, tokenizer, max_len, mode='train', normalize_prefix=False):
        assert mode in ['train', 'test']
        self.normalize_prefix = normalize_prefix
        pad_id = tokenizer.pad_token_id
        if mode == 'train':
            save_path = join(os.path.dirname(clip_data_path), 'train.pkl')
        else:
            save_path = join(os.path.dirname(clip_data_path), 'test.pkl')

        # 加载缓存
        if os.path.isfile(save_path):
            with open(save_path, 'rb') as f:
                self.clip_embeds, self.caption_ids_list, self.mask_list = pickle.load(f)
            logger.info('num of training data'.format(len(self.clip_embeds)))
        else:
            logger.info('loading dataset:{}'.format(clip_data_path))
            with open(clip_data_path, 'rb') as f:
                caption_list, image_id2embed = pickle.load(f)
            logger.info('num of image embedding:{}'.format(len(image_id2embed)))
            logger.info('num of captions:{}'.format(len(caption_list)))

            clip_embeds = []
            caption_ids_list = []
            mask_list = []
            for image_id, caption in caption_list:
                clip_embed = image_id2embed[image_id].squeeze(0).float()
                caption_ids = tokenizer.encode(caption, add_special_tokens=False)
                caption_ids.append(tokenizer.sep_token_id)

                # truncate
                caption_ids = caption_ids[:max_len-prefix_len]
                mask = [1] * (prefix_len + len(caption_ids))

                # padding
                padding_len = max_len - prefix_len - len(caption_ids)
                caption_ids += [pad_id]*padding_len
                mask += [0]*padding_len

                caption_ids = torch.tensor(caption_ids).long()
                mask = torch.tensor(mask).long()

                clip_embeds.append(clip_embed)
                caption_ids_list.append(caption_ids)
                mask_list.append(mask)
            with open(save_path, 'wb') as f:
                pickle.dump([clip_embeds, caption_ids_list, mask_list], f)
            self.clip_embeds = clip_embeds
            self.caption_ids_list = caption_ids_list
            self.mask_list = mask_list
            logger.info('num of training data'.format(len(self.clip_embeds)))

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        clip_embed = self.clip_embeds[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        if self.normalize_prefix:
            clip_embed = clip_embed / clip_embed.norm(2, -1)    # todo check
        return clip_embed, caption_ids, mask


class ImageDataset(Dataset):
    def __init__(self, path, preprocess):
        self.images = []
        with open(path, 'r') as fp:
            data = json.load(fp)
        self.image_ids = []
        for sample in data:
            if type(sample) == int:
                # Only image ids
                image_id = sample
                file_path = f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg'
            elif type(sample) == dict:
                # Image ids and paths
                assert len(sample) == 2
                image_id = sample['image_id']
                file_path = sample['file_path']
            self.image_ids.append(image_id)
            image = io.imread(file_path)
            image = preprocess(Image.fromarray(image)).squeeze(0)
            self.images.append(image)

    def __getitem__(self, item):
        return self.images[item], self.image_ids[item]

    def __len__(self) -> int:
        return len(self.images)
