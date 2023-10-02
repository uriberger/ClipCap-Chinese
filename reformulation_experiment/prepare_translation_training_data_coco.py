from collections import defaultdict
import json
import sys
import random
sys.path.append('.')
from utils import get_coco_data
from config import coco_image_path

assert len(sys.argv) == 2
exp_ind = int(sys.argv[1])

with open('reformulation_experiment/data/translated_data/coco_zh_translated_helsinki.json', 'r') as fp:
    coco_translated_data = json.load(fp)

image_id_to_captions = defaultdict(list)
for x in coco_translated_data:
    image_id_to_captions[x['image_id']].append(x['caption'])

coco_data = get_coco_data()
iid_to_split = {}
for x in coco_data:
    if x['split'] == 'train':
        iid_to_split[x['cocoid']] = 'train'
    else:
        iid_to_split[x['cocoid']] = 'val'

train_data = []
for image_id, captions in image_id_to_captions.items():
    split = iid_to_split[image_id]
    train_data.append({'image_id': image_id, 'image_path': f'{coco_image_path}/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'caption': random.choice(captions)})

with open(f'reformulation_experiment/data/translated_train_data/translated_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(train_data))
