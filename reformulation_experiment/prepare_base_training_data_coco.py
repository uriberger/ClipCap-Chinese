import json
import os
import sys
sys.path.append('.')
from utils import get_flickr8kcn_data, get_coco_data
from config import flickr30k_image_path, coco_image_path

assert len(sys.argv) == 2
exp_ind = sys.argv[1]

flickr8kcn_data = get_flickr8kcn_data()

with open(f'reformulation_experiment/data/base_train_data/base_train_data_{exp_ind}.txt', 'w') as fp:
    for sample in flickr8kcn_data:
        fp.write(f'{sample["image_id"]}\t{sample["caption"]}\n')

coco_data = get_coco_data()
res = []
for x in coco_data:
    res.append({'image_id': x['cocoid'], 'file_path': f'{coco_image_path}/{x["filepath"]}/{x["filename"]}'})
with open('reformulation_experiment/data/base_train_data/coco_image_ids_and_paths.json', 'w') as fp:
    fp.write(json.dumps(res))
