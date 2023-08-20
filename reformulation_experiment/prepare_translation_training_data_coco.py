from collections import defaultdict
import json
import sys
import random

assert len(sys.argv) == 2
exp_ind = int(sys.argv[1])

with open('reformulation_experiment/data/translated_data/flickr30k_zh_translated_helsinki.json', 'r') as fp:
    flickr_translated_data = json.load(fp)
with open('reformulation_experiment/data/translated_data/coco_zh_translated_helsinki.json', 'r') as fp:
    coco_translated_data = json.load(fp)
with open(f'reformulation_experiment/data/base_train_data/additional_train_data_{exp_ind}.json', 'r') as fp:
    additional_train_data = json.load(fp)
with open(f'reformulation_experiment/data/image_ids/flickr_orig_to_new_image_id_{exp_ind}.json', 'r') as fp:
    flickr_image_ids_dict = json.load(fp)
with open(f'reformulation_experiment/data/image_ids/coco_orig_to_new_image_id_{exp_ind}.json', 'w') as fp:
    coco_image_ids_dict = json.load(fp)

image_id_to_path = {i: additional_train_data[i]['file_path'] for i in range(len(additional_train_data))}

image_id_to_captions = defaultdict(list)
for x in flickr_translated_data:
    if x['image_id'] in flickr_image_ids_dict:
        image_id_to_captions[flickr_image_ids_dict[x['image_id']]].append(x['caption'])
for x in coco_translated_data:
    if x['image_id'] in coco_image_ids_dict:
        image_id_to_captions[coco_image_ids_dict[x['image_id']]].append(x['caption'])

train_data = []
for image_id, captions in image_id_to_data.items():
    train_data.append({'image_id': image_id, 'image_path': image_id_to_path[image_id], 'caption': random.choice(captions)})

with open(f'reformulation_experiment/data/translated_train_data/translated_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(train_data))
