import json
import os
import sys
import random
sys.path.append('.')
from utils import get_flickr8kcn_data, get_flickr30k_data, get_coco_data
from config import flickr30k_image_path, coco_image_path

assert len(sys.argv) == 2
exp_ind = sys.argv[1]
base_train_sample_num = 7000

flickr8kcn_data = get_flickr8kcn_data()

flickr8kcn_image_ids = list(set([x['image_id'] for x in flickr8kcn_data]))
base_image_ids = random.sample(flickr8kcn_image_ids, base_train_sample_num)
base_image_ids_dict = {x: True for x in base_image_ids}

non_train_image_ids = [x for x in flickr8kcn_image_ids if x not in base_image_ids_dict]
test_image_num = 1000
test_image_ids = random.sample(non_train_image_ids, test_image_num)
test_image_dict = {x: True for x in test_image_ids}

flickr30k_data = get_flickr30k_data()
all_flickr_data = [{'image_id': int(x['filename'].split('.jpg')[0]), 'file_path': os.path.join(flickr30k_image_path, x['filename']), 'dataset': 'flickr30k'} for x in flickr30k_data]
flickr_additional_train_data = [x for x in all_flickr_data if x['image_id'] not in base_image_ids_dict and x['image_id'] not in test_image_dict]

coco_data = get_coco_data()
all_coco_data = [{'image_id': x['cocoid'], 'file_path': os.path.join(coco_image_path, x['filepath'], x['filename']), 'dataset': 'COCO'}]

additional_train_data = flickr_additional_train_data + all_coco_data
# Create new image ids, so that we'll have unique image ids (otherwise we'll have collisions between flickr and coco)
flickr_orig_to_new_image_id = {}
coco_orig_to_new_image_id = {}
for i in range(len(additional_train_data)):
    if sample['dataset'] == 'flickr30k':
        flickr_orig_to_new_image_id[sample['image_id']] = i
    elif sample['dataset'] == 'COCO':
        coco_orig_to_new_image_id[sample['image_id']] = i

print(f'Collected {len(base_image_ids)} base train samples, {len(additional_train_data)} additional train samples, and {len(test_image_ids)} test samples')

with open(f'reformulation_experiment/data/base_train_data/base_train_data_{exp_ind}.txt', 'w') as fp:
    for sample in flickr8kcn_data:
        fp.write(f'{sample["image_id"]}\t{sample["caption"]}\n')

with open(f'reformulation_experiment/data/base_train_data/additional_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(additional_train_data))
with open(f'reformulation_experiment/data/image_ids/flickr_orig_to_new_image_id_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(flickr_orig_to_new_image_id))
with open(f'reformulation_experiment/data/image_ids/coco_orig_to_new_image_id_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(coco_orig_to_new_image_id))
