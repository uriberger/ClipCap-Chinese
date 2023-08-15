import json
import sys
import random
from utils import get_flickr8kcn_data, get_flickr30k_data

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
base_train_sample_num = int(sys.argv[2])

flickr8kcn_data = get_flickr8kcn_data()

print(f'Collected {len(flickr8kcn_data)} samples')

flickr8kcn_image_ids = list(set([x['image_id'] for x in flickr8kcn_data]))
base_image_ids = random.sample(flickr8kcn_image_ids, base_train_sample_num)
base_image_ids_dict = {x: True for x in base_image_ids}

flickr30k_data = get_flickr30k_data()
all_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr30k_data]
flickr8kcn_image_ids_dict = {x: True for x in flickr8kcn_image_ids}
non_flickr8kcn_image_ids = [x for x in all_image_ids if x not in flickr8kcn_image_ids_dict]

test_image_num = 1000
test_image_ids = random.sample(non_flickr8kcn_image_ids, test_image_num)
test_image_dict = {x: True for x in test_image_ids}

additional_image_ids = [x for x in all_image_ids if x not in base_image_ids_dict and x not in test_image_dict]
additional_image_ids_dict = {x: True for x in additional_image_ids}

with open(f'reformulation_experiment/data/image_ids/base_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(base_image_ids))
with open(f'reformulation_experiment/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(additional_image_ids))
with open(f'reformulation_experiment/data/image_ids/test_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(test_image_ids))
print(f'Selected {len(base_image_ids)} base train image ids, {len(additional_image_ids)} additional train image ids and {len(test_image_ids)} test image ids')

with open(f'reformulation_experiment/data/base_train_data/base_train_data_{exp_ind}.txt', 'w') as fp:
    for sample in flickr8kcn_data:
        if sample['image_id'] in base_image_ids_dict:
            fp.write(f'{sample["image_id"]}\t{sample["caption"]}\n')
