import json
import sys
import pickle

assert len(sys.argv) == 2
exp_ind = sys.argv[1]

with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_reformulated_flickr.json', 'r') as fp:
    flickr_reformulated_data = json.load(fp)
with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_reformulated_coco.json', 'r') as fp:
    coco_reformulated_data = json.load(fp)
with open(f'reformulation_experiment/data/image_ids/flickr_orig_to_new_image_id_{exp_ind}.pkl', 'rb') as fp:
    flickr_orig_to_new_image_id = pickle.load(fp)
with open(f'reformulation_experiment/data/image_ids/coco_orig_to_new_image_id_{exp_ind}.pkl', 'rb') as fp:
    coco_orig_to_new_image_id = pickle.load(fp)

reformulated_data = []
for sample in flickr_reformulated_data:
    reformulated_data.append({'image_id': flickr_orig_to_new_image_id[sample['image_id']], 'caption': sample['caption']})
for sample in coco_reformulated_data:
    reformulated_data.append({'image_id': coco_orig_to_new_image_id[sample['image_id']], 'caption': sample['caption']})
with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_reformulated.json', 'w') as fp:
    fp.write(json.dumps(reformulated_data))
