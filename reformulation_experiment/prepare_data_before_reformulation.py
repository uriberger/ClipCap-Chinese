import json
import sys

assert len(sys.argv) == 2
exp_ind = sys.argv[1]

with open(f'reformulation_experiment/data/base_train_data/additional_train_data_{exp_ind}.json', 'r') as fp:
    additional_train_data = json.load(fp)
with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_translated.json', 'r') as fp:
    translated_data = json.load(fp)

# Split to Flickr and COCO
flickr_translated_data = []
coco_translated_data = []
for sample in translated_data:
    image_id = sample['image_id']
    if additional_train_data[image_id]['dataset'] == 'flickr30k':
        flickr_translated_data.append({'image_id': additional_train_data[image_id]['image_id'], 'caption': sample['caption']})
    elif additional_train_data[image_id]['dataset'] == 'COCO':
        coco_translated_data.append({'image_id': additional_train_data[image_id]['image_id'], 'caption': sample['caption']})
    else:
        assert False

with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_translated_flickr.json', 'w') as fp:
    fp.write(json.dumps(flickr_translated_data))
with open(f'reformulation_experiment/data/infer/base_infer_on_additional_train_{exp_ind}_en_translated_coco.json', 'w') as fp:
    fp.write(json.dumps(coco_translated_data))
