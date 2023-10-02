import json
import sys
sys.path.append('.')
from utils import get_coco_data
from config import coco_image_path

assert len(sys.argv) == 4
input_file = sys.argv[1]
output_file = sys.argv[2]
exp_ind = int(sys.argv[3])

with open(input_file, 'r') as fp:
    data = json.load(fp)

coco_data = get_coco_data()
iid_to_split = {}
for x in coco_data:
    if x['split'] == 'train':
        iid_to_split[x['cocoid']] = 'train'
    else:
        iid_to_split[x['cocoid']] = 'val'

train_data = []
for sample in data:
    image_id = sample['image_id']
    split = iid_to_split[image_id]
    train_data.append({'image_id': image_id, 'image_path': f'{coco_image_path}/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'caption': sample['caption']})
with open(output_file, 'w') as fp:
    fp.write(json.dumps(train_data))
