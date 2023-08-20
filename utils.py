import json
import os
from config import coco_path, flickr30k_path, flickr8kcn_root

def get_flickr8kcn_data():
    data_dir_name = 'data'
    caption_file_name = 'flickr8kzhc.caption.txt'
    caption_file_path = os.path.join(flickr8kcn_root, data_dir_name, caption_file_name)

    image_id_captions_pairs = []
    with open(caption_file_path, 'r', encoding='utf8') as fp:
        for line in fp:
            striped_line = line.strip()
            if len(striped_line) == 0:
                continue
            line_parts = striped_line.split()
            assert len(line_parts) == 2
            image_id = int(line_parts[0].split('_')[0])
            caption = line_parts[1]
            image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})
            
    return image_id_captions_pairs

def get_flickr30k_data():
    with open(flickr30k_path, 'r') as fp:
        flickr_data = json.load(fp)['images']
    return flickr_data

def get_coco_data():
    with open(coco_path, 'r') as fp:
        coco_data = json.load(fp)['images']
    return coco_data
