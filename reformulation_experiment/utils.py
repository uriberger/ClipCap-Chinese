from collections import defaultdict
import json

def get_flickr_captions():
    with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
        flickr_data = json.load(fp)['images']
    image_ids_dict = {int(x['filename'].split('.jpg')[0]): True for x in flickr_data}
    
    with open('datasets/flickr_caption.txt', 'r') as f:
        lines = f.readlines()
    image_id_to_captions = defaultdict(list)

    for i in range(len(lines)):
        line = lines[i].strip()
        image_id, caption = line.split('\t')
        image_id = int(image_id)
        if image_id in image_ids_dict:
            image_id_to_captions[image_id].append(caption)

    return image_id_to_captions
