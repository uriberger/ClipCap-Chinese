from collections import defaultdict

def get_flickr_captions():
    with open('datasets/flickr_caption.txt', 'r') as f:
        lines = f.readlines()
    image_id_to_captions = defaultdict(list)

    for i in range(len(lines)):
        line = lines[i].strip()
        image_id, caption = line.split('\t')
        image_id = int(image_id)
        image_id_to_captions[image_id].append(caption)

    return image_id_to_captions
