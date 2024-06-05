import numpy as np
# 读数据
def read_images(file_name):
    with open(file_name, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        images = []

        for i in range(num_images):
            image = np.fromfile(f, dtype=np.uint8, count=rows*cols) / 255
            images.append([image])

    return np.vstack(images)

def read_labels(file_name):
    with open(file_name, "rb") as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        labels = []

        for i in range(num_labels):
            label = int.from_bytes(f.read(1), 'big')
            labels.append([label])

    return np.vstack(labels)