import os
import h5py
import random
import numpy as np
import multiprocessing
import cv2
from tqdm import tqdm
from functools import partial
from data import preproc as pp


class Dataset:

    def __init__(self, source, name):
        self.source = source
        self.name = name.lower()
        self.partitions = ['train', 'valid', 'test']
        self.dataset = None

    def _init_dataset(self):
        return {p: {"dt": [], "gt": [], "path": []} for p in self.partitions}

    def read_partitions(self):
        data = getattr(self, f"_{self.name}")()
        self.dataset = data

    def save_partitions(self, target, image_input_size, max_text_length):

        os.makedirs(os.path.dirname(target), exist_ok=True)

        total = sum(len(self.dataset[p]['dt']) for p in self.partitions)

        # Create HDF5 with correct shape: (N, 1024, 128, 1)
        with h5py.File(target, "w") as hf:
            for p in self.partitions:
                size = (len(self.dataset[p]['dt']),) + image_input_size
                hf.create_dataset(f"{p}/dt", size, dtype=np.uint8, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{p}/gt", (size[0],), dtype=f"S{max_text_length}", compression="gzip", compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 512

        for p in self.partitions:
            for i in range(0, len(self.dataset[p]['dt']), batch_size):

                imgs = self.dataset[p]['dt'][i:i+batch_size]
                gts  = self.dataset[p]['gt'][i:i+batch_size]

                # Preprocess
                with multiprocessing.Pool() as pool:
                    images = pool.map(partial(pp.preprocess, input_size=image_input_size), imgs)

                images = np.array(images, dtype=np.uint8)

                with h5py.File(target, "a") as hf:
                    hf[f"{p}/dt"][i:i+len(images)] = images
                    hf[f"{p}/gt"][i:i+len(images)] = [x.encode() for x in gts]

                pbar.update(len(images))


    # ====================== IAM DATASET ======================

    def _iam(self):

        part_dir = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")

        paths = {
            "train": open(os.path.join(part_dir, "trainset.txt")).read().splitlines(),
            "valid": open(os.path.join(part_dir, "validationset1.txt")).read().splitlines(),
            "test":  open(os.path.join(part_dir, "testset.txt")).read().splitlines()
        }

        dataset = self._init_dataset()

        for split in self.partitions:
            for p in paths[split]:
                print("LOADING: ",p)
                p = p.strip().replace("\\", "/")

                # Convert relative path into absolute real path
                img_path = os.path.normpath(os.path.join("..", p))

                if not os.path.isfile(img_path):
                    continue

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue


                label = os.path.basename(img_path).replace(".png", "")

                dataset[split]["path"].append(img_path)
                dataset[split]["dt"].append(img)
                dataset[split]["gt"].append(label)

        return dataset
