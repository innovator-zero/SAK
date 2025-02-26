import tarfile
import os
import numpy as np
from PIL import Image
from torch.utils import data

from .utils.mypath import MyPath
from utils import global_print


class ImageTarDataset(data.Dataset):
    def __init__(self, root=MyPath.db_root_dir("imagenet"), train=True, transform=None):
        """
        return_labels:
        Whether to return labels with the samples
        transform:
        A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        """
        if train:
            tar_file = os.path.join(root, "train.tar")
        else:
            tar_file = os.path.join(root, "val.tar")
        self.tar_file = tar_file
        self.tar_handle = None
        categories_set = set()
        self.tar_members = []
        self.categories = {}
        self.categories_to_examples = {}
        with tarfile.open(tar_file, "r:") as tar:
            for index, tar_member in enumerate(tar.getmembers()):
                if tar_member.name.count("/") != 2:
                    continue
                category = self._get_category_from_filename(tar_member.name)
                categories_set.add(category)
                self.tar_members.append(tar_member)
                cte = self.categories_to_examples.get(category, [])
                cte.append(index)
                self.categories_to_examples[category] = cte
        categories_set = sorted(categories_set)
        for index, category in enumerate(categories_set):
            self.categories[category] = index
        self.num_examples = len(self.tar_members)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        global_print(
            "Loaded the dataset from {}. It contains {} samples.".format(
                tar_file, self.num
            )
        )
        self.transform = transform

    def _get_category_from_filename(self, filename):
        begin = filename.find("/")
        begin += 1
        end = filename.find("/", begin)
        return filename[begin:end]

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        if self.tar_handle is None:
            self.tar_handle = tarfile.open(self.tar_file, "r:")

        sample = self.tar_handle.extractfile(self.tar_members[index])
        image = Image.open(sample).convert("RGB")
        image = self.transform(image)

        return image
