import glob
import os
import os.path
import re
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
    verify_str_arg,
)


class Market1501(VisionDataset):
    name = "Market1501"
    mirrors = [
        "http://188.138.127.15:81/Datasets/",
    ]
    resources = [
        ("Market-1501-v15.09.15.zip", "226086b4b519c148eb2f3030729c27e257e3f5ea"),
    ]

    def __init__(
        self,
        root: str,
        mode: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        market1501_500k=False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.mode = mode
        self.market1501_500k = market1501_500k

        if download:
            self.download()

        self.image_paths, self.targets = self._load_data()
        self.classes = set(self.targets)

    def _load_data(self):
        data = []
        if self.mode == "train":
            data = _process_dir(self.train_folder, relabel=True)
        elif self.mode == "query":
            data = _process_dir(self.query_folder, relabel=False)
        elif self.mode == "gallery":
            data = _process_dir(self.gallery_folder, relabel=False)
            if self.market1501_500k:
                data += _process_dir(self.extra_gallery_folder, relabel=False)
        targets = [i[1] for i in data]  # (pid, cid)
        image_paths = [i[0] for i in data]

        return image_paths, torch.Tensor(targets).long()

    def __getitem__(self, index):
        img_path, target = self.image_paths[index], self.targets[index]
        img = _read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        # return 2000
        return len(self.targets)

    @property
    def download_folder(self) -> str:
        return os.path.join(self.root, self.name, "download")

    @property
    def extract_folder(self) -> str:
        return os.path.join(self.root, self.name, "extract")

    @property
    def train_folder(self):
        return os.path.join(self.extract_folder, "bounding_box_train")

    @property
    def query_folder(self):
        return os.path.join(self.extract_folder, "query")

    @property
    def gallery_folder(self):
        return os.path.join(self.extract_folder, "bounding_box_test")

    @property
    def extra_gallery_folder(self):
        return os.path.join(self.extract_folder, "images")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.download_folder, os.path.basename(url)))
            for url, _ in self.resources
        )

    def download(self) -> None:
        if self._check_exists():
            print("Files already downloaded and verified")
            return

        os.makedirs(self.download_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url,
                        download_root=self.download_folder,
                        extract_root=self.extract_folder,
                        filename=filename,
                        md5=md5,
                    )
                except URLError as error:
                    print("Failed to download (trying next):\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))


def _read_image(path):
    got_img = False
    if not os.path.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert("RGB")
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                    path
                )
            )
    return img


def _process_dir(dir_path, relabel=False) -> List[Tuple[str, int, int]]:
    img_paths = glob.glob(os.path.join(dir_path, "*.jpg"))
    pattern = re.compile(r"([-\d]+)_c(\d)")
    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    data = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))  # (IMG_SRC, Person ID, Cam ID)
    return data


class PairedMarket1501(Market1501):
    def __init__(
        self, *args: any, **kwargs: any,
    ):
        super().__init__(*args, **kwargs)

        self.targets_set = set(self.targets.numpy())
        self.target_to_indices = {
            target: np.where(self.targets.numpy() == target)[0]
            for target in self.targets_set
        }

    def __getitem__(self, index_a):
        img_a, target_a = super().__getitem__(index_a)
        target_a = int(target_a)

        index_p = index_a
        while index_p == index_a:
            index_p = np.random.choice(self.target_to_indices[target_a])
        img_p, target_p = super().__getitem__(index_p)

        target_n = np.random.choice(list(self.targets_set - set([target_a])))
        index_n = np.random.choice(self.target_to_indices[target_n])
        img_n, target_n = super().__getitem__(index_n)

        return (img_a, img_p, img_n), (target_a, target_p, target_n)
