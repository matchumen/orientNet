import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CubeDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'annotations', 'masks'))))
        self.anns = os.path.join(root, 'annotations', 'ann.csv')

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'annotations', 'masks', self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # doplnit keypoints
        keypoints = []
        with open(self.anns, 'r') as anns:
            line = anns.readline()
            while line:
                if img_path.replace("\\", "/") in line:
                    record = parse_record(line)
                    keypoints = record['keypoints']
                line = anns.readline()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        keypoints = torch.as_tensor(np.array(keypoints)[None, :, :], dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["keypoints"] = keypoints


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def parse_record(raw_record):
    out_dict = {}
    raw_data = raw_record.split(";")

    out_dict["file_path"] = raw_data[0]
    tmp_keypoints = [data.split(",") for data in raw_data[1:9]]

    out_dict["keypoints"] = []
    for keypoint in tmp_keypoints:
        keypoint = [int(elem) for elem in keypoint]
        out_dict["keypoints"].append(keypoint)

    out_dict["position"] = [float(data) for data in raw_data[9:12]]

    out_dict["rotation"] = [float(data) for data in raw_data[12:]]

    return out_dict
