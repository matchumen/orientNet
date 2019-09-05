import cv2
from torchvision.transforms import functional as F
import torch


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, target):
        #image, target = sample[0], sample[1]

        w, h = image.size

        new_h, new_w = self.output_size, self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = image.resize((new_h, new_w))


        target['keypoints'][0] = target['keypoints'][0] * torch.Tensor([new_w / w, new_h / h, 1])

        masks = target['masks']
        # hack
        mask = masks[0].numpy()
        mask = cv2.resize(mask, (self.output_size, self.output_size))
        new_mask = torch.as_tensor(mask, dtype=torch.uint8)

        target['masks'] = new_mask[None, :, :]

        target['boxes'][0] = target['boxes'][0] * torch.Tensor([new_w / w, new_h / h, new_w / w, new_h / h])

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target