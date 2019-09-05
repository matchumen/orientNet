import cv2
import pprint
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def vis_one(image, target):
	image = np.array(transforms.ToPILImage()(image))

	if 'masks' in target.keys():
		masks = target['masks']
		for mask in masks:
			mask = mask.numpy()
			rgb_mask = np.copy(image)
			rgb_mask[mask == 0] = 0
			rgb_mask[mask == 1] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
			image = cv2.addWeighted(image, 1, rgb_mask, 0.9, 0)
	
	objekts = target['keypoints']
	for objekt in objekts:
		vis_col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
		invis_col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
		for keypoint in objekt:
			color = vis_col if keypoint[2] else invis_col
			cv2.circle(image, (keypoint[0], keypoint[1]), int(image.shape[0]*0.01), color, thickness=-1)

	bboxes = target['boxes']
	for bbox in bboxes:
		cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), int(image.shape[0]*0.005))


	imgplot = plt.imshow(image)
	img_id = np.array(target['image_id']) if 'image_id' in target.keys() else 'unknown'
	plt.suptitle('Image ID: {}'.format(img_id), fontsize=10)
	plt.show()