import PIL
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)

model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(512, 8)

model.load_state_dict(torch.load('model'))

model.eval()

image = PIL.Image.open('C:\\Users\\TCM\\keypoints\\data\\keypoints_data\\images\\0000.png')

image_tensor = torchvision.transforms.functional.to_tensor(image)

# pass a list of (potentially different sized) tensors
# to the model, in 0-1 range. The model will take care of
# batching them together and normalizing
output = model([image_tensor])

print(output)