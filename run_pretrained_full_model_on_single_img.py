from main import get_args_parser
import argparse
from models import build_model
import torch
import util.misc as utils

from PIL import Image
import requests
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as T

from run_pretrained_on_single_image import detect, plot_results, plot_results

# params:
# --backbone resnet101
def main(args):
    utils.init_distributed_mode(args)

    device = 'cpu'
    model, criterion, postprocessors = build_model(args)

    ckpt = torch.load(r"D:\data\__research_only\detr_models\detr-r101-2c7b67e5.pth", map_location="cpu")
    # ckpt = torch.load(r"D:\data\__research_only\detr_training\demo_training\checkpoint_12.pth", map_location="cpu")
    # state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_pah = r"D:\data\__research_only\coco17\val2017\000000039769.jpg"
    im = Image.open(img_pah)
    w, h = im.size
    im_cropped = im.crop((0, 0, 320, h))
    scores, boxes = detect(im, model, transform)
    scores_cr, boxes_cr = detect(im_cropped, model, transform)

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plot_results(im, scores, boxes, CLASSES, COLORS)
    plot_results(im_cropped, scores_cr, boxes_cr, CLASSES, COLORS)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
