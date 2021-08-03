from yolox.utils.boxes import postprocess
from yolox.data.data_augment import preproc
import torch
import torch.nn as nn
import numpy as np
from BaseDetector import baseDet
import os
from yolox.utils import fuse_model
from yolox.data.datasets import COCO_CLASSES


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        # check availability
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

    return torch.device('cuda:0' if cuda else 'cpu')


class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        
        self.build_config()
        self.mdepth = 0.33
        self.mwidth = 0.50
        self.confthre=0.01
        self.nmsthre=0.65
        self.test_size=(640, 640)
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.init_model()

    def init_model(self):

        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.mdepth, self.mwidth, in_channels=in_channels)
            head = YOLOXHead(80, self.mwidth, in_channels=in_channels)
            model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        self.weights = 'weights/yolox_s.pth'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        ckpt = torch.load(self.weights)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        model = fuse_model(model)
        self.m = model

        self.names = COCO_CLASSES
        self.num_classes = len(self.names)

    def preprocess(self, img):
        
        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()

        return img_info, img

    def detect(self, im):

        img_info, img = self.preprocess(im)

        outputs = self.m(img)
        outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )[0]
        pred_boxes = []
        ratio = img_info["ratio"]
        img = img_info["raw_img"]

        boxes = outputs[:, 0:4]

        # preprocessing: resize
        boxes /= ratio

        cls_ids = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]

        for i in range(len(boxes)):
            box = boxes[i].cpu()
            lbl = self.names[int(cls_ids[i])]
            conf = scores[i]
            if conf < self.confthre:
                continue
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            pred_boxes.append(
                            (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes


if __name__ == '__main__':
    
    det = Detector()
    