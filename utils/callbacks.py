import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt

import shutil
import numpy as np
from matplotlib import rcParams
from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map
import wandb 
import re
from thop import clever_format, profile
from config import *

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.scene_loss = []
        self.route_loss = []

        os.makedirs(self.log_dir, exist_ok=True)

    def append_loss(self, epoch, loss, scene_loss, route_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.scene_loss.append(scene_loss)
        self.route_loss.append(route_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"{epoch}: {loss}\n")
        with open(os.path.join(self.log_dir, "epoch_scene_loss.txt"), 'a') as f:
            f.write(f"{epoch}: {scene_loss}\n")
        with open(os.path.join(self.log_dir, "epoch_route_loss.txt"), 'a') as f:
            f.write(f"{epoch}: {route_loss}\n")

        self.loss_plot()

    def loss_plot(self):

        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['figure.dpi'] = 150
        rcParams['axes.titleweight'] = 'bold'

        iters = range(len(self.losses))

        def plot_loss(values, title, filename, color, smooth_color):
            plt.figure(figsize=(8, 6))
            plt.plot(iters, values, color=color, linewidth=2.5, label=title)
            try:
                num = 5 if len(values) < 25 else 15
                smooth = scipy.signal.savgol_filter(values, num, 3)
                plt.plot(iters, smooth, color=smooth_color, linestyle='--', linewidth=2, label='Smoothed')
            except:
                pass
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlabel('Epoch', fontsize=12, fontweight='bold')
            plt.ylabel('Loss', fontsize=12, fontweight='bold')
            plt.title(title, fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, filename))
            plt.close()

        plot_loss(self.losses, "Train Loss", "epoch_loss.png", color='royalblue', smooth_color='skyblue')
        plot_loss(self.scene_loss, "Scene Loss", "epoch_scene_loss.png", color='darkorange', smooth_color='navajowhite')
        plot_loss(self.route_loss, "Route Loss", "epoch_route_loss.png", color='purple', smooth_color='orchid')

        

class EvalCallback():
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines1, val_lines2, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.class_names        = class_names
        self.num_classes        = num_classes

        self.val_lines1          = val_lines1
        self.val_lines2          = val_lines2

        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.bbox_util          = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)

            # outputs--> outputs[0]
            outputs = self.bbox_util.decode_box(outputs[0])
            
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    def init_map_dir(self):
        os.makedirs(self.map_out_path, exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            for annotation_line in tqdm(self.val_lines1):
                self.init_map_dir()
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                image       = Image.open(line[0])
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            # wandb.log({"mAP": temp_map})
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_map_val1.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map_val1.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)

            for annotation_line in tqdm(self.val_lines2):
                self.init_map_dir()
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                image       = Image.open(line[0])
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
            self.maps.append(temp_map)
            # wandb.log({"mAP": temp_map})
            self.epoches.append(epoch)
            with open(os.path.join(self.log_dir, "epoch_map_val2.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map_val2.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)