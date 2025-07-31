# config.py
Cuda = True
seed = 11
distributed = False
sync_bn = False
fp16 = False
programName = 'demo'
dataset_name = 'rtts'
num_classes = 5
classes_path = f'model_data/{dataset_name}_classes.txt'
model_path = 'model_data/yolov7_tiny_weights.pth'
input_shape = [640, 640]
pretrained = False
Init_Epoch = 0
Freeze_Epoch = 35
Freeze_batch_size = 16
UnFreeze_Epoch = 100
Unfreeze_batch_size = 16
Freeze_layers = 70
Freeze_Train = False
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = "cos"
save_period = 5
save_dir = 'logs'
anchors_path = 'model_data/yolo_anchors.txt'
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
eval_flag = True
eval_period = 5
num_workers = 0

train_annotation_path   = f'datasets_split/train.txt'
val_annotation_path1    = f''
val_annotation_path2    = f''



