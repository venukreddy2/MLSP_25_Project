import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np

from config import get_default_config
from common_utils import get_output
from common_utils import get_transformations, get_train_dataset, get_test_dataset, \
    get_test_dataloader, get_train_dataloader, get_optimizer, get_criterion, get_cuda_mem_stats
from model import get_model
from train import Trainer
import os

class DirectResize:
    """Resize samples so that the max dimension is the same as the giving one. The aspect ratio is kept.
    """

    def __init__(self, size):
        self.size = size

        self.mode = {
            'image': cv2.INTER_LINEAR
        }


    def resize(self, key, ori):
        new = cv2.resize(ori, self.size[::-1], interpolation=self.mode[key])

        return new

    def __call__(self, sample):
        for key, val in sample.items():
            if key == 'meta':
                continue
            sample[key] = self.resize(key, val)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from the keys in the state_dict.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
    
def load_model(config, checkpoint_path):
    model = get_model(config)
    
    ckpt = torch.load(checkpoint_path)
    state_dict = remove_module_prefix(ckpt["model"])
    model.load_state_dict(state_dict)
    model = model.cuda()
    return model
    

def get_infer_transforms(config):
    from data import transforms
    import torchvision
    if config.dataset == 'PASCAL_MT':
        dims = (3, 512, 512)
        infer_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.MaxResize(max_dim=512),
            DirectResize(dims[-2:]),
            # transforms.PadImage(size=dims[-2:]),
            transforms.ToTensor(),
        ])
    elif config.dataset == 'Cityscapes3D':
        dims = (3, *config.image_size)
        infer_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            DirectResize(dims[-2:]),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    return infer_transforms

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap

def visualise_task(config, sample, output, save_dir, task, img_name):
    image = sample["image"]
    
    im_height = sample["img_size"][0]
    im_width = sample["img_size"][1]
    
    # output_task = F.interpolate(output[task], (im_height, im_width), mode='bilinear')
    # During visualization, here we always use the train class to draw prediction (totally 19)
    output_task = output[task]
    output_task = get_output(output_task, task).cpu().data.numpy()
    pred = output_task[0]
    
    if task=="semseg":
        if config.dataset=="PASCAL_MT":
            new_cmap = labelcolormap(21)
        else:
            new_cmap = create_cityscapes_label_colormap()
            
        arr = new_cmap[pred]
        
        print('successfully saved for sem seg')
    
    elif task=="human_parts":
        new_cmap = labelcolormap(7)
        arr = new_cmap[pred]
    
    elif task=="depth":
        arr = pred.squeeze()
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        arr_colored = cv2.applyColorMap((arr).astype(np.uint8), cv2.COLORMAP_JET)
        filepath = os.path.join(save_dir, '{}_{}.png'.format(img_name, task))
        cv2.imwrite(filepath, arr_colored)
        print('successfully saved for depth')
        return
    
    arr_uint8 = arr.astype(np.uint8)
    if arr_uint8.ndim == 3:
        arr_uint8 = arr_uint8[:, :, [2, 1, 0]] # Convert RGB to BGR for OpenCV
    # else:
    #     arr_uint8 = cv2.applyColorMap(arr_uint8, cv2.COLORMAP_JET)
    filename = '{}_{}.png'.format(img_name, task)
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, arr_uint8) 

def infer_image(image_path, ckpt_path, save_dir, config):
    
    img_name = image_path.split('/')[-1].split('.')[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if config.dataset == 'PASCAL_MT':
        tasks = ['semseg', 'normals', 'sal', 'edge', 'human_parts']
    elif config.dataset == 'Cityscapes3D': 
        tasks = ['semseg', 'depth']
    
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_size = img.shape[:2] # (h, w, 3)
    sample = {}
    sample["img_size"] = config.image_size
    img = {'image':img}
    infer_transforms = get_infer_transforms(config)
    inp = infer_transforms(img)
    inp = inp["image"]
    inp = inp.unsqueeze(0).cuda() # (1, 3, h, w)
    
    model = load_model(config, ckpt_path)
    print("Loaded model")
    model.eval()
    output = model(inp)
    
    sample["image"] = inp
    for task in tasks:
        visualise_task(config, sample, output, save_dir, task, img_name)


if __name__=="__main__":
    image_path = "/raid/home/rizwank/courses/DLCV/Project/Cityscapes3D/leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png"
    config = get_default_config()
    
    ckpt_path = "/raid/home/rizwank/courses/DLCV/Project/results/checkpoint-encoder-decoder-Cityscapes3D"
    save_dir = f"/raid/home/rizwank/courses/DLCV/Project/results/{config.dataset}"
    infer_image(image_path, ckpt_path, save_dir, config)