import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange
import os.path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_train_img(gt_filepath, lr_filepath, bic_filepath, nFrames, scale):
    tt = int(nFrames / 2)  ##
    GT = modcrop(Image.open(gt_filepath).convert('RGB'), scale)  # PIL
    input = modcrop(Image.open(lr_filepath).convert('RGB'), scale)
    Bic = modcrop(Image.open(bic_filepath).convert('RGB'), scale)
    char_len = len(lr_filepath)  #
    neibor = []

    seq = [x for x in range(-tt, tt+1) if x != 0]  # seq = range(-3,4)=[-3,-2,-1,1, 2, 3]
    for i in seq:
        index1 = int(lr_filepath[char_len-7:char_len-4]) + i  # last 4-7 char,4 is .png，if name is 03d，then 7=3+4，if is 08d，replace 7 by 12=8+4
        file_name1 = lr_filepath[0:char_len-7]+'{0:03d}'.format(index1) + '.png'  # {0:0xd}
        if os.path.exists(file_name1):
            temp = modcrop(Image.open(file_name1).convert('RGB'), scale)
            neibor.append(temp)
        else:
            print('Neigbor frame does not exist！')
            temp = input  # !
            neibor.append(temp)

    return GT, input, neibor, Bic

def load_eval_img(eval_filepath, lr_filepath, bic_filepath, nFrames, scale):
    tt = int(nFrames / 2)
    GT = modcrop(Image.open(gt_filepath).convert('RGB'), scale)
    input = modcrop(Image.open(lr_filepath).convert('RGB'), scale)
    Bic = modcrop(Image.open(bic_filepath).convert('RGB'), scale)
    char_len = len(lr_filepath)
    neibor = []

    seq = [x for x in range(-tt, tt+1) if x != 0]  # seq = range(-3,4)=[-3,-2,-1,1, 2, 3]
    for i in seq:
        index1 = int(lr_filepath[char_len-7:char_len-4]) + i
        file_name1 = lr_filepath[0:char_len-7]+'{0:03d}'.format(index1) + '.png'

        if os.path.exists(file_name1):
            temp = modcrop(Image.open(file_name1).convert('RGB'), scale)
            neibor.append(temp)
        else:
            print('Neigbor frame does not exist！')
            temp = input
            neibor.append(temp)

    return GT, input, neibor, Bic


def load_test_img(lr_filepath, nFrames, scale):
    tt = int(nFrames / 2)
    LR = modcrop(Image.open(lr_filepath).convert('RGB'), scale)
    char_len = len(lr_filepath)
    neibor = []

    seq = [x for x in range(-tt, tt+1) if x != 0]  # seq = range(-3,4)=[-3,-2,-1,1, 2, 3]
    for i in seq:
        index1 = int(lr_filepath[char_len-12:char_len-4]) + i
        file_name1 = lr_filepath[0:char_len-12]+'{0:08d}'.format(index1) + '.png'
        if os.path.exists(file_name1):
            temp = modcrop(Image.open(file_name1).convert('RGB'), scale)
            neibor.append(temp)
        else:
            print('Neigbor frame does not exist！')
            temp = LR
            neibor.append(temp)

    return LR, neibor

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)  # tt = 3
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]  # -3, -2, -1, 1, 2, 3

        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i  # 6 + i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_gt, img_nn, img_bic, scale, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size  #
    (th, tw) = (ih * scale, iw * scale)

    path_mult = scale
    tp = path_mult * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (ix * scale, iy * scale)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:,iy:iy+ip,ix:ix+ip]
    img_gt = img_gt.crop((ty, tx, ty + tp, tx + tp))  # [:,ty:ty+tp,tx:tx+tp]
    img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  #

    info_patch = {'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_gt, img_nn, img_bic, info_patch


def augment(img_in, img_gt, img_nn, img_bic, flip_h = True, rot = True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:  # mirror
        img_in = ImageOps.flip(img_in)
        img_gt = ImageOps.flip(img_gt)
        img_bic = ImageOps.flip(img_bic)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:  # flip
            img_in = ImageOps.mirror(img_in)
            img_gt = ImageOps.mirror(img_gt)
            img_bic = ImageOps.mirror(img_bic)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True

        if random.random() < 0.5:  # 180°
            img_in = img_in.rotate(180)
            img_gt = img_gt.rotate(180)
            img_bic = img_bic.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_gt, img_nn, img_bic, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class LoadTrainingDataFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, path_size, transform=None):
        super(LoadTrainingDataFromFolder, self).__init__()
        GT_folder_name = 'training_set/GT/'
        LR_folder_name = 'training_set/LR4x/'
        Bic_folder_name = 'training_set/Bicubic4x/'
        # ['GT/000/001.png',···,'GT/000/100.png']
        gt_alist = [GT_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        lr_alist = [LR_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        bic_alist = [Bic_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        # ['E:\\Github_package/video_dataloader/train\\GT/000/001.png',···]
        self.gt_image_filenames = [join(image_dir, x) for x in gt_alist]
        self.lr_image_filenames = [join(image_dir, x) for x in lr_alist]
        self.bic_image_filenames = [join(image_dir, x) for x in bic_alist]

        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.patch_size = path_size
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        gt, input, neigbor, bic = load_train_img(self.gt_image_filenames[index], self.lr_image_filenames[index], self.bic_image_filenames[index], self.nFrames, self.upscale_factor)
        if self.patch_size != 0:
            input, gt, neigbor, bic, _ = get_patch(input, gt, neigbor, bic, self.upscale_factor, self.patch_size)

        if self.data_augmentation != 0:
            input, gt, neigbor, bic, _ = augment(input, gt, neigbor, bic)

        if self.transform:
            gt = self.transform(gt)
            input = self.transform(input)
            bic = self.transform(bic)
            neigbor = [self.transform(j) for j in neigbor]


        return gt, input, neigbor, bic

    def __len__(self):
        return len(self.gt_image_filenames)

class LoadEvalDataFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, transform=None):
        super(LoadEvalDataFromFolder, self).__init__()
        GT_folder_name = 'test_set/GT/'
        LR_folder_name = 'test_set/LR4x/'
        Bic_folder_name = 'test_set/Bicubic4x/'
        # ['GT/000/001.png',···,'GT/000/100.png']
        gt_alist = [GT_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        lr_alist = [LR_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        bic_alist = [Bic_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        # ['E:\\Github_package/video_dataloader/train\\GT/000/001.png',···]
        self.gt_image_filenames = [join(image_dir, x) for x in gt_alist]
        self.lr_image_filenames = [join(image_dir, x) for x in lr_alist]
        self.bic_image_filenames = [join(image_dir, x) for x in bic_alist]

        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        gt, input, neigbor, bic = load_train_img(self.gt_image_filenames[index], self.lr_image_filenames[index], self.bic_image_filenames[index], self.nFrames, self.upscale_factor)
        if self.transform:
            gt = self.transform(gt)
            input = self.transform(input)
            bic = self.transform(bic)
            neigbor = [self.transform(j) for j in neigbor]


        return gt, input, neigbor, bic

    def __len__(self):
        return len(self.gt_image_filenames)


class LoadTestDataFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, transform=None):
        super(LoadTestDataFromFolder, self).__init__()
        LR_folder_name = ''
        lr_alist = [LR_folder_name + line.rstrip() for line in open(join(image_dir, file_list))]
        self.lr_image_filenames = [join(image_dir, x) for x in lr_alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform

        print(self.lr_image_filenames)

    def __getitem__(self, index):
        lr, neigbor = load_test_img(self.lr_image_filenames[index], self.nFrames, self.upscale_factor)
        flow = [get_flow(lr, j) for j in neigbor]


        if self.transform:
            lr = self.transform(lr)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]


        return lr, neigbor, flow

    def __len__(self):
        return len(self.lr_image_filenames)