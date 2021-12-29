#coding=utf-8
import numpy as np
import os
import cv2
from xml.etree import ElementTree as ET
from PIL import Image
import h5py
import random

'''path that should satisfy your machine'''
file_dir = 'E:/train_Skim/type_video'
VID_ROOT_DIR = 'E:/ILSVRC2015_VID_new/ILSVRC2015'
save = 'E:/same_type_negtive_data.h5'  # 数据集存储文件

type_names = ['airplane','antelope','bear','bicycle','bird',
              'bus','car','cattle','dog','domestic_cat',
              'elephant','fox','giant_panda','hamster','horse',
              'lion','lizard','monkey','motorcycle','rabbit',
              'red_panda','sheep','snake','squirrel','tiger',
              'train','turtle','watercraft','whale','zebra']

N = 1000
aug_N = int(N * 1.04)
result = np.random.randint(0, 30, (2, aug_N)).astype(np.uint8)
mask = (result[0] != result[1])
final_result = result[:, mask][:, :N]


type_video_dict = {}
for type in type_names:
    # print(type)
    type_video_dict[type] = {}
    type_video_dict[type]['path'] = []
    type_video_dict[type]['track_id'] = []
    type_video_dict[type]['num_frames'] = []
    path_file = os.path.join(file_dir, type, 'path.txt')
    track_id_file = os.path.join(file_dir, type, 'track_id.txt')
    num_frames_file = os.path.join(file_dir, type, 'num_frames.txt')

    with open(path_file, 'r') as f:
        path = f.readlines()
    type_video_dict[type]['path'] = path

    type_video_dict[type]['track_ids'] = np.loadtxt(track_id_file, np.uint8)
    type_video_dict[type]['num_frames'] = np.loadtxt(num_frames_file, np.uint32)
    # assert(len(type_video_dict[type]['track_id']) == len(type_video_dict[type]['num_frames']))
    type_video_dict[type]['num_track_ids'] = len(type_video_dict[type]['track_ids'])
    # print(type,type_video_dict[type]['num_track_ids'])

image_root_dir = os.path.join(VID_ROOT_DIR, 'Data/VID/train')
anno_root_dir = os.path.join(VID_ROOT_DIR, 'Annotations/VID/train')


def get_T(type,object_idx,frame_idx,track_id):  # 得到模板块
    '''instruction'''
    frame_path = os.path.join(image_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                              '%06d.JPEG' % frame_idx)
    frame = cv2.imread(frame_path)
    anno_path = os.path.join(anno_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                             '%06d.xml' % frame_idx)
    tree = ET.parse(anno_path)
    root = tree.getroot()
    objects = root.findall('object')
    if objects != None:
        EXIST = False
        for o in objects:
            tmp = o.find('trackid').text
            if tmp == str(track_id):
                EXIST = True
                # object = objects[track_id] # wrong
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                # print(ymin,ymax,xmin,xmax)
                H, W, _ = frame.shape
                xmin = np.clip(xmin, 0, W)
                xmax = np.clip(xmax, 0, W)
                ymin = np.clip(ymin, 0, H)
                ymax = np.clip(ymax, 0, H)

                w = xmax - xmin
                h = ymax - ymin

                half_w = int(w / 2 * 1.3)
                half_h = int(h / 2 * 1.3)
                cw = int((xmin + xmax)/2)
                ch = int((ymin + ymax)/2)

                top, bottom, left, right = (0, 0, 0, 0)
                if cw < half_w: left = half_w - cw
                if ch < half_h: top = half_h - ch
                if (cw + half_w) > W: right = half_w + cw - W
                if (ch + half_h) > H: bottom = half_h + ch - H

                cw += left
                ch += top

                new_im = cv2.copyMakeBorder(  # BGR [123.68, 116.779, 103.939]
                    frame, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=[123, 117, 104])

                new_im = new_im[
                         ch - half_h:ch + half_h,
                         cw - half_w:cw + half_w, :]
                frame = cv2.resize(new_im, (140, 140))  # template
                frame = frame.astype('float32')

                return frame

        return None
    else:
        return None

def get_S(type, object_idx, frame_idx, track_id, object_idx_neg, frame_idx_neg, track_id_neg):  # 得到搜索块
    '''instruction'''
    frame_path = os.path.join(image_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                              '%06d.JPEG' % frame_idx)
    frame = cv2.imread(frame_path)

    frame_path_neg = os.path.join(image_root_dir, type_video_dict[type]['path'][object_idx_neg][:-1],
                             '%06d.JPEG' % frame_idx_neg)

    frame_neg = cv2.imread(frame_path_neg)

    anno_path = os.path.join(anno_root_dir, type_video_dict[type]['path'][object_idx][:-1],
                             '%06d.xml' % frame_idx)
    anno_path_neg = os.path.join(anno_root_dir, type_video_dict[type]['path'][object_idx_neg][:-1],
                             '%06d.xml' % frame_idx_neg)
    if np.random.rand() > 0.5:  # 抽取正样本的搜索块
        tree = ET.parse(anno_path)
        root = tree.getroot()
        objects = root.findall('object')
        if objects != None:
            EXIST = False
            for o in objects:
                tmp = o.find('trackid').text
                if tmp == str(track_id):
                    EXIST = True
                    # object = objects[track_id] # wrong
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    # print(ymin,ymax,xmin,xmax)
                    H, W, _ = frame.shape
                    xmin = np.clip(xmin, 0, W-1)
                    xmax = np.clip(xmax, 0, W-1)
                    ymin = np.clip(ymin, 0, H-1)
                    ymax = np.clip(ymax, 0, H-1)

                    w = xmax - xmin
                    h = ymax - ymin

                    if h >= 30 and w >= 30 and h * 1.0 / H < 0.7 and w * 1.0 / W < 0.7:
                        label = 1
                        half = int((w + h) / 2 * 2.4)

                        cw = xmin + w / 2 + np.random.randint(-half * 0.5, half * 0.5)
                        ch = ymin + h / 2 + np.random.randint(-half * 0.5, half * 0.5)

                        crop = np.array([cw - half, ch - half, cw + half, ch + half], dtype=int)

                        img_pos = Image.fromarray(frame)
                        img_pos = img_pos.crop(crop)
                        img_pos = img_pos.resize([256, 256])
                        img_pos = np.array(img_pos).astype('float32')
                        return img_pos, label
            return None, None
        else:
            return None, None

    else:  # 0.5的概况
        tree = ET.parse(anno_path_neg)  # 抽取负样本的搜索块
        root = tree.getroot()
        objects = root.findall('object')
        if objects != None:
            EXIST = False
            for o in objects:
                tmp = o.find('trackid').text
                if tmp == str(track_id_neg):  # 找到负目标（比如同样是狗，正样本是哈士奇，负样本是中华田园犬）
                    # object = objects[track_id] # wrong
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    # print(ymin,ymax,xmin,xmax)
                    H, W, _ = frame.shape
                    xmin = np.clip(xmin, 0, W - 1)
                    xmax = np.clip(xmax, 0, W - 1)
                    ymin = np.clip(ymin, 0, H - 1)
                    ymax = np.clip(ymax, 0, H - 1)

                    w = xmax - xmin
                    h = ymax - ymin
                    if h >= 30 and w >= 30 and h * 1.0 / H < 0.7 and w * 1.0 / W < 0.7:
                        label = 0
                        half = int((w + h) / 2 * 2.4)

                        cw = xmin + w / 2 + np.random.randint(-half * 0.5, half * 0.5)
                        ch = ymin + h / 2 + np.random.randint(-half * 0.5, half * 0.5)

                        crop = np.array([cw - half, ch - half, cw + half, ch + half], dtype=int)

                        img = Image.fromarray(frame_neg)
                        img = img.crop(crop)
                        img = img.resize([256, 256])
                        img = np.array(img).astype('float32')
                        return img, label
                else:  # 找不到负目标，则选择随机图  # 以0.3的概率
                    label = 0
                    x = np.random.randint(0, 10)
                    frame_neg = frame_neg[x: x + 256, x: x + 256]
                    frame_neg = cv2.resize(frame_neg, [256, 256])
                    img = frame_neg.astype('float32')
                    return img, label
        return None, None


def get_two_samples(type):  # 得到模板-搜索块
    '''instruction'''
    '''pick up 1 object from a specific type'''
    object_idx = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]
    track_id = type_video_dict[type]['track_ids'][object_idx]
    track_id_num_frames = type_video_dict[type]['num_frames'][object_idx]  # 得到正样本的相关信息

    object_idx_neg = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]
    while object_idx_neg == object_idx:
        object_idx_neg = np.random.randint(0, type_video_dict[type]['num_track_ids'], 1)[0]  # 确保正负样本不会是同一个东西

    track_id_neg = type_video_dict[type]['track_ids'][object_idx_neg]
    track_id_num_frames_neg = type_video_dict[type]['num_frames'][object_idx_neg]
    chosen_frame_neg = np.random.randint(0, track_id_num_frames_neg, 1)  # 得到负样本的相关信息
    '''pick up 2 frame from above video'''

    chosen_frame_idx1, chosen_frame_idx2 = np.random.randint(0, track_id_num_frames, 2)

    roi1 = get_T(type, object_idx, chosen_frame_idx1, track_id)

    roi2, label = get_S(type, object_idx, chosen_frame_idx2, track_id, object_idx_neg, chosen_frame_neg, track_id_neg)

    return (roi1, roi2, label)


num = 0
summ = 6000  # 数据总数


fdata = h5py.File(save, 'w')
fdata.create_dataset('template',(summ,140, 140, 3), dtype='float32')
fdata.create_dataset('search',(summ, 256, 256, 3), dtype='float32')
fdata.create_dataset('label',(summ, 1), dtype='float32')

while num < summ:
    type = type_names[final_result[0, num % 500]]
    x = random.randint(0, 24)

    while type_names[x] == type:
        x = random.randint(0, 24)

    type2 = type_names[x]
    template, search, label = get_two_samples(type)
    if search is not None and template is not None:
        fdata['search'][num] = search
        fdata['template'][num] = template
        fdata['label'][num] = label
        num += 1
        print(num, label, type)
fdata.close()
