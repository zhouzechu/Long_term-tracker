# coding=utf-8
import cv2
import numpy as np
import onnxruntime
model = onnxruntime.InferenceSession('E:/model.onnx')
def gen_search_patch_Hao(img, last_reliable_w, last_reliable_h):  # 得到搜索的块
    # 2.8 300
    # 2.4 256
    max_wh = max(last_reliable_h, last_reliable_w)
    crop_sz = int(max_wh * 1.5)
    H = int(img.shape[0] / crop_sz) * 256  # 1024  img.shape[0] = 1920

    W = int(img.shape[1] / crop_sz) * 256  # 512  img.shape[1] = 1080

    crop_win = np.array([0, 0, 256, 256], dtype=int)
    if H == 0:
        H = 256
    if W == 0:
        W = 256

    Y, X = np.mgrid[0:H - 128:128, 0:W - 128:128]  # 以128为步长，在H*W的大框内，得到若干小框坐标序列
    Y = Y.reshape(-1)
    X = X.reshape(-1)
    if len(X) > 500:
        step = int(len(X) / 500)
        '''TypeError: slice indices must be integers or None or have an __index__ method'''
        sel_idx = list(range(len(X)))[::step][:500]

        X = X[sel_idx]
        Y = Y[sel_idx]
    else:
        pass

    search = cv2.resize(img, (W, H))

    im_save = np.zeros((len(X), 256, 256, 3), dtype=np.uint8)

    scale_h = img.shape[0] / H  # 图像在img的高上的放缩尺寸
    scale_w = img.shape[1] / W  # 图像在img的宽上的放缩尺寸

    pos_i = np.zeros([len(X), 4])
    for i in range(len(X)):
        temp = search[crop_win[1] + Y[i]:crop_win[3] + Y[i], crop_win[0] + X[i]:crop_win[2] + X[i]] # 宽

        im_save[i] = temp.astype(np.uint8)

        pos_i[i] = np.array([
            (crop_win[0] + X[i] + crop_win[2] + X[i]) / 2.0 * scale_w,  # 小框图的中心x坐标
            (crop_win[1] + Y[i] + crop_win[3] + Y[i]) / 2.0 * scale_h,  # 小框图的中心y坐标
            256 * scale_w,  # 小框图在原图上的宽
            256 * scale_h  # 小框图在原图上的高  实际上，如果要与siamFC结合，此处应该为最近一个追踪正确的框大小
        ])

    return im_save, pos_i.astype(int)

def crop_template_Hao(img, box, times=1.3):  # 得到原本模板的块
    im_h, im_w, _ = img.shape

    cw = int(box[0] + box[2] / 2)
    ch = int(box[1] + box[3] / 2)

    half_w = int(box[2] / 2 * times)
    half_h = int(box[3] / 2 * times)

    top, bottom, left, right = (0, 0, 0, 0)
    if cw < half_w: left = half_w - cw
    if ch < half_h: top = half_h - ch
    if (cw + half_w) > im_w: right = half_w + cw - im_w
    if (ch + half_h) > im_h: bottom = half_h + ch - im_h

    cw += left
    ch += top

    new_im = cv2.copyMakeBorder(  # BGR [123.68, 116.779, 103.939]
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[123, 117, 104])

    new_im = new_im[
             ch - half_h:ch + half_h,
             cw - half_w:cw + half_w, :]
    return cv2.resize(new_im, (140, 140))


def glob_redect(cur_ori_img, gt, template):  # 进行全局搜索
    softmax_test, pos_i = gen_search_patch_Hao(cur_ori_img, gt[0], gt[1])
    # softmax_test就是搜索图（原图）切分的若干块序列
    batch_sz = 32

    cls_out = np.empty(softmax_test.shape[0])
    if softmax_test.shape[0] <= batch_sz:
        template = template[None, :, :, :].astype('float32')
        search_img = softmax_test.astype('float32')
        # cls_out = model.predict([template.repeat(search_img.shape[0], axis=0), search_img]).reshape(-1)
        cls_out = model.run([], {'input_6': template.repeat(search_img.shape[0], axis=0), 'input_7': search_img})
        # 使用onnx模型进行推理，其中模板块要按照搜索块数进行扩充repeat
        cls_out = np.array(cls_out).reshape(-1)

    elif softmax_test.shape[0] > batch_sz:  # 因为原模型的batch_size是32，所以切分数大于32时要额外处理！！！！！此处未写完全，因为目前还没有大于32的块数
        cls_out_list = []

        template = template[None, :, :, :].astype('float32')
        search_img = softmax_test.astype('float32')

        for_i = softmax_test.shape[0] // batch_sz
        for jj in range(for_i):
            kk = search_img[batch_sz * jj:batch_sz * (jj + 1)]

            cls_out_batch = model.run([], {'input_6': template.repeat(kk.shape[0], axis=0), 'input_7': kk})
            cls_out_batch = np.array(cls_out_batch).reshape(-1)
            cls_out_list.append(cls_out_batch)

        if softmax_test.shape[0] % batch_sz == 0:
            pass
        else:
            kk = search_img[batch_sz * (jj + 1):]
            cls_out = model.run([], {'input_6': template.repeat(kk.shape[0], axis=0), 'input_7': kk})
            cls_out = np.array(cls_out).reshape(-1)
            cls_out_list.append(cls_out)

        cls_out = np.concatenate(cls_out_list)

    search_rank = np.argsort(-cls_out)
    pos_i = pos_i[search_rank]
    cls_out[search_rank]
    # cv2.imwrite('%d.jpg', softmax_test[search_rank[0]])
    return pos_i[0], search_rank
    """-------------------------------------------------------------------------"""

