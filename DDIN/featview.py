import cv2
import numpy as np
def torch_vis_color(feature_tensor,col,raw,save_path,colormode=2,margining=1):
    '''
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    :param feature_tensor: torch.Tensor [1,c,w,h]
    :param col: col num
    :param raw: raw num
    :param save_path: save path
    :param colormode: cv2.COLORMAP
    :return:None
    '''
    show_k = col * raw
    f = feature_tensor[0, :show_k, :, :]
    size = f[0, :, :].shape
    f = f.data.numpy()
    for i in range(raw):
        tem = f[i * col, :, :]*255/(np.max(f[i * col, :, :]+1e-14))
        tem = cv2.applyColorMap(np.array(tem,dtype=np.uint8), colormode)
        for j in range(col):
            if not j == 0:
                tem = np.concatenate((tem, np.ones((size[0],margining,3),dtype=np.uint8)*255), 1)
                tem2=cv2.applyColorMap(np.array(f[i * col + j, :, :]*255/(np.max(f[i * col + j, :, :])+1e-14),dtype=np.uint8), colormode)
                tem = np.concatenate((tem,tem2), 1)
        if i == 0:
            final = tem
        else:
            final = np.concatenate((final, np.ones((margining, size[1] * col + (col - 1)*margining,3),dtype=np.uint8)*255), 0)
            final = np.concatenate((final, tem), 0)
    print(final.shape)
    cv2.imwrite(save_path,final)