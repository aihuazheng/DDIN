from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import os.path as osp
import numpy as np
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import random
from time import time

"""
def read_img(img_path, color):
    img = Image.open(img_path)
    img = img.resize((144, 288))
    
    
    if color == "red":
      img = ImageOps.expand(img, border=15, fill='red')##left,top,right,bottom
    elif color == "green":
      img = ImageOps.expand(img, border=15, fill='green')##left,top,right,bottom
    else:
      img = ImageOps.expand(img, border=15, fill='white')##left,top,right,bottom
      
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img


# for one row
def save_single_fig(which_row, imgs_path, matches):
    
    fig, ax = plt.subplots(1, 11, figsize = (40,10))
    for i, axi in enumerate(ax.flat):
        
        # no border color
        if i == 0:
            img = read_img(imgs_path[which_row, 0], "none")
        
        # border color: 0 == "red" 1 == "green"
        else:
            if matches[which_row, i-1] == 0:
                img = read_img(imgs_path[which_row, i], "red")
            else:
                img = read_img(imgs_path[which_row, i], "green")
        
        #pdb.set_trace()
        w, h, c = img.shape[1], img.shape[2], img.shape[3]
        axi.imshow(img.reshape(w, h, c))
        axi.set(xticks=[], yticks=[])
    
    fig.savefig("result-" + str(which_row) + ".jpg")
    

# from the start-th to end-th row
def save_figs(start, end, gall_img, query_img, indices, matches):
    
    gall_img = np.asarray(gall_img)
    query_img = np.asarray(query_img)
    
    gallery = gall_img[indices][:, :10]
    
    imgs_path = np.hstack([query_img.reshape(-1, 1), gallery])
    
    for row in range(start, end):
        save_single_fig(row, imgs_path, matches)
"""
"""
def read_img(img_path, color):
    img = Image.open(img_path)
    img = img.resize((144, 288))
    
    
    if color == "red":
      img = ImageOps.expand(img, border=15, fill='red')##left,top,right,bottom
    elif color == "green":
      img = ImageOps.expand(img, border=15, fill='green')##left,top,right,bottom
    else:
      img = ImageOps.expand(img, border=15, fill='white')##left,top,right,bottom
      
    np_img = np.array(img) # (224, 224, 3)
    np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
    return np_img



def save_fig(rows_for_each_pic, rows, gall_img, query_img, indices, matches):
    
    '''
        rows_for_each_pic: the rows for each pic
        rows: the nums of pic, each pic's size equals rows_for_each_pic * 11
        gall_img: the list of gallery's picture path
        query_img: the list of query's picture path
        indices: the ranked indexs matrix
        matches: the ranked match matrix, each value for 0 or 1
    '''
    
    gall_img = np.asarray(gall_img)
    query_img = np.asarray(query_img)
    
    #pdb.set_trace()
    # save the pig
    query_len = len(query_img)
    gallery = gall_img[indices][:, :10]
    
    imgs_path = np.hstack([query_img.reshape(-1, 1), gallery])
    #imgs_path = imgs_path.reshape(1, imgs_path.shape[0] * imgs_path.shape[1])
    
    #int rows_for_each_pic = 100
    for epoch in range(rows):
        fig, ax = plt.subplots(rows_for_each_pic, 11, figsize = (40,10))
        for i, axi in enumerate(ax.flat):
            
            # no border color
            if i % 11 == 0:
                img = read_img(imgs_path[(i // 11) + epoch * rows_for_each_pic, i%11], "none")
            # border color: 0 == "red" 1 == "green"
            else:
                if matches[i//10, i%10] == 0:
                    img = read_img(imgs_path[(i // 11) + epoch * rows_for_each_pic, i%11], "red")
                else:
                    img = read_img(imgs_path[(i // 11) + epoch * rows_for_each_pic, i%11], "green")
            
            #pdb.set_trace()
            w, h, c = img.shape[1], img.shape[2], img.shape[3]
            axi.imshow(img.reshape(w, h, c))
            axi.set(xticks=[], yticks=[])
        
        fig.savefig("result-" + str(1 + (rows_for_each_pic)*epoch) + "-" + str(rows_for_each_pic*(epoch+1)) + ".jpg")
"""



"""Cross-Modality ReID"""

#def eval_sysu(query_img,gall_img,distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    #save_figs(40, 80, gall_img, query_img, indices, matches)
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return new_all_cmc, mAP
    
    
    
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    #save_figs(0, 10, gall_img, query_img, indices, matches)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP