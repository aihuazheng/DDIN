from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
#import torchvision
#import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from  model_dy_nohu_lwpa import embed_net
from utils import *
import time 
import scipy.io as scio
import Transform as transforms
import cv2
import numpy as np
import pdb
from gradcam import *
from feature_visualizer import FeatureVisualizer

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='sysu_id_bn_relu_drop_0.0_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_resnet50_best.t', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='/DATA/fengmengya/save_model/dy_nohu_lwpa/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='Method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=8, type=int,
                    help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.5, type=float,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--thd', default=0, type=float,
                    help='threshold of Hetero-Center Loss')
parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)   
np.random.seed(1)
random.seed(1)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/panpeng/SYSU/'
    log_path = args.log_path + 'sysu_log/'
    n_class = 395
    test_mode = [1, 2] 
elif dataset =='regdb':
    data_path = '/DATA/fengmengya/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    n_class = 206
    test_mode = [2, 1]
    
if not os.path.isdir(log_path):
    os.makedirs(log_path)

if args.method =='id':
    suffix = dataset + '_id_bn_relu'
suffix = suffix + '_drop_{}'.format(args.drop)
suffix = suffix + '_lr_{:1.1e}'.format(args.lr)
suffix = suffix + '_dim_{}'.format(args.low_dim)
suffix = suffix + '_whc_{}'.format(args.w_hc)
suffix = suffix + '_thd_{}'.format(args.thd)
suffix = suffix + '_pimg_{}'.format(args.per_img)
suffix = suffix + '_gm_{}'.format(args.gall_mode)
suffix = suffix + '_m_{}'.format(args.mode)
test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
print('==> LWPA_Dy_noHu_all_single')
print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch,lma = True)
net.to(device)    
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        print('==> loading checkpoint {}'.format(args.resume), file=test_log_file)
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']), file=test_log_file)
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
        print('==> no checkpoint found at {}'.format(args.resume), file=test_log_file)


if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
print('==> Loading data..', file=test_log_file)
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((args.img_h,args.img_w)),
    transforms.RectScale(args.img_h, args.img_w),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    # testing set
    #pdb.set_trace()
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0, gall_mode=args.gall_mode)
   

      
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    #color_pos, visible_pos = GenIdx(trainset.train_color_label, trainset.train_visible_label)
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')
    
    gallset  = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
nquery = len(query_label)
ngall = len(gall_label)
print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

print("Dataset statistics:", file=test_log_file)
print("  ------------------------------", file=test_log_file)
print("  subset   | # ids | # images", file=test_log_file)
print("  ------------------------------", file=test_log_file)
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery), file=test_log_file)
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall), file=test_log_file)
print("  ------------------------------", file=test_log_file)

queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))   
#pdb.set_trace()
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)

print('Data Loading Time:\t {:.3f}'.format(time.time()-end))
print('Data Loading Time:\t {:.3f}'.format(time.time()-end), file=test_log_file)

feature_dim = args.low_dim

if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512
#view
   
"""
def save_featmap(feat, name, output_dir,colormode=2):
    #pdb.set_trace()
    if not os.path.exists(output_dir):
        p = os.path.abspath(output_dir)
        os.mkdir(p)
        print("dir dose not exist, make it:"+p)
    
    shape = feat.shape
    if len(shape) != 3:
        raise Exception("input feat should be a 3-dim tensor")

    C, H, W = shape
    target_H, target_W = H, W
    flag_resize = False
    if H < 32 or W < 32:
        flag_resize = True

    feat = feat.cuda().data.cpu().numpy()
    fmin = np.min(feat)
    fmax = np.max(feat)
    print(fmax, fmin)
    for i in range(C):
        #pdb.set_trace()
        map_name = name + '_c{}'.format(i)
        featmap = feat[i, :, :]
        featmap = (featmap - fmin)/(fmax - fmin + 0.0001)
        featmap = (featmap * 255).astype(np.uint8)
        featmap = cv2.applyColorMap(np.array(featmap,dtype=np.uint8), colormode)
        if flag_resize:
            featmap = cv2.resize(featmap, (W*5, H*5), interpolation=cv2.INTER_LINEAR)
            map_name += '_upsamp'
        map_name += '.jpg'

        cv2.imwrite(os.path.join(output_dir, map_name), featmap)
     # feature mean
    feature_mean=featmap.mean(dim = 1, keepdim=True)  #n,c,h,w
    #print("feature_mean",feature_mean.shape)  #n,1,h,w
    feature_mean=feature_mean*255/torch.max(feature_mean+1e-14)
    feature_mean = F.interpolate(feature_mean, size=(617,216), mode='bilinear', align_corners=False)
    feature_mean=feature_mean.squeeze().data.cpu().numpy()
    feature_mean = cv2.applyColorMap(np.array(feature_mean,dtype=np.uint8), colormode)
    mean_name = '_mean.jpg'
    cv2.imwrite(os.path.join(output_dir, map_name),feature_mean)
""" 

def torch_vis_color(name,feature_tensor,col,raw,save_path,colormode=2,margining=1):
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
    #pdb.set_trace()
    show_k = col * raw    # total num
    f = feature_tensor[0, :show_k, :, :]  # n,c,h,w
    
    size = f[0, :, :].shape  #h*w
    f = f.data.cpu().numpy()
    fmin = np.min(f)
    fmax = np.max(f)
    print(fmax, fmin)
    for i in range(raw):
        f = (f - fmin)/(fmax - fmin + 0.0001)
        tem = f[i * col, :, :]*255/(np.max(f[i * col, :, :]+1e-14))
        #print("tem",tem.shape)
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
    #print(final.shape)
    #cv2.imwrite(save_path+name+'.jpg',final) 
    
    #cv2.imwrite(save_path+name+str(col)+'*'+str(raw)+'.png',final)  
     # feature mean
    feature_mean=feature_tensor.mean(dim=1,keepdim=True)  #n,c,h,w
    #print("feature_mean",feature_mean.shape)  #n,1,h,w
    feature_mean=feature_mean*255/torch.max(feature_mean+1e-14)
    feature_mean = F.interpolate(feature_mean, size=(617,216), mode='bilinear', align_corners=False)
    feature_mean=feature_mean.squeeze().data.cpu().numpy()
    feature_mean = cv2.applyColorMap(np.array(feature_mean,dtype=np.uint8), colormode)
    cv2.imwrite(save_path+name+'mean.jpg',feature_mean)

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    print('Extracting Gallery Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 8*feature_dim))
    gall_feat_pool = np.zeros((ngall, 8*pool_dim))
    #gall_s3_hu = np.zeros((nquery,2*feature_dim,18,9))
    #gall_label = np.zeros((nquery))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            
            pool_feat, feat,s3_hu  = net(input, input, test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            gall_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            #gall_s3_hu[ptr:ptr+batch_num,:,:,:] = s3_hu.detach().cpu().numpy()
            #gall_label[ptr:ptr+batch_num] = label.detach().cpu().numpy()
            ptr = ptr + batch_num 
            
           
    print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_pool
    
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    print('Extracting Query Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 8*feature_dim))
    query_feat_pool = np.zeros((nquery, 8*pool_dim))
    query_s3_hu = np.zeros((nquery,2*feature_dim,18,9))
    query_label = np.zeros((nquery))
    with torch.no_grad():
       
        for batch_idx , (input, label ) in enumerate(query_loader):
            #pdb.set_trace()
            batch_num = input.size(0)
            input = Variable(input.cuda())
            #input = input[:1,:,:,:]
            pool_feat, feat,s3_hu= net(input, input, test_mode[1])
            #print(" input:{}, label:{}".format( input, label))
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            query_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            query_s3_hu[ptr:ptr+batch_num,:,:,:] = s3_hu.detach().cpu().numpy()
            query_label[ptr:ptr+batch_num] = label.detach().cpu().numpy()
            ptr = ptr + batch_num 
                       
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    return query_feat , query_feat_pool,query_s3_hu,query_label




#person vis
"""
query_feat, query_feat_pool,query_s3_hu,query_label = extract_query_feat(query_loader) 
query_s3_hu = torch.from_numpy(query_s3_hu)

f = query_s3_hu[369:370,:,:,:]  # torch.Size([4, 2048,  8,   4])
query_label_now = query_label[369]
print(query_label_now)
## Feature visualizer
V = FeatureVisualizer(
    cmap_type='jet',
    reduce_type='mean',
    upsample_size=(18, 9),
    upsample_type='bilinear',
)

## Visualize Feature
V.save_feature(f, save_path='/DATA/fengmengya/save_image/demo/feature.jpg')
#person vis
"""

#pdb.set_trace()

query_feat, query_feat_pool,query_s3_hu,query_label = extract_query_feat(query_loader) 

query_s3_hu = torch.from_numpy(query_s3_hu)
query_cam6_0117_0008 = query_s3_hu[1750:1751:,:,:]
query_label_now = query_label[1750]
print(query_label_now)
view_query_s3_hu = torch_vis_color('cam6_0117_0008',query_cam6_0117_0008,32,32,'/DATA/panpeng/fmy/save_image/LWPA_DY/',colormode=2,margining=1) 



#pdb.set_trace() 

#view_query_s3_hu = save_featmap(query_s3_hu,'query_s3_hu','/DATA/fengmengya/save_image/query_s3_hu/test/')
#view_query_s3_hu = torch_vis_color(query_s3_hu,32,32,'/DATA/fengmengya/save_image/li_query_s3_hu/',colormode=2,margining=1)
#save_query = save_image('/DATA/fengmengya/save_image/query/',query)
#view_query_s3_hu = torch_vis_color('cam6_0006',query_s3_hu,32,32,'/DATA/fengmengya/save_image/li_query_s3_hu/',colormode=2,margining=1) 





all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_gall_feat(gall_loader)
    # fc feature 
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP  = eval_regdb(query_img,gall_img,-distmat, query_label, gall_label)
    
    # pool5 feature
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool = eval_regdb(query_img,gall_img,-distmat_pool, query_label, gall_label)

    print ('Test Trial: {}'.format(args.trial))
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))

    print('Test Trial: {}'.format(args.trial), file=test_log_file)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP), file=test_log_file)
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)

  
elif dataset =='sysu':
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = trial, gall_mode=args.gall_mode)
        
        #pdb.set_trace()
        trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
        trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        gall_feat, gall_feat_pool= extract_gall_feat(trial_gall_loader)
        
        #gall_s3_hu = torch.from_numpy(gall_s3_hu)
        #gall_now = gall_s3_hu[:10,:,:,:]
        #gall_label_now = gall_label[:10]
        #cv2.imwrite('/DATA/fengmengya/gallery_image/'+'finel'+'.jpg',gall_img[:1])
        #imshow(gall_img[:1])
        #print(gall_now)
        #print(gall_label_now)
        #view_gall_s3_hu = torch_vis_color('gall_now',gall_now,32,32,'/DATA/fengmengya/save_image/li_gall_s3_hu/',colormode=2,margining=1) 
    
        # fc feature 
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP  = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
        
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial ==0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        
        print ('Test Trial: {}'.format(trial))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        print('mAP: {:.2%}'.format(mAP_pool))

        print('Test Trial: {}'.format(trial), file=test_log_file)
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
        print('mAP: {:.2%}'.format(mAP), file=test_log_file)
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
        print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)

    cmc = all_cmc /10 
    mAP = all_mAP /10

    cmc_pool = all_cmc_pool /10 
    mAP_pool = all_mAP_pool /10
    print ('All Average:')
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))

    print('All Average:', file=test_log_file)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP), file=test_log_file)
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]), file=test_log_file)
    print('mAP: {:.2%}'.format(mAP_pool), file=test_log_file)