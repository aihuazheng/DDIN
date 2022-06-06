# DDIN#
Code for paper "面向双向动态交互网络的跨模态行人重识别"

## Update:
2022-05-31:
we fix up a bug in learning rate schedule, before that only the first three parameter group's learning rate will be correctly decay to 1/10. However, after fixing up the bug, the model's performance still stay the same. The updated model and code have been upload.

## Requirments:
**pytorch: 0.4.1(the higher version may lead to performance fluctuation)**

torchvision: 0.2.1

numpy: 1.17.4

python: 3.7


## Dataset:
**SYSU-MM01**

**Reg-DB**


## Run:
### SYSU-MM01数据集预处理:python pre_process_sysu.py
注意修改路径
```
#训练入口：
python train_LWPA_Dy_hu.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 1 --epochs 60 --w_hc 0.5 --per_img 8 
```

#测试入口(single-shot all-search)
```
python test_LWPA_hu.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 1 --low-dim 512 --resume 'Your model name' --w_hc 0.5 --mode all --gall-mode single --model_path 'Your model path'
```
#代码结构：
#所有的.log文件，无需在意，只是训练记录
#所有的model文件，根据model命名代表不同的模型结构，其中LWPA表示跨层多分辨率模块，DY表示动态卷积模块，hu表示交互，DY+hu即完整的双向动态交互模块
#所有的train及test对应相应的model，训练不同模型的时候记得在train和test中修改成对应的model，一般为from model_xxx import embed_net

#model中的出LWPA,DY，hu等模型无需过多在意，均为方法试验。
#view_rank.py为可视化rank列表
#可视化特征图在test文件中就有


#论文中主要模块代码位置
1.双向动态交互模块
见model_LWPA_Dy_hu.py第288-333行
2.跨层多分辨率特征融合模块
见model_LWPA_Dy_hu.py第335-341行
定义见LWPA.py文件
