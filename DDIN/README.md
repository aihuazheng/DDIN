# DDIN#
Code for paper "����˫��̬��������Ŀ�ģ̬������ʶ��"

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
### SYSU-MM01���ݼ�Ԥ����:python pre_process_sysu.py
ע���޸�·��
```
#ѵ����ڣ�
python train_LWPA_Dy_hu.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 1 --epochs 60 --w_hc 0.5 --per_img 8 
```

#�������(single-shot all-search)
```
python test_LWPA_hu.py --dataset sysu --lr 0.01 --drop 0.0 --trial 1 --gpu 1 --low-dim 512 --resume 'Your model name' --w_hc 0.5 --mode all --gall-mode single --model_path 'Your model path'
```
#����ṹ��
#���е�.log�ļ����������⣬ֻ��ѵ����¼
#���е�model�ļ�������model��������ͬ��ģ�ͽṹ������LWPA��ʾ����ֱ���ģ�飬DY��ʾ��̬���ģ�飬hu��ʾ������DY+hu��������˫��̬����ģ��
#���е�train��test��Ӧ��Ӧ��model��ѵ����ͬģ�͵�ʱ��ǵ���train��test���޸ĳɶ�Ӧ��model��һ��Ϊfrom model_xxx import embed_net

#model�еĳ�LWPA,DY��hu��ģ������������⣬��Ϊ�������顣
#view_rank.pyΪ���ӻ�rank�б�
#���ӻ�����ͼ��test�ļ��о���


#��������Ҫģ�����λ��
1.˫��̬����ģ��
��model_LWPA_Dy_hu.py��288-333��
2.����ֱ��������ں�ģ��
��model_LWPA_Dy_hu.py��335-341��
�����LWPA.py�ļ�
