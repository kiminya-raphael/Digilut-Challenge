# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,warnings,argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
from torch import nn
import cv2

# parser = argparse.ArgumentParser()

IMG_SIZE = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
LOSS_FUNC = sys.argv[3]
FOLD = int(sys.argv[4])

# IMG_SIZE = 512;  BATCH_SIZE = 16; LOSS_FUNC = 'ce'
# IMG_SIZE = 640;  BATCH_SIZE = 16; LOSS_FUNC = 'foc'
# IMG_SIZE = 768; BATCH_SIZE = 16; LOSS_FUNC = 'ce'
# IMG_SIZE = 896; BATCH_SIZE = 12; LOSS_FUNC = 'foc'
# IMG_SIZE = 1024; BATCH_SIZE = 8; LOSS_FUNC = 'ce'

RANDOM_STATE = IMG_SIZE
N_SPLITS = 5

# DIR_PROCESSED_DATA = '/home/jovyan/digilut/data/processed'
# DIR_MODELS = '/home/jovyan/digilut/models'

DIR_PROCESSED_DATA = os.environ['DIR_PROCESSED_DATA']
DIR_MODELS = os.getenv('DIR_MODELS')
DIR_TRAIN_DATASET =  f'{DIR_PROCESSED_DATA}/p3'



def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")





df = pd.read_pickle(f'{DIR_TRAIN_DATASET}/anno.pkl')
df_neg = pd.read_pickle(f'{DIR_TRAIN_DATASET}/neg.pkl')
df['path'] = DIR_TRAIN_DATASET +'/images/'+ df.file_id
df_neg['path'] = DIR_TRAIN_DATASET +'/images/'+ df_neg.file_id
# print('all df_neg',len(df_neg))
df_neg['patch_size'] = df_neg.sz.str.split('_').str[1].astype(int)
df_low_hq = df_neg[(df_neg.patch_size<IMG_SIZE) & (df_neg.Score>=.3)].copy()
df_neg = df_neg[df_neg.patch_size==IMG_SIZE].copy().reset_index(drop=True)
if len(df_low_hq)>0:
    df_neg = pd.concat([df_low_hq,df_neg]).reset_index(drop=True)

# print('new df_neg',len(df_neg))

df_slides = df[['slide_id']].drop_duplicates().reset_index(drop=True)

def split_data(df):
    kf  = KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    df['fold'] = -1

    for fold_id, (train_index, test_index) in enumerate(kf.split(df)):
    # for fold_id, (train_index, test_index) in enumerate(gkf.split(df,df.category)):
        df.loc[test_index,'fold'] = fold_id
    return df

def get_splits(df,fold):
    df_trn = df[df.fold!=fold].copy()
    df_val = df[df.fold==fold].copy()

    return df_trn,df_val

df_slides = split_data(df_slides)

# val_slides = ['1Im8CC1CC4_b', '42B8pnFQUm_a', '4ooLAejU32_a', '6Ie5p2WuhA_a', '6TJbZR5ssD_a', '6UX55R871W_b', 'C4ZsQw520P_b', 'ELfWcuBmIQ_b', 'EvCh35AKcd_a', 'FSlCJqyahA_b', 'Fn4HYIDBKM_a', 'Fn4HYIDBKM_b', 'GvAgxdkkx5_b', 'JvxiXClFKl_a', 'KuWAlQ7Uim_a', 'LOKtENseMR_a', 'NUrwibHLHL_a', 'NUrwibHLHL_b', 'P6q9YockQP_a', 'SHpdYEvPRk_a', 'StKznQnYkK_a', 'ZfxfH2xiL4_a', 'ZfxfH2xiL4_b', 'as3EWwqKEE_a', 'd9YWQcTLzD_a', 'fGTMeSz7fy_a', 'fGTMeSz7fy_b', 'fHPzaszsFO_a', 'flo7JJldPf_a', 'flo7JJldPf_b', 'hpY4Q7BFsz_b', 'hqi5y2OzZy_a', 'iinx2cvN22_a', 'j7lnUcdLs9_a', 'ju6xhS2O19_a', 'kGR3yrYLds_b', 'kiAdmoDDMM_b', 'lGm6TmVXhe_b', 'mReC9GXuTz_a', 'mVcdQIceEr_a', 'nqUWcVW9rx_a', 'pxYb4VfAwX_a', 'rDPDbEQidT_a', 'tN07oAdLhU_a', 'vLpr7qRDqk_a', 'vLpr7qRDqk_b', 'w3DXwGBBdw_b']
val_slides = df_slides[df_slides.fold==FOLD].slide_id.tolist()
df['fold'] = -1
df.loc[df.slide_id.isin(val_slides),'fold'] = FOLD

df_neg['fold'] = -1
df_neg.loc[df_neg.slide_id.isin(val_slides),'fold'] = FOLD

# print(df.fold.value_counts())
# print(df_neg.fold.value_counts())

#filter partial bbs (<90% area)
min_perc = 0.9
df['bb_w'] =  df.apply(lambda d: d.crop_bbox[2]-d.crop_bbox[0] ,axis=1)
df['bb_h'] =  df.apply(lambda d: d.crop_bbox[3]-d.crop_bbox[1] ,axis=1)

df['bb_id'] = df.og_bbox.astype(str)
df['bb_area'] = df.bb_h*df.bb_w
d = df.groupby(['bb_id','slide_id']).agg(n=('bb_id','count'),max_bb_area=('bb_area','max'),).reset_index()
df = pd.merge(d,df)
df['perc'] = df.bb_area/df.max_bb_area

print(len(df)-len(df[df.perc>=min_perc]))
df = df[df.perc>=min_perc]

# get data splits
fold = FOLD
df_trn_pos,df_val_pos = get_splits(df,fold)
df_trn_neg,df_val_neg = get_splits(df_neg,fold)
# print('all val pos:',df_val_pos.shape)
df_val_pos = df_val_pos[df_val_pos.perc==1]
# print('new val pos:',df_val_pos.shape)

df_trn = pd.concat([df_trn_pos,df_trn_neg])
# df_trn = df_trn_pos
# df_val = pd.concat([df_val_pos,df_val_neg])

df_val = df_val_pos.copy()
df_trn.shape,df_val.shape
df_trn['split'] = 'train'
df_val['split'] = 'val'


cols = ['path', 'slide_id', 'file_id', 'crop_bbox', 'og_bbox', 'height', 'width', 'category_id', 'fold','split']
meta = pd.concat([df_trn,df_val]).reset_index(drop=True)[cols]

#generate annotation files
def gen_anno(meta,split,json_filename):
    meta1 = meta[meta.split==split].copy()
    annotations = []
    images = []
    anno_id = 0
    image_id = 0
    for file_name,d in tqdm(meta1.groupby('file_id')):
        path = d.path.values[0]
        im = cv2.imread(path)
        height,width,_ = im.shape
#         shutil.copy2(path, f'{img_dir}/{file_name}')
        d = d.dropna(subset=['crop_bbox'])
        for _,row in d.iterrows():
            # print(file_name,width,row.width,height,row.height)
            assert width==row.width
            assert height==row.height
            x0, y0, x1, y1 = np.array(row.crop_bbox).round().astype(int).tolist()
            w = x1 - x0
            h = y1 - y0
            bbox = np.array([x0, y0, w, h]).tolist()
            area = w * h
            # assert area >0
            if area<1:
                print('skipping',file_name,area)
                continue

            anno = dict(
                        image_id = image_id,
                        id = anno_id,
                        category_id = row.category_id,
                        bbox = bbox,
                        area = area,
                        iscrowd = 0
                    )
            anno_id += 1
            annotations.append(anno)

        images.append(dict(id=image_id, file_name=file_name,height=height,width=width))
        image_id += 1


    coco_json = dict(images=images, annotations=annotations,
                        categories=[dict(id=0,name='tp')])
                        # categories=[dict(id=0,name='fp'),dict(id=1,name='tp')])
    path = f'{DIR_TRAIN_DATASET}/{json_filename}'
    print(f'written {len(annotations)} annotations for {len(images)} images to {path}')
    with open(path,'w', encoding='utf-8') as f:
        json.dump(coco_json,f,ensure_ascii=False)


train_json_filename = f'train_sz{IMG_SIZE}_r{RANDOM_STATE}_f{FOLD}.json'
val_json_filename = f'val_sz{IMG_SIZE}_r{RANDOM_STATE}_f{FOLD}.json'

gen_anno(meta,'train',train_json_filename)
gen_anno(meta,'val',val_json_filename)



from mmengine import Config
from mmengine.runner import Runner
import mmdet

# %cd ../../mmdetection

cfg = 'faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(f'configs/{cfg}')
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# customize loss functions
gamma=0.5; alpha=0.25
cfg.model.roi_head.bbox_head=dict(
    bbox_coder=dict(
        target_means=[
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        target_stds=[
            0.1,
            0.1,
            0.2,
            0.2,
        ],
        type='DeltaXYWHBBoxCoder'),
    fc_out_channels=1024,
    in_channels=256,
    loss_bbox=dict(loss_weight=1.0, type='EIoULoss'),
    loss_cls=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
    num_classes=1,
    reg_class_agnostic=False,
    reg_decoded_bbox=True,
    roi_feat_size=7,
    type='Shared2FCBBoxHead')

if LOSS_FUNC=='foc':
    cfg.model.roi_head.bbox_head.loss_cls = dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=gamma,alpha=alpha,
    loss_weight=1.0)



img_prefix = 'images'
# max_epochs = 1
# cfg.train_batch_size_per_gpu = 4
# sz_train = sz_test = (32,32)

max_epochs = 12
cfg.train_batch_size_per_gpu = BATCH_SIZE
sz_train = sz_test = (IMG_SIZE,IMG_SIZE)

val_interval = max_epochs
cfg.train_num_workers = 2
MAX_PROPOSALS = 100; MAX_OBJECTS = 10
cfg.model.train_cfg.rpn_proposal.max_per_img = MAX_PROPOSALS
cfg.model.test_cfg.rpn.max_per_img = MAX_PROPOSALS
cfg.model.test_cfg.rcnn.max_per_img = MAX_OBJECTS
cfg.model.train_cfg.rpn_proposal.nms_pre = MAX_PROPOSALS
cfg.model.test_cfg.rpn.nms_pre = MAX_PROPOSALS

cfg.data_root = DIR_TRAIN_DATASET
cfg.work_dir = f'./output_sz{IMG_SIZE}_r{RANDOM_STATE}_f{FOLD}'
metainfo = {
    'classes': ('tp',),
    'palette': [
        (128, 60, 30),
    ]
}
cfg.dataset_type = 'CocoDataset'
num_classes = 1
cfg.model.roi_head.bbox_head.num_classes = num_classes

cfg.train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='MinIoURandomCrop', min_ious=(0.75,0.8,0.85,0.9,0.95), min_crop_size=0.5),
    dict(keep_ratio=True, scale= sz_train, type='Resize'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Sharpness',prob=0.5),
    dict(type='AutoContrast',prob=0.5,min_mag=0.1,max_mag=1.9,level=10),

    dict(type='Rotate', level=10, min_mag=180.,max_mag=180.,prob=0.5),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
cfg.test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale= sz_test, type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]


cfg.train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=cfg.train_batch_size_per_gpu,
    dataset=dict(
        data_root=cfg.data_root,
        metainfo=metainfo,
        ann_file=train_json_filename,
        backend_args=None,
        data_prefix=dict(img=img_prefix),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=cfg.train_pipeline,
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

cfg.val_dataloader = dict(
    dataset=dict(
        data_root=cfg.data_root,
        metainfo=metainfo,
        data_prefix=dict(img=img_prefix),
        ann_file=val_json_filename,
        pipeline = cfg.test_pipeline,
        test_mode=True,
    type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))

cfg.test_dataloader = cfg.val_dataloader


cfg.load_from = load_from
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
cfg.visualizer = dict( name='visualizer',type='DetLocalVisualizer',vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
cfg.val_evaluator = cfg.test_evaluator = dict(
    ann_file=f'{DIR_TRAIN_DATASET}/{val_json_filename}',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
    ],
    type='CocoMetric')

fix_seed(RANDOM_STATE)
runner = Runner.from_cfg(cfg)
runner.train()



model_path = f'{cfg.work_dir}/epoch_{max_epochs}.pth'
model_dir = f'{DIR_MODELS}/{IMG_SIZE}/model_{IMG_SIZE}_f{FOLD}'
os.makedirs(model_dir,exist_ok=True)
print('saving model to {model_dir}')
shutil.copy(model_path,model_dir)
for f in glob.glob(f'{cfg.work_dir}/*.py'):
    shutil.copy(f,model_dir)