#import os
#from dotenv import load_dotenv
#env_path = '/home/jovyan/digilut/ENV'
#load_dotenv(env_path)

import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,warnings,pickle
from tqdm import tqdm
import torch
from torch import nn
import cv2

IMG_SIZE = int(sys.argv[1]) 

DIR_RAW_DATA = os.getenv('DIR_RAW_DATA')
DIR_PROCESSED_DATA = os.getenv('DIR_PROCESSED_DATA')
DIR_PREDICTIONS = os.getenv('DIR_PREDICTIONS')
DIR_MODELS = os.getenv('DIR_MODELS')

DIR_TILES = f'{DIR_PROCESSED_DATA}/workspace/test/tiles'
os.makedirs(DIR_PREDICTIONS,exist_ok=True)

N_SPLITS = 5

from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import mmcv

def load_model(cfg_path, model_path, scale):
    cfg = Config.fromfile(cfg_path)
    cfg.test_pipeline = [
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=scale, type='Resize'),
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
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    cfg.test_dataloader = cfg.val_dataloader
    
    model = init_detector(cfg, model_path, 'cuda:0')

    return model


def get_predictions(df,model):
    df_sub = df.drop_duplicates(subset='file_id')
    df_res = []
    for _,row in tqdm(df_sub.iterrows(),total=len(df_sub)):
        img = cv2.imread(row.path)
        result = inference_detector(model, img)
        if len(result.pred_instances)>0:
            bbox = result.pred_instances.bboxes.cpu().numpy().tolist()
            conf = result.pred_instances.scores.cpu().numpy().tolist()
            d=pd.DataFrame(dict(conf=conf,bbox=bbox))
            d['file_id'] = row.file_id
            df_res.append(d)

    df_res = pd.concat(df_res)
    return df_res

def get_model_predictions(model,base_sz,tile_sz):
    paths = glob.glob(f'{DIR_TILES}/{base_sz}/{page}_{tile_sz}/*/img/*')
    df = pd.DataFrame(dict(path=paths))
    df['slide_id'] = df.path.str.split('/').str[-3]
    df['sz'] = df.path.str.split('/').str[-4]
    df['file_id'] = df.slide_id+'_'+df.sz+'_'+df.path.str.split('/').str[-1]
    df['tile_id'] = df.path.str.split('/').str[-1].str.split('_').str[0].astype(np.int32)

    paths=glob.glob(f'{DIR_TILES}/{base_sz}/{page}_{tile_sz}/*/*.csv')
    print('generating predictions for ',len(paths), 'slides')
    d_tiles=[pd.read_csv(p) for p in paths]
    d_tiles=pd.concat(d_tiles).sort_values(by=['slide_id','tile_id']).reset_index(drop=True)
    d_tiles = d_tiles.rename(columns={'w':'twidth','h':'theight'})
    df = pd.merge(df,d_tiles,on=['slide_id','tile_id'])

    res = get_predictions(df,model)
    df_res = pd.merge(df,res,on='file_id',how='left').drop(columns=['path'])

    return df_res

tile_sz = IMG_SIZE
page = 3
base_sz = int(tile_sz/(2**(5-page)))
scale = (IMG_SIZE,IMG_SIZE)
out_dir = f'{DIR_PREDICTIONS}/{tile_sz}'
os.makedirs(out_dir,exist_ok=True)

for fold in range(N_SPLITS):
    model_dir = f'{DIR_MODELS}/{IMG_SIZE}/model_{IMG_SIZE}_f{fold}'
    model_path = f'{model_dir}/epoch_12.pth'
    cfg_path = f'{model_dir}/faster-rcnn_r50_fpn_1x_coco.py'
    model = load_model(cfg_path, model_path, scale)
    df_res = get_model_predictions(model,base_sz,tile_sz)
    path = f'{out_dir}/test-{tile_sz}_{fold}.pkl'
    print(f'saving predictions to {path}')
    df_res.to_pickle(path)
