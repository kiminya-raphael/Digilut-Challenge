cd mmdetection && mim download mmdet --config  faster-rcnn_r50_fpn_1x_coco --dest ./checkpoints
cd ../
gdown -O models.zip https://drive.google.com/uc?id=1-KHHUBY9P7FDHypQIozEVYdeAGQxAsEr
unzip -qqn models.zip 
