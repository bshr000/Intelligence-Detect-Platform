import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/mnt/data/zxy/32/YOLOv11-RGBT-vedai/train_program_ogsod/runs/train/yolo11s-hybridfusion-CMFM-400-OGSOD-True/weights/best.pt')
    model.val(data=r'/mnt/data/zxy/32/YOLOv11-RGBT-vedai/data/OGSOD_harbor.yaml',
              split='val',
              project='runs/val',
              name='yolo11s-cmfm-harbor-test',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              )