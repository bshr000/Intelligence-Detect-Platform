import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/mnt/data/zxy/32/YOLOv11-RGBT-vedai/train_program_ogsod/runs/train/yolo11s-CMFM-Oil-Tank/weights/best.pt')
    model.val(data=r'/mnt/data/zxy/32/YOLOv11-RGBT-vedai/data/OGSOD_oil_tank.yaml',
              split='val',
              project='runs/val',
              name='yolo11s-cmfm-oil_tank',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              conf=0.45,
              save_json=True, # if you need to cal coco metrice
              )