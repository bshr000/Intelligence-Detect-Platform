import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'/workspace/weights/best-ogsod.pt')
    model.val(data=r'/workspace/code/YOLOv11-CMFM/data/OGSOD.yaml',
              split='val',
              project='runs/val',
              name='yolo11s-cmfm-ogsod',
              imgsz=640,
              batch=16,
              device=7,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              conf=0.4,
              save_json=False, # if you need to cal coco metrice
              plots=False
              )