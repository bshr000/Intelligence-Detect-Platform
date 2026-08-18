import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/mnt/data/zxy/YOLOv11-RGBT-vedai/multimodels/hybridfusion-ogsod/11s-hybridfusion-CMFM_lega_cross_gate2.yaml')
    # model.info(True,True)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'/mnt/data/zxy/32/YOLOv11-RGBT-vedai/data/OGSOD_bridge.yaml',
                project='runs/train',
                name='yolo11s-CMFM-bridge',
                device='5',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=16,
                close_mosaic=10,
                workers=2,
                optimizer='SGD',  # using SGD
                # lr0=0.002,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                )