import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"/mnt/data/zxy/YOLOv11-RGBT-vedai/train_program_ogsod/runs/train/yolo11s-hybridfusion-CMFM(+gate)-400/weights/best.pt") # select your model.pt path
    model.predict(source=r"/mnt/data/zxy/dataset/ogsod/OGSOD_trainval.txt",
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBT",
                  channels=4,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )