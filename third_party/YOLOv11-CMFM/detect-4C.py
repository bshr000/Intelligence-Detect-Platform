import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"D:/Codex/AI_Workspace/03_AI_Engineering/YOLO-CMFM-Platform/weights/yolo11s-ogsod-best.pt") # select your model.pt path
    model.predict(source=r"D:/Codex/AI_Workspace/03_AI_Engineering/YOLO-CMFM-Platform/images/visible/1__1__0___0.png",
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