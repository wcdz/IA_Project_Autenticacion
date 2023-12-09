# libraries
from ultralytics import YOLO

# Model
model = YOLO('modelos/yolov8m.pt')


def main():
    # train
    model.train(data='objectDetection/SplitData/Dataset.yaml', epochs=1, batch=4, imgsz=640)


if __name__ == '__main__':
    main()
