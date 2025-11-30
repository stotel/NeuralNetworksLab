from ultralytics import YOLO

model = YOLO('yolov8m.pt')

if __name__ == '__main__':
    results = model.train(data='../dataset/data.yaml', epochs=50, name='yolov8_tuned')