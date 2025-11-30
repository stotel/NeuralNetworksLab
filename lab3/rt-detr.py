from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')

if __name__ == '__main__':
    results = model.train(
        data='../dataset/data.yaml',
        epochs=25,
        imgsz=640,
        batch=8,
        name='rtdetr_tuned',
        workers=0
    )

    print("✅ Training completed. Best model saved in runs/detect/rtdetr_tuned/weights/best.pt")