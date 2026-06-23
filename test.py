from ultralytics import YOLO

model = YOLO(r'D:\PotatoProject\runs\segment\train-5\weights\best.pt')
results = model.predict(source=r'D:\PotatoProject\dataset\test\images', save=True, conf=0.25)