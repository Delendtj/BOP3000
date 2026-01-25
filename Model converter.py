from ultralytics import YOLO
import shutil

model = YOLO('models/updated_model.pt')

#konverterer modellen til ONNX
'''
openVINO er straight bedre enn ONNX på intel cpu, men ONNX er cross platform, 
så den kan kjøre på både cpu og gpu, openVINO modeller liker ikke nvidia GPU

vi bytter til dedicated models etter vi låser ned hardware requirements?
'''

#FP32 Dette er CPU Modell
model.export(format='onnx', imgsz=640, simplify=True, half=False)
shutil.move('models/updated_model.onnx', 'models/model_fp32.onnx')

#FP16 Moderne GPU med tensor cores
model.export(format='onnx', imgsz=640, simplify=True, half=True)
shutil.move('models/updated_model.onnx', 'models/model_fp16.onnx')

'''
#FP32 modell for CPU
model.export(format='onnx', imgsz=640, simplify=True, half=False)

#FP16 modell for intel gpu
model.export(format='openvino', imgsz=640, simplify=True, half=True)
'''