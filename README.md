#   facial_recognition_system
#   source ~/miniconda3/bin/activate 

# Environment Setup
```python 
conda install faceid torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate faceid
pip install opencv-python
pip install onnxruntime==1.20.1
pip install onnxruntime-gpu==1.20.1
git clone https://github.com/PhucNDA/FaceID--YOLOV5.ArcFace
cd FaceID--YOLOV5.ArcFace
```
Ensuring the right data tree format

    FaceID--YOLOV5.ArcFace
    ├── database_image
    │   ├── profile1.png
    |   ├── profile2.png
    |   ├── profile3.png
    |   ├── ...
    ├── database_tensor
    │   ├── profile1.npy
    |   ├── profile2.npy
    |   ├── profile3.npy
    |   ├── ...

**database_image**: containing image for each profile

**database_tensor**: containing vector feature extracted by pretrained backbone for each profile

'''
    python feature_extraction.py --weight 'weights/backbone.pth' --path_database database_image

    python converttoonnx.py

    python detection_gpu.py
'''

