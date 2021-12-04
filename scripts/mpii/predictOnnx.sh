cd "$(dirname "$0")/../.."

# checkpoint.pth.tar model_best
python pose_estimation/predict_onnx.py --model-in weights/pose_resnet_50_256x256.onnx -i "data/*.jpg"
