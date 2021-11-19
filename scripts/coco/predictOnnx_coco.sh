cd "$(dirname "$0")/../.."

# checkpoint.pth.tar model_best
python pose_estimation/predict_onnx.py --model-in weights/coco_256x192.onnx -i "data/abc*.jpg" --input-size  256 192 --scale 256
