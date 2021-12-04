cd "$(dirname "$0")/../.."

# checkpoint.pth.tar model_best
python pose_estimation/predict_rknn.py weights/face300b_256x256.rknn --device rk1808 -i "data/faces/*.png"   --show-img
