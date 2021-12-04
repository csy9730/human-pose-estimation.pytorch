cd "$(dirname "$0")/../.."

# checkpoint.pth.tar model_best failed
# python pose_estimation/predict_rknn.py weights/tmp/pose_resnet_50_256x256_q.rknn --device rk1808 -i "data/*.jpg"   --show-img --with-normalize

# failed
python pose_estimation/predict_rknn.py weights/pose_resnet_50_256x256_q.rknn --device rk1808 -i "data/*.jpg"   --show-img
