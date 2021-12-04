cd "$(dirname "$0")/../.."

# success
python pose_estimation/predict_rknn.py weights/coco_256x192_q.rknn --device rk1808 -i "data/*.jpg"   --show-img --input-size 256 192
