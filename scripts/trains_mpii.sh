cd "$(dirname "$0")/.."
# export PYTHONUNBUFFERED="True"
LOG="./logs/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

python pose_estimation/train.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3_a.yaml 2>&1 |tee $LOG
# H:\Dataset\keypoint\MPII