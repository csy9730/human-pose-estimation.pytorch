cd "$(dirname "$0")/../.."

python pose_estimation/predictor.py --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml --model-file "output\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3\model_best.pth.tar" -i "data/*.jpg"
