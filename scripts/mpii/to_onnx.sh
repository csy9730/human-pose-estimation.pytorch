cd "$(dirname "$0")/../.."
python pose_estimation/exporter_onnx.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml   --model-file  models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar --shape-list 1 3 256 256 -o weights/pose_resnet_50_256x256.onnx
