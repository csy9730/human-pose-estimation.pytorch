cd "$(dirname "$0")/../.."

# checkpoint.pth.tar model_best
python pose_estimation/predictor.py --cfg experiments/face300w/256x256_d256x3_adam_lr1e-3_c.yaml --model-file "output/CsvKptDataset/pose_resnet_50/256x256_d256x3_adam_lr1e-3_c/checkpoint.pth.tar" -i "data/faces/a*g"