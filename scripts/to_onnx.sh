cd "$(dirname "$0")/.."
python pose_estimation/exporter_onnx.py --cfg experiments/face300w/256x256_d256x3_adam_lr1e-3_a.yaml 
--model-file  output/CsvKptDataset/pose_resnet_50/256x256_d256x3_adam_lr1e-3_b/model_best.pth.tar --shape-list 1 3 256 256 -o weights/face300b_256x256.onnx
