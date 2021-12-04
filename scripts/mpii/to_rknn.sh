cd "$(dirname "$0")/../.."

# python -m zalaiConvert.convert.onnx2rknn weights/pose_resnet_50_256x256.onnx -o weights/pose_resnet_50_256x256.rknn --framework onnx --normalize-params 124.075 116.28  103.53 57.120003 --dataset 1.txt --device rk1808

python -m zalaiConvert.convert.onnx2rknn weights/pose_resnet_50_256x256.onnx -o weights/pose_resnet_50_256x256_q.rknn --framework onnx --normalize-params 124.075 116.28  103.53 57.120003 --dataset 1.txt --device rk1808 --do-quantization


# python -m zalaiConvert.convert.onnx2rknn pose_resnet_50_256x256.onnx -o pose_resnet_50_256x256.rknn --framework torchscript --input-size-list  3 256 256 --dataset 1.txt