cd "$(dirname "$0")/../.."

# python -m zalaiConvert.convert.onnx2rknn weights/face300b_256x256.onnx -o weights/face300b_256x256_q.rknn --framework onnx --do-quantization --dataset $(dirname "$0")/1.txt --normalize-params 0.485 0.456 0.406 0.225 --device rk1808

python -m zalaiConvert.convert.onnx2rknn weights/face300b_256x256.onnx -o weights/face300b_256x256.rknn --framework onnx  --normalize-params 0.485 0.456 0.406 0.225 --device rk1808
