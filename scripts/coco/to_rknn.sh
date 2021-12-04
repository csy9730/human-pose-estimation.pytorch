cd "$(dirname "$0")/../.."

# python -m zalaiConvert.convert.onnx2rknn weights/coco_256x192.onnx -o coco_256x192.rknn --framework onnx --normalize-params 124.075 116.28  103.53   57.120003 # 58.395 57.375 

python -m zalaiConvert.convert.onnx2rknn weights/coco_256x192.onnx -o coco_256x192_q.rknn --framework onnx --do-quantization --dataset 1.txt --normalize-params 124.075 116.28  103.53 57.120003