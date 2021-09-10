# csy read

## main
### train
```
./scripts/trains.sh
```
### valid
```
python pose_estimation/valid.py 
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
    --flip-test 
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### export onnx

```
python pose_estimation/exporter.py --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml --flip-test --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### export rknn
```
python -m zalaiConvert.convert.onnx2rknn pose_resnet_50_256x256.onnx -o pose_resnet_50_256x256.rknn --framework onnx

python -m zalaiConvert.convert.onnx2rknn abc.pth -o pose_resnet_50_256x256.rknn --framework torchscript --input-size-list  3 256 256 --dataset 1.txt

python -m zalaiConvert.convert.onnx2rknn pose_resnet_50_256x256.onnx -o pose_resnet_50_256x256_q.rknn --framework onnx --do-quantization --dataset 1.txt

python -m zalaiConvert.farward.farward_tmc pose_resnet_50_256x256.rknn --device rk1808 

python -m zalaiConvert.farward.farward_tmc pose_resnet_50_256x256_q.rknn --device rk1808 
```

## source

### dataset
数据集使用了COCO人体数据集，MPII人体数据集。
#### preprocess

基本的图像通用前处理。
标注的关键点前处理：

``` python
def xywh2cs(x,y,w,h):
    # 把box转换成中心点和scale值，scale的基准是200.
    return c,s

class COCODataset(JointsDataset):
	def __init__(self):
		# 描述关键点执行对称操作后，标注也要交换位置。
		self.num_joints = 17
    self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
						   
		self.db = [dict(center=[x,y],joints_3d=array([[x,y,0]]), image='foo.jpg', scale=[1,1.5])]
				   
	def _load_coco_keypoint_annotation_kernal(self, index):
		joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
		joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
		for ipt in range(self.num_joints):
			joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
			joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
			joints_3d[ipt, 2] = 0
			t_vis = obj['keypoints'][ipt * 3 + 2]
			if t_vis > 1:
				t_vis = 1
			joints_3d_vis[ipt, 0] = t_vis
			joints_3d_vis[ipt, 1] = t_vis
			joints_3d_vis[ipt, 2] = 0
``` 
`_load_coco_keypoint_annotation_kernal()`, 该代码有些冗余，强行把2d点转换成3d点，引入了joints_3d_vis[3]来描述一维的可见性。可能是为了和openpose保持一致吧。
joints_3d_vis.shape =[num_joints, 3]

#### postprocess

``` python
class JointsDataset(Dataset):
  def __getitem__(self, idx):
        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta
```
注意： 

- 似乎没有使用pafs，没有描述点与点的关系（关节）的方法？
- 虽然没有使用pafs，但是多了target_weight。
- target中没有增加背景层。

### network
网络结构一句话就可以表达清楚，就是一个普通的backbone（用resnet50就很好）加一些转置卷积层（作为一个head network）。作者认为这可能是得到heatmaps的最简单方式，并且包含了从深到浅的特征。
比较simple baselines，hourglass和cpn, 三个网络最大的区别就是在head network（头部网络）是如何得到高分辨率的feature map的，前两个方法都是上采样得到heatmap，但是simple baseline的方法是使用deconv ，deconv相当于同时做了卷积和上采样。


- output: => 17×64×48
- head: conv(17,256,1,1) 
- neck: {ConvTr(256,x,4,4),BN,Relu, ConvTr(256,256,4,4),BN,Relu, ConvTr(256,256,4,4),BN,Relu}
- backbone
    - resnet18
    - resnet50
    - resnet101
    - resnet158
- foot:  `{conv(64,3,7,7),bn1,relu,maxpool}`
- input: 
    - 3×256×192
    - 3×256×256


作者从heatmaps的尺寸，deconv的卷积核尺寸，backbone结构，输入图像尺寸等4个方面分别作了对比：

结论是：heatmaps尺寸最好是64*48，三层deconv，kernel的size最好是4，backbone是越大越好，图像尺寸越大越好，但是后两者会极大增加计算量和显存。要做好精度和速度的平衡。

~~估计人体都是直立的，所以高大宽小的网络更加适配~~


``` python
model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
pose_resnet.get_pose_net(config, is_train=False)
```

### trainer

训练几乎没有任何trick，网络里面也没有任何其他的骚操作，比如各种ohem，ohkm等等，也没有中继监督，相同条件下（主要是输入尺寸和bakcbone）效果都是领先水平。


#### Loss
Loss的设计：就是普通L2 loss，只在最后的输出算loss，并没有中继监督。

```
loss = criterion(output, target, target_weight)
# target_weight 的值 都是0或1.
```

#### optimizer
``` python
def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer
```
## misc

- [1] Simple Baselines for Human Pose Estimation and Tracking(https://arxiv.org/abs/1804.06208)
- [2] https://github.com/Microsoft/human-pose-estimation.pytorch

### rknn
```

done
--> Loading model
done
--> Building model
W The target_platform is not set in config, using default target platform rk1808.
W The channel_mean_value is deprecated. Please use mean_values and std_values to replace it.
E Catch exception when building RKNN model!
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 615, in rknn.api.rknn_base.RKNNBase.build
E   File "rknn\api\rknn_base.py", line 428, in rknn.api.rknn_base.RKNNBase._build
E   File "rknn\base\ovxconfiggenerator.py", line 132, in rknn.base.ovxconfiggenerator.generate_vx_config_from_files
E   File "rknn\base\RKNNlib\RKNNnet.py", line 637, in rknn.base.RKNNlib.RKNNnet.RKNNNet.load_input_meta
E   File "rknn\base\RKNNlib\RKNNnet.py", line 640, in rknn.base.RKNNlib.RKNNnet.RKNNNet.load_input_meta
E   File "rknn\base\RKNNlib\RKNNnet.py", line 373, in rknn.base.RKNNlib.RKNNnet.RKNNNet.update_input_meta
E   File "rknn\base\RKNNlib\net_input_meta.py", line 20, in rknn.base.RKNNlib.net_input_meta.NetInputMeta.__init__
E   File "rknn\base\RKNNlib\net_input_meta.py", line 56, in rknn.base.RKNNlib.net_input_meta.DatabaseMeta.__init__
E AssertionError
Build model failed!
```


```

--> Building model
W The target_platform is not set in config, using default target platform rk1808.
W The channel_mean_value is deprecated. Please use mean_values and std_values to replace it.
E Catch exception when building RKNN model!
E Traceback (most recent call last):
E   File "rknn\api\rknn_base.py", line 615, in rknn.api.rknn_base.RKNNBase.build
E   File "rknn\api\rknn_base.py", line 428, in rknn.api.rknn_base.RKNNBase._build
E   File "rknn\base\ovxconfiggenerator.py", line 132, in rknn.base.ovxconfiggenerator.generate_vx_config_from_files
E   File "rknn\base\RKNNlib\RKNNnet.py", line 637, in rknn.base.RKNNlib.RKNNnet.RKNNNet.load_input_meta
E   File "rknn\base\RKNNlib\RKNNnet.py", line 640, in rknn.base.RKNNlib.RKNNnet.RKNNNet.load_input_meta
E   File "rknn\base\RKNNlib\RKNNnet.py", line 373, in rknn.base.RKNNlib.RKNNnet.RKNNNet.update_input_meta
E   File "rknn\base\RKNNlib\net_input_meta.py", line 20, in rknn.base.RKNNlib.net_input_meta.NetInputMeta.__init__
E   File "rknn\base\RKNNlib\net_input_meta.py", line 56, in rknn.base.RKNNlib.net_input_meta.DatabaseMeta.__init__
E AssertionError
Build model failed!
```

#### layers rename
```


(zal_pytorch120) H:\Project\Github\openpose_misc\Githubs\human-pose-estimation.pytorch>
(zal_pytorch120) H:\Project\Github\openpose_misc\Githubs\human-pose-estimation.pytorch>E:/ProgramData/Anaconda3/envs/zal_pytorch120/python.exe h:/Project/Github/openpose_misc/Githubs/human-pose-estimation.pytorch/pose_estimation/exporter.py
experiments\coco\resnet50\256x192_d256x3_adam_lr1e-3.yaml cfg
h:\Project\Github\openpose_misc\Githubs\human-pose-estimation.pytorch\pose_estimation\..\lib\core\config.py:161: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  exp_config = edict(yaml.load(f))
=> creating output\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3
=> creating log\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3_2021-08-30-11-26
Namespace(cfg='experiments\\coco\\resnet50\\256x192_d256x3_adam_lr1e-3.yaml', coco_bbox_file=None, flip_test=False, frequent=100, gpus=None, model_file='output\\coco\\pose_resnet_50\\256x192_d256x3_adam_lr1e-3\\model_best.pth.tar', post_process=False, shift_heatmap=False, use_detect_bbox=False, workers=None)
{'CUDNN': {'BENCHMARK': True, 'DETERMINISTIC': False, 'ENABLED': True},
 'DATASET': {'DATASET': 'coco',
             'DATA_FORMAT': 'jpg',
             'FLIP': True,
             'HYBRID_JOINTS_TYPE': '',
             'ROOT': 'data/coco/',
             'ROT_FACTOR': 40,
             'SCALE_FACTOR': 0.3,
             'SELECT_DATA': False,
             'TEST_SET': 'val2017',
             'TRAIN_SET': 'train2017'},
 'DATA_DIR': '',
 'DEBUG': {'DEBUG': True,
           'SAVE_BATCH_IMAGES_GT': True,
           'SAVE_BATCH_IMAGES_PRED': True,
           'SAVE_HEATMAPS_GT': True,
           'SAVE_HEATMAPS_PRED': True},
 'GPUS': '0',
 'LOG_DIR': 'log',
 'LOSS': {'USE_TARGET_WEIGHT': True},
 'MODEL': {'EXTRA': {'DECONV_WITH_BIAS': False,
                     'FINAL_CONV_KERNEL': 1,
                     'HEATMAP_SIZE': array([48, 64]),
                     'NUM_DECONV_FILTERS': [256, 256, 256],
                     'NUM_DECONV_KERNELS': [4, 4, 4],
                     'NUM_DECONV_LAYERS': 3,
                     'NUM_LAYERS': 50,
                     'SIGMA': 2,
                     'TARGET_TYPE': 'gaussian'},
           'IMAGE_SIZE': array([192, 256]),
           'INIT_WEIGHTS': True,
           'NAME': 'pose_resnet',
           'NUM_JOINTS': 17,
           'PRETRAINED': 'models/pytorch/imagenet/resnet50-19c8e357.pth',
           'STYLE': 'pytorch'},
 'OUTPUT_DIR': 'output',
 'PRINT_FREQ': 100,
 'TEST': {'BATCH_SIZE': 32,
          'BBOX_THRE': 1.0,
          'COCO_BBOX_FILE': 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
          'FLIP_TEST': False,
          'IMAGE_THRE': 0.0,
          'IN_VIS_THRE': 0.2,
          'MODEL_FILE': 'output\\coco\\pose_resnet_50\\256x192_d256x3_adam_lr1e-3\\model_best.pth.tar',
          'NMS_THRE': 1.0,
          'OKS_THRE': 0.9,
          'POST_PROCESS': True,
          'SHIFT_HEATMAP': True,
          'USE_GT_BBOX': True},
 'TRAIN': {'BATCH_SIZE': 32,
           'BEGIN_EPOCH': 0,
           'CHECKPOINT': '',
           'END_EPOCH': 140,
           'GAMMA1': 0.99,
           'GAMMA2': 0.0,
           'LR': 0.001,
           'LR_FACTOR': 0.1,
           'LR_STEP': [90, 120],
           'MOMENTUM': 0.9,
           'NESTEROV': False,
           'OPTIMIZER': 'adam',
           'RESUME': False,
           'SHUFFLE': True,
           'WD': 0.0001},
 'WORKERS': 4}
=> loading model from output\coco\pose_resnet_50\256x192_d256x3_adam_lr1e-3\model_best.pth.tar
Traceback (most recent call last):
  File "h:/Project/Github/openpose_misc/Githubs/human-pose-estimation.pytorch/pose_estimation/exporter.py", line 242, in
<module>
    main(cmds)
  File "h:/Project/Github/openpose_misc/Githubs/human-pose-estimation.pytorch/pose_estimation/exporter.py", line 218, in
main
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\site-packages\torch\nn\modules\module.py", line 845, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for PoseResNet:
        Missing key(s) in state_dict: "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.0.bn2.running_mean", "layer1.0.bn2.running_var", "layer1.0.conv3.weight", "layer1.0.bn3.weight", "layer1.0.bn3.bias", "layer1.0.bn3.running_mean", "layer1.0.bn3.running_var", "layer1.0.downsample.0.weight", "layer1.0.downsample.1.weight", "layer1.0.downsample.1.bias", "layer1.0.downsample.1.running_mean", "layer1.0.downsample.1.running_var", "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", "layer1.1.bn1.running_var", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", "layer1.1.bn2.running_var", "layer1.1.conv3.weight", "layer1.1.bn3.weight", "layer1.1.bn3.bias", "layer1.1.bn3.running_mean", "layer1.1.bn3.running_var", "layer1.2.conv1.weight", "layer1.2.bn1.weight", "layer1.2.bn1.bias", "layer1.2.bn1.running_mean", "layer1.2.bn1.running_var", "layer1.2.conv2.weight", "layer1.2.bn2.weight", "layer1.2.bn2.bias", "layer1.2.bn2.running_mean", "layer1.2.bn2.running_var", "layer1.2.conv3.weight", "layer1.2.bn3.weight", "layer1.2.bn3.bias", "layer1.2.bn3.running_mean", "layer1.2.bn3.running_var", "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.bn1.running_mean", "layer2.0.bn1.running_var", "layer2.0.conv2.weight",
"layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.bn2.running_mean", "layer2.0.bn2.running_var", "layer2.0.conv3.weight", "layer2.0.bn3.weight", "layer2.0.bn3.bias", "layer2.0.bn3.running_mean", "layer2.0.bn3.running_var", "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight", "layer2.0.downsample.1.bias", "layer2.0.downsample.1.running_mean", "layer2.0.downsample.1.running_var", "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", "layer2.1.bn1.running_var", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", "layer2.1.bn2.running_var", "layer2.1.conv3.weight", "layer2.1.bn3.weight", "layer2.1.bn3.bias", "layer2.1.bn3.running_mean", "layer2.1.bn3.running_var", "layer2.2.conv1.weight", "layer2.2.bn1.weight", "layer2.2.bn1.bias", "layer2.2.bn1.running_mean", "layer2.2.bn1.running_var", "layer2.2.conv2.weight", "layer2.2.bn2.weight", "layer2.2.bn2.bias", "layer2.2.bn2.running_mean", "layer2.2.bn2.running_var", "layer2.2.conv3.weight", "layer2.2.bn3.weight", "layer2.2.bn3.bias", "layer2.2.bn3.running_mean", "layer2.2.bn3.running_var", "layer2.3.conv1.weight", "layer2.3.bn1.weight", "layer2.3.bn1.bias", "layer2.3.bn1.running_mean", "layer2.3.bn1.running_var", "layer2.3.conv2.weight", "layer2.3.bn2.weight", "layer2.3.bn2.bias", "layer2.3.bn2.running_mean", "layer2.3.bn2.running_var", "layer2.3.conv3.weight", "layer2.3.bn3.weight",
"layer2.3.bn3.bias", "layer2.3.bn3.running_mean", "layer2.3.bn3.running_var", "layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.bn1.running_mean", "layer3.0.bn1.running_var", "layer3.0.conv2.weight", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.bn2.running_mean", "layer3.0.bn2.running_var", "layer3.0.conv3.weight", "layer3.0.bn3.weight", "layer3.0.bn3.bias", "layer3.0.bn3.running_mean", "layer3.0.bn3.running_var", "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight", "layer3.0.downsample.1.bias", "layer3.0.downsample.1.running_mean", "layer3.0.downsample.1.running_var", "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", "layer3.1.bn1.running_var", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", "layer3.1.bn2.running_var", "layer3.1.conv3.weight", "layer3.1.bn3.weight", "layer3.1.bn3.bias", "layer3.1.bn3.running_mean", "layer3.1.bn3.running_var", "layer3.2.conv1.weight", "layer3.2.bn1.weight", "layer3.2.bn1.bias", "layer3.2.bn1.running_mean", "layer3.2.bn1.running_var", "layer3.2.conv2.weight", "layer3.2.bn2.weight", "layer3.2.bn2.bias", "layer3.2.bn2.running_mean", "layer3.2.bn2.running_var", "layer3.2.conv3.weight", "layer3.2.bn3.weight", "layer3.2.bn3.bias", "layer3.2.bn3.running_mean", "layer3.2.bn3.running_var", "layer3.3.conv1.weight", "layer3.3.bn1.weight", "layer3.3.bn1.bias", "layer3.3.bn1.running_mean", "layer3.3.bn1.running_var", "layer3.3.conv2.weight", "layer3.3.bn2.weight", "layer3.3.bn2.bias", "layer3.3.bn2.running_mean", "layer3.3.bn2.running_var", "layer3.3.conv3.weight", "layer3.3.bn3.weight", "layer3.3.bn3.bias", "layer3.3.bn3.running_mean", "layer3.3.bn3.running_var", "layer3.4.conv1.weight", "layer3.4.bn1.weight", "layer3.4.bn1.bias", "layer3.4.bn1.running_mean", "layer3.4.bn1.running_var", "layer3.4.conv2.weight", "layer3.4.bn2.weight", "layer3.4.bn2.bias", "layer3.4.bn2.running_mean", "layer3.4.bn2.running_var", "layer3.4.conv3.weight", "layer3.4.bn3.weight", "layer3.4.bn3.bias", "layer3.4.bn3.running_mean", "layer3.4.bn3.running_var", "layer3.5.conv1.weight", "layer3.5.bn1.weight", "layer3.5.bn1.bias", "layer3.5.bn1.running_mean", "layer3.5.bn1.running_var", "layer3.5.conv2.weight", "layer3.5.bn2.weight", "layer3.5.bn2.bias", "layer3.5.bn2.running_mean", "layer3.5.bn2.running_var", "layer3.5.conv3.weight", "layer3.5.bn3.weight", "layer3.5.bn3.bias", "layer3.5.bn3.running_mean", "layer3.5.bn3.running_var", "layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.bn1.running_mean", "layer4.0.bn1.running_var", "layer4.0.conv2.weight", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.bn2.running_mean", "layer4.0.bn2.running_var", "layer4.0.conv3.weight", "layer4.0.bn3.weight", "layer4.0.bn3.bias", "layer4.0.bn3.running_mean", "layer4.0.bn3.running_var", "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight", "layer4.0.downsample.1.bias", "layer4.0.downsample.1.running_mean",
"layer4.0.downsample.1.running_var", "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", "layer4.1.bn1.running_var", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", "layer4.1.bn2.running_var", "layer4.1.conv3.weight", "layer4.1.bn3.weight", "layer4.1.bn3.bias", "layer4.1.bn3.running_mean", "layer4.1.bn3.running_var", "layer4.2.conv1.weight", "layer4.2.bn1.weight", "layer4.2.bn1.bias",
"layer4.2.bn1.running_mean", "layer4.2.bn1.running_var", "layer4.2.conv2.weight", "layer4.2.bn2.weight", "layer4.2.bn2.bias", "layer4.2.bn2.running_mean", "layer4.2.bn2.running_var", "layer4.2.conv3.weight", "layer4.2.bn3.weight", "layer4.2.bn3.bias", "layer4.2.bn3.running_mean", "layer4.2.bn3.running_var", "deconv_layers.0.weight", "deconv_layers.1.weight", "deconv_layers.1.bias", "deconv_layers.1.running_mean", "deconv_layers.1.running_var", "deconv_layers.3.weight", "deconv_layers.4.weight", "deconv_layers.4.bias", "deconv_layers.4.running_mean", "deconv_layers.4.running_var", "deconv_layers.6.weight", "deconv_layers.7.weight", "deconv_layers.7.bias", "deconv_layers.7.running_mean", "deconv_layers.7.running_var",
"final_layer.weight", "final_layer.bias".

        Unexpected key(s) in state_dict: "module.conv1.weight", "module.bn1.weight", "module.bn1.bias", "module.bn1.running_mean", "module.bn1.running_var", "module.bn1.num_batches_tracked", "module.layer1.0.conv1.weight", "module.layer1.0.bn1.weight", "module.layer1.0.bn1.bias", "module.layer1.0.bn1.running_mean", "module.layer1.0.bn1.running_var", "module.layer1.0.bn1.num_batches_tracked", "module.layer1.0.conv2.weight", "module.layer1.0.bn2.weight", "module.layer1.0.bn2.bias", "module.layer1.0.bn2.running_mean", "module.layer1.0.bn2.running_var", "module.layer1.0.bn2.num_batches_tracked", "module.layer1.0.conv3.weight", "module.layer1.0.bn3.weight", "module.layer1.0.bn3.bias", "module.layer1.0.bn3.running_mean", "module.layer1.0.bn3.running_var", "module.layer1.0.bn3.num_batches_tracked", "module.layer1.0.downsample.0.weight", "module.layer1.0.downsample.1.weight", "module.layer1.0.downsample.1.bias", "module.layer1.0.downsample.1.running_mean", "module.layer1.0.downsample.1.running_var", "module.layer1.0.downsample.1.num_batches_tracked", "module.layer1.1.conv1.weight", "module.layer1.1.bn1.weight", "module.layer1.1.bn1.bias", "module.layer1.1.bn1.running_mean", "module.layer1.1.bn1.running_var", "module.layer1.1.bn1.num_batches_tracked", "module.layer1.1.conv2.weight", "module.layer1.1.bn2.weight", "module.layer1.1.bn2.bias", "module.layer1.1.bn2.running_mean", "module.layer1.1.bn2.running_var", "module.layer1.1.bn2.num_batches_tracked", "module.layer1.1.conv3.weight", "module.layer1.1.bn3.weight", "module.layer1.1.bn3.bias", "module.layer1.1.bn3.running_mean", "module.layer1.1.bn3.running_var", "module.layer1.1.bn3.num_batches_tracked", "module.layer1.2.conv1.weight", "module.layer1.2.bn1.weight", "module.layer1.2.bn1.bias", "module.layer1.2.bn1.running_mean", "module.layer1.2.bn1.running_var", "module.layer1.2.bn1.num_batches_tracked", "module.layer1.2.conv2.weight", "module.layer1.2.bn2.weight", "module.layer1.2.bn2.bias", "module.layer1.2.bn2.running_mean", "module.layer1.2.bn2.running_var", "module.layer1.2.bn2.num_batches_tracked", "module.layer1.2.conv3.weight", "module.layer1.2.bn3.weight", "module.layer1.2.bn3.bias", "module.layer1.2.bn3.running_mean", "module.layer1.2.bn3.running_var", "module.layer1.2.bn3.num_batches_tracked", "module.layer2.0.conv1.weight", "module.layer2.0.bn1.weight", "module.layer2.0.bn1.bias", "module.layer2.0.bn1.running_mean", "module.layer2.0.bn1.running_var", "module.layer2.0.bn1.num_batches_tracked", "module.layer2.0.conv2.weight", "module.layer2.0.bn2.weight", "module.layer2.0.bn2.bias", "module.layer2.0.bn2.running_mean", "module.layer2.0.bn2.running_var", "module.layer2.0.bn2.num_batches_tracked", "module.layer2.0.conv3.weight", "module.layer2.0.bn3.weight", "module.layer2.0.bn3.bias", "module.layer2.0.bn3.running_mean", "module.layer2.0.bn3.running_var", "module.layer2.0.bn3.num_batches_tracked", "module.layer2.0.downsample.0.weight", "module.layer2.0.downsample.1.weight", "module.layer2.0.downsample.1.bias", "module.layer2.0.downsample.1.running_mean", "module.layer2.0.downsample.1.running_var", "module.layer2.0.downsample.1.num_batches_tracked", "module.layer2.1.conv1.weight", "module.layer2.1.bn1.weight", "module.layer2.1.bn1.bias", "module.layer2.1.bn1.running_mean", "module.layer2.1.bn1.running_var", "module.layer2.1.bn1.num_batches_tracked", "module.layer2.1.conv2.weight", "module.layer2.1.bn2.weight", "module.layer2.1.bn2.bias", "module.layer2.1.bn2.running_mean", "module.layer2.1.bn2.running_var", "module.layer2.1.bn2.num_batches_tracked", "module.layer2.1.conv3.weight", "module.layer2.1.bn3.weight", "module.layer2.1.bn3.bias", "module.layer2.1.bn3.running_mean", "module.layer2.1.bn3.running_var", "module.layer2.1.bn3.num_batches_tracked", "module.layer2.2.conv1.weight", "module.layer2.2.bn1.weight", "module.layer2.2.bn1.bias", "module.layer2.2.bn1.running_mean", "module.layer2.2.bn1.running_var", "module.layer2.2.bn1.num_batches_tracked", "module.layer2.2.conv2.weight", "module.layer2.2.bn2.weight", "module.layer2.2.bn2.bias", "module.layer2.2.bn2.running_mean", "module.layer2.2.bn2.running_var", "module.layer2.2.bn2.num_batches_tracked", "module.layer2.2.conv3.weight", "module.layer2.2.bn3.weight", "module.layer2.2.bn3.bias", "module.layer2.2.bn3.running_mean", "module.layer2.2.bn3.running_var", "module.layer2.2.bn3.num_batches_tracked", "module.layer2.3.conv1.weight", "module.layer2.3.bn1.weight", "module.layer2.3.bn1.bias", "module.layer2.3.bn1.running_mean", "module.layer2.3.bn1.running_var", "module.layer2.3.bn1.num_batches_tracked", "module.layer2.3.conv2.weight", "module.layer2.3.bn2.weight", "module.layer2.3.bn2.bias", "module.layer2.3.bn2.running_mean", "module.layer2.3.bn2.running_var", "module.layer2.3.bn2.num_batches_tracked", "module.layer2.3.conv3.weight", "module.layer2.3.bn3.weight", "module.layer2.3.bn3.bias", "module.layer2.3.bn3.running_mean", "module.layer2.3.bn3.running_var", "module.layer2.3.bn3.num_batches_tracked", "module.layer3.0.conv1.weight", "module.layer3.0.bn1.weight", "module.layer3.0.bn1.bias", "module.layer3.0.bn1.running_mean", "module.layer3.0.bn1.running_var", "module.layer3.0.bn1.num_batches_tracked", "module.layer3.0.conv2.weight", "module.layer3.0.bn2.weight", "module.layer3.0.bn2.bias", "module.layer3.0.bn2.running_mean", "module.layer3.0.bn2.running_var", "module.layer3.0.bn2.num_batches_tracked", "module.layer3.0.conv3.weight", "module.layer3.0.bn3.weight", "module.layer3.0.bn3.bias", "module.layer3.0.bn3.running_mean", "module.layer3.0.bn3.running_var", "module.layer3.0.bn3.num_batches_tracked", "module.layer3.0.downsample.0.weight", "module.layer3.0.downsample.1.weight", "module.layer3.0.downsample.1.bias", "module.layer3.0.downsample.1.running_mean", "module.layer3.0.downsample.1.running_var", "module.layer3.0.downsample.1.num_batches_tracked", "module.layer3.1.conv1.weight", "module.layer3.1.bn1.weight", "module.layer3.1.bn1.bias", "module.layer3.1.bn1.running_mean", "module.layer3.1.bn1.running_var", "module.layer3.1.bn1.num_batches_tracked", "module.layer3.1.conv2.weight", "module.layer3.1.bn2.weight", "module.layer3.1.bn2.bias", "module.layer3.1.bn2.running_mean", "module.layer3.1.bn2.running_var", "module.layer3.1.bn2.num_batches_tracked", "module.layer3.1.conv3.weight", "module.layer3.1.bn3.weight", "module.layer3.1.bn3.bias", "module.layer3.1.bn3.running_mean", "module.layer3.1.bn3.running_var", "module.layer3.1.bn3.num_batches_tracked", "module.layer3.2.conv1.weight", "module.layer3.2.bn1.weight", "module.layer3.2.bn1.bias", "module.layer3.2.bn1.running_mean", "module.layer3.2.bn1.running_var", "module.layer3.2.bn1.num_batches_tracked", "module.layer3.2.conv2.weight", "module.layer3.2.bn2.weight", "module.layer3.2.bn2.bias", "module.layer3.2.bn2.running_mean", "module.layer3.2.bn2.running_var", "module.layer3.2.bn2.num_batches_tracked", "module.layer3.2.conv3.weight", "module.layer3.2.bn3.weight", "module.layer3.2.bn3.bias", "module.layer3.2.bn3.running_mean", "module.layer3.2.bn3.running_var", "module.layer3.2.bn3.num_batches_tracked", "module.layer3.3.conv1.weight", "module.layer3.3.bn1.weight", "module.layer3.3.bn1.bias", "module.layer3.3.bn1.running_mean", "module.layer3.3.bn1.running_var", "module.layer3.3.bn1.num_batches_tracked", "module.layer3.3.conv2.weight", "module.layer3.3.bn2.weight", "module.layer3.3.bn2.bias", "module.layer3.3.bn2.running_mean", "module.layer3.3.bn2.running_var", "module.layer3.3.bn2.num_batches_tracked", "module.layer3.3.conv3.weight", "module.layer3.3.bn3.weight", "module.layer3.3.bn3.bias", "module.layer3.3.bn3.running_mean", "module.layer3.3.bn3.running_var", "module.layer3.3.bn3.num_batches_tracked", "module.layer3.4.conv1.weight", "module.layer3.4.bn1.weight", "module.layer3.4.bn1.bias", "module.layer3.4.bn1.running_mean", "module.layer3.4.bn1.running_var", "module.layer3.4.bn1.num_batches_tracked", "module.layer3.4.conv2.weight", "module.layer3.4.bn2.weight", "module.layer3.4.bn2.bias", "module.layer3.4.bn2.running_mean", "module.layer3.4.bn2.running_var", "module.layer3.4.bn2.num_batches_tracked", "module.layer3.4.conv3.weight", "module.layer3.4.bn3.weight", "module.layer3.4.bn3.bias", "module.layer3.4.bn3.running_mean", "module.layer3.4.bn3.running_var", "module.layer3.4.bn3.num_batches_tracked", "module.layer3.5.conv1.weight", "module.layer3.5.bn1.weight", "module.layer3.5.bn1.bias", "module.layer3.5.bn1.running_mean", "module.layer3.5.bn1.running_var", "module.layer3.5.bn1.num_batches_tracked", "module.layer3.5.conv2.weight", "module.layer3.5.bn2.weight", "module.layer3.5.bn2.bias", "module.layer3.5.bn2.running_mean", "module.layer3.5.bn2.running_var", "module.layer3.5.bn2.num_batches_tracked", "module.layer3.5.conv3.weight", "module.layer3.5.bn3.weight", "module.layer3.5.bn3.bias", "module.layer3.5.bn3.running_mean", "module.layer3.5.bn3.running_var", "module.layer3.5.bn3.num_batches_tracked", "module.layer4.0.conv1.weight", "module.layer4.0.bn1.weight", "module.layer4.0.bn1.bias", "module.layer4.0.bn1.running_mean", "module.layer4.0.bn1.running_var", "module.layer4.0.bn1.num_batches_tracked", "module.layer4.0.conv2.weight", "module.layer4.0.bn2.weight", "module.layer4.0.bn2.bias", "module.layer4.0.bn2.running_mean", "module.layer4.0.bn2.running_var", "module.layer4.0.bn2.num_batches_tracked", "module.layer4.0.conv3.weight", "module.layer4.0.bn3.weight", "module.layer4.0.bn3.bias", "module.layer4.0.bn3.running_mean", "module.layer4.0.bn3.running_var", "module.layer4.0.bn3.num_batches_tracked", "module.layer4.0.downsample.0.weight", "module.layer4.0.downsample.1.weight", "module.layer4.0.downsample.1.bias", "module.layer4.0.downsample.1.running_mean", "module.layer4.0.downsample.1.running_var", "module.layer4.0.downsample.1.num_batches_tracked", "module.layer4.1.conv1.weight", "module.layer4.1.bn1.weight", "module.layer4.1.bn1.bias", "module.layer4.1.bn1.running_mean", "module.layer4.1.bn1.running_var", "module.layer4.1.bn1.num_batches_tracked", "module.layer4.1.conv2.weight", "module.layer4.1.bn2.weight", "module.layer4.1.bn2.bias", "module.layer4.1.bn2.running_mean", "module.layer4.1.bn2.running_var", "module.layer4.1.bn2.num_batches_tracked", "module.layer4.1.conv3.weight", "module.layer4.1.bn3.weight", "module.layer4.1.bn3.bias", "module.layer4.1.bn3.running_mean", "module.layer4.1.bn3.running_var", "module.layer4.1.bn3.num_batches_tracked", "module.layer4.2.conv1.weight", "module.layer4.2.bn1.weight", "module.layer4.2.bn1.bias", "module.layer4.2.bn1.running_mean", "module.layer4.2.bn1.running_var", "module.layer4.2.bn1.num_batches_tracked", "module.layer4.2.conv2.weight", "module.layer4.2.bn2.weight", "module.layer4.2.bn2.bias", "module.layer4.2.bn2.running_mean", "module.layer4.2.bn2.running_var", "module.layer4.2.bn2.num_batches_tracked", "module.layer4.2.conv3.weight", "module.layer4.2.bn3.weight", "module.layer4.2.bn3.bias", "module.layer4.2.bn3.running_mean", "module.layer4.2.bn3.running_var", "module.layer4.2.bn3.num_batches_tracked", "module.deconv_layers.0.weight", "module.deconv_layers.1.weight", "module.deconv_layers.1.bias", "module.deconv_layers.1.running_mean", "module.deconv_layers.1.running_var", "module.deconv_layers.1.num_batches_tracked", "module.deconv_layers.3.weight", "module.deconv_layers.4.weight", "module.deconv_layers.4.bias", "module.deconv_layers.4.running_mean", "module.deconv_layers.4.running_var", "module.deconv_layers.4.num_batches_tracked", "module.deconv_layers.6.weight", "module.deconv_layers.7.weight", "module.deconv_layers.7.bias", "module.deconv_layers.7.running_mean", "module.deconv_layers.7.running_var", "module.deconv_layers.7.num_batches_tracked", "module.final_layer.weight", "module.final_layer.bias".
```

#### nms
```
(zal_pytorch120) H:\Project\Github\openpose_misc\Githubs\human-pose-estimation.pytorch\lib\nms>python setup.py build_ext --inplace
running build_ext
Traceback (most recent call last):
  File "setup.py", line 139, in <module>
    cmdclass={'build_ext': custom_build_ext},
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\site-packages\setuptools\__init__.py", line 144, in setup
    return distutils.core.setup(**attrs)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\distutils\core.py", line 148, in setup
    dist.run_commands()
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\distutils\dist.py", line 955, in run_commands
    self.run_command(cmd)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\distutils\dist.py", line 974, in run_command
    cmd_obj.run()
  File "E:\ProgramData\Anaconda3\envs\zal_pytorch120\lib\site-packages\Cython\Distutils\old_build_ext.py", line 186, in runch120\lib\site-packages\Cython\Distutils\old_build_ext.py", line 186, in run                  ch120\lib\distutils\command\build_ext.py", line 339, in run
    _build_ext.build_ext.run(self)
  File "E:\ProgramData\Anaconda3\envs\zal_pytorsch120\lib\distutils\command\build_ext.py", line 339, in run                                   er_for_nvcc
    self.build_extensions()
  File "setup.py", line 105, in build_extensions
    customize_compiler_for_nvcc(self.compiler)
  File "setup.py", line 78, in customize_compiler_for_nvcc
    default_compiler_so = self.compiler_so
AttributeError: 'MSVCCompiler' object has no attribute 'compiler_so'
```