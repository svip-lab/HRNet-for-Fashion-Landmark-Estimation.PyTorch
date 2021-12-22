
# HRNet for Fashion Landmark Estimation
(Modified from [deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch))

## Introduction
This code applies the HRNet ([*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212)) onto fashion landmark estimation task using the [*DeepFashion2*](https://github.com/switchablenorms/DeepFashion2) dataset. HRNet **maintains high-resolution representations** throughout the forward path. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise.

![Illustrating the architecture of the proposed HRNet](/figures/fashion-landmark-estimation.png)

Please note that every image in *DeepFashion2* contains multiple fashion items, while our model assumes that there exists only one item in each image. Therefore, what we feed into the HRNet is not the original image but the cropped ones provided by a detector. In experiments, one can either use the ground truth bounding box annotation to generate the input data or use the output of a detecter (you can try [this](https://github.com/ShenhanQian/KGDet) clothing detector).

## Main Results
### Landmark Estimation Performance on DeepFashion2 Test set
We won **the third place** in the "DeepFashion2 Challenge 2020 - Track 1 Clothes Landmark Estimation" competition.
![DeepFashion2 Challenge 2020 - Track 1 Clothes Landmark Estimation](/figures/DF2_Challenge_Result.png)

### Landmark Estimation Performance on DeepFashion2 Validation Set
| Arch       | BBox Source | AP   | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pose_hrnet |Detector | 0.579 | 0.793 | 0.658 | 0.460 | 0.581 | 0.706 | 0.939 | 0.784 | 0.548 | 0.708 |
| pose_hrnet | GT      |0.702 | 0.956 | 0.801 | 0.579 | 0.703 | 0.740 | 0.965 | 0.827 | 0.592 | 0.741 |

## Quick start
### Installation
1. Install pytorch >= v1.2 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should follow the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as `${POSE_ROOT}`.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   |-- lib
   |-- tools 
   |-- experiments
   |-- models
   |-- data
   |-- log
   |-- output
   |-- README.md
   `-- requirements.txt
   ```

7. Download pretrained models from our [Onedrive Cloud Storage](#OneDrive-Cloud-Storage)

### Data preparation
Our experiments were conducted on [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), clone this repo, and we'll call the directory that you cloned as `${DF2_ROOT}`.
#### 1) Download the dataset
Extract the dataset under `${POSE_ROOT}/data`.
#### 2) Convert annotations into coco-type
The above code repo provides a script to convert annotations into *coco-type*. 

We uploaded our converted annotation file onto [OneDrive](#OneDrive-Cloud-Storage) named as `train/val-coco_style.json`. We also made truncated json files such as `train-coco_style-32.json` meaning the first 32 samples in the dataset to save the loading time during development period.


#### 3) Install the deepfashion_api
Enter `${DF2_ROOT}/deepfashion2_api/PythonAPI` and run
   ```
   python setup.py install
   ```
Note that the `deepfashion2_api` is modified from the `cocoapi` without changing the package name. Therefore, conflicts occur if you try to install this package when you have installed the original `cocoapi` in your computer. We provide two feasible solutions: 1) run our code in a `virtualenv` 2) use the `deepfashion2_api` as a local pacakge. Also note that `deepfashion2_api` is different with `cocoapi` mainly in the number of classes and the values of standard variations for keypoints.

At last the directory should look like this:

```
${POSE_ROOT}
|-- data
`-- |-- deepfashion2
    `-- |-- train
        |   |-- image
        |   |-- annos                           (raw annotation)
        |   |-- train-coco_style.json           (converted annotation file)
        |   `-- train-coco_style-32.json      (truncated for fast debugging)
        |-- validation
        |   |-- image
        |   |-- annos                           (raw annotation)
        |   |-- val-coco_style.json             (converted annotation file)
        |   `-- val-coco_style-64.json        (truncated for fast debugging)
        `-- json_for_test
            `-- keypoints_test_information.json
```

### Training and Testing

Note that the `GPUS` parameter in the `yaml` config file is deprecated. To select GPUs, use the environment varaible:

```bash
 export CUDA_VISIBLE_DEVICES=1
```

**Testing** on DeepFashion2 dataset with **BBox from ground truth** using trained models:
```bash
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    TEST.USE_GT_BBOX True
```

**Testing** on DeepFashion2 dataset with **BBox from a detector** using trained models:
```bash
python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    TEST.DEEPFASHION2_BBOX_FILE data/bbox_result_val.pkl \
```

**Training** on DeepFashion2 dataset using pretrained models:
```bash
python tools/train.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
     MODEL.PRETRAINED models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth
```

Other options

```bash
python tools/test.py \
    ... \
    DATASET.MINI_DATASET True \ # use a subset of the annotation to save loading time
    TAG 'experiment description' \ # this info will appear in the output directory name
    WORKERS 4 \ # num_of_worker for the dataloader
    TEST.BATCH_SIZE_PER_GPU 8 \
    TRAIN.BATCH_SIZE_PER_GPU 8 \
```

## OneDrive Cloud Storage
[OneDrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianshh_shanghaitech_edu_cn/Eo1g551GvWpHtrXxdeYptH4BGUqWCI81fbT1prL93e0z2Q?e=cj6phH)

We provide the following files:
- Model checkpoint files
- Converted annotation files in *coco-type*
- Bounding box results from our self-implemented detector in a pickle file.

```
hrnet-for-fashion-landmark-estimation.pytorch
|-- models
|   `-- pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth
|
|-- data
|   |-- bbox_result_val.pkl
|   |
`-- |-- deepfashion2
    `---|-- train
        |   |-- train-coco_style.json           (converted annotation file)
        |   `-- train-coco_style-32.json      (truncated for fast debugging)
        `-- validation
            |-- val-coco_style.json             (converted annotation file)
            `-- val-coco_style-64.json        (truncated for fast debugging)
        
```

## Discussion

### Experiment Configuration

- For the regression target of keypoint heatmaps, we tuned the standard deviation value `sigma` and finally set it to 2.
- During training, we found that the data augmentation from the original code was too intensive which makes the training process unstable. We weakened the augmentation parameters and observed performance gain.
- Due to the imbalance of classes in *DeepFashion2* dataset, the model's performance on different classes varies a lot. Therefore, we adopted a weighted sampling strategy rather than the naive random shuffling strategy, and observed performance gain.
- We expermented with the value of `weight decay`, and found that either `1e-4` or `1e-5` harms the performance. Therefore, we simply set `weight decay` to 0.
