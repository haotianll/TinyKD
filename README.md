# TinyKD

## Environment

```
conda create -n tinykd python=3.7 -y
conda activate tinykd

conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

cd TinyKD/
pip install -r requirements.txt
pip install -v -e .
```

## Training

- TinyPerson (batch size: 4)

```shell
# teacher
bash tools/dist_train.sh configs_tiny/tinyperson/faster_rcnn_hr48_fpn_1x.py 2
# student
bash tools/dist_train.sh configs_tiny/tinyperson/faster_rcnn_r50_fpn_1x.py 2
# student + TinyKD
bash tools/dist_train.sh configs_tiny/tinyperson_kd/faster_rcnn_r50_fpn_1x_tea_hr48_kd.py 2
```

- AI-TOD (batch size: 2)

```shell
# teacher
bash tools/dist_train.sh configs_tiny/aitod/aitod_faster_rcnn_hr48_1x.py 1
# student
bash tools/dist_train.sh configs_tiny/aitod/aitod_faster_rcnn_r50_1x.py 1
# student + TinyKD
bash tools/dist_train.sh configs_tiny/aitod_kd/aitod_faster_r50_1x_tea_hr48_kd.py 1
```

## Results and models

**(TODO)** The pre-trained models will be publicly available in the future.

<!--
## Citation
If you find this code useful in your research, please consider citing:

```latex
```
-->
