# Robust Dense Registration

Implementation of the paper with datasets and examples used in the original paper.
```
Title: Robust point-cloud registration based on dense point matching and probabilistic modeling
Journal: The Visual Computer
DOI: 10.1007/s00371-022-02525-y
```

### Requirements
To create a virtual environment and install the required dependences please run:
```shell
conda create -n RDR python=3.8.8
conda activate RDR
python3 -m pip install -r requirements.txt
```
in your working folder.

### Furthest point sampling

Compile the furthest point sampling operation for PyTorch. This is only used by a downsample/upsample function adapted from BCPD to torch with batches. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).

```shell
cd models/lib
python3 setup.py install
cd ../../
```

### Pretrained models

In this example we have the models for ModelNet40, RMANet(Self) and Custom(Self).

Download the models from https://drive.google.com/drive/folders/102jZoSkndDzJZRXSkz1-FPmpyCaZX5Bn?usp=sharing and extract inside the `pretrained`.

### RMANet Training Dataset

Download the dataset from https://drive.google.com/drive/folders/102jZoSkndDzJZRXSkz1-FPmpyCaZX5Bn?usp=sharing and extract inside `datasets`.

- rmanet.tar.gz
    - train_rma.bin - Supervised dataset
    - train_rmanet_def.bin - Self-learning dataset.

The evaluation set is the same provided by [RMANet](https://github.com/WanquanF/RMA-Net) and it should also be saved in the same directory.

If different folders are desired, please set the environment variables, `TRAINING_SET`, `VALIDATION_SET`, and `EVALUATION_SET`, with to the respective directories.

### Custom Dataset

Download the dataset from https://drive.google.com/drive/folders/102jZoSkndDzJZRXSkz1-FPmpyCaZX5Bn?usp=sharing and extract inside `datasets` the desired dataset.
- custom.tar.gz - Supervised dataset
- custom_self.tar.gz - Self-supervised dataset, the evaluation set is shared with supervised.

If different folders are desired, please set the environment variables, `TRAINING_SET`, `VALIDATION_SET`, and `EVALUATION_SET`, with to the respective directories.

### Training

#### Train RMANet dataset

```
TRAINING_SET=../deform/data/ python3 train.py configs/rmanet/experiment_nonrigid/config_train.yaml
```

#### Train Custom dataset

```
python3 train.py configs/custom/experiment_nonrigid/config_train.yaml
``` 

#### Train ModelNet40 dataset

```
python3 train.py configs/modelnet40/experiment_rigid/config_train.yaml
```

### Evaluating

#### Evaluate RMANet dataset

Example of evaluation using RMANet dataset.

```
EVALUATION_SET=$HOME/Downloads/  python3 evaluate_nonrigid.py configs/rmanet/experiment_nonrigid/config_evaluate.yaml
```

Update `NOISE_TYPE` at `config_evaluate.yaml` to the other cases (`outlier`, `holes`, `clean`).

#### Evaluate Custom dataset

Example of evaluation using Custom dataset.

```
python3 evaluate_nonrigid.py configs/custom/experiment_nonrigid/config_evaluate.yaml
```

Update `EVAL` at `CYCLE` or `LOOP` to use them (details in the paper).

#### Evaluate Rigid

```
python3 evaluate_rigid.py configs/modelnet40/experiment_rigid/config_evaluate.yaml
```

### Visualize Pair Registration

All experiments used `beta=0.5` when `NOISE_TYPE=clean`. 

``` shell
# To reproduce paper results for cat1, dog3, person2, person1

python3 visualize_nonrigid.py configs/rmanet/experiment_pair/config_evaluate.yaml

visualize_nonrigid.py configs/rmanet/experiment_pair/config_evaluate2.yaml

python3 visualize_nonrigid.py configs/rmanet/experiment_pair/config_evaluate3.yaml

python3 visualize_nonrigid.py configs/rmanet/experiment_pair/config_evaluate4.yaml

# To reproduce paper results for table1, armadillo1
python3 visualize_nonrigid.py configs/custom/experiment_pair/config_evaluate.yaml

python3 visualize_nonrigid.py configs/custom/experiment_pair/config_evaluate2.yaml

### Referenced Repositories

We use code from the following repositories:
- RGM - https://github.com/fukexue/RGM
- SuperGlue - https://github.com/magicleap/SuperGluePretrainedNetwork
- DCP - https://github.com/WangYueFt/dcp
- RMANet - https://github.com/WanquanF/RMA-Net
- BCPD - https://github.com/ohirose/bcpd
- LOFTR - https://github.com/zju3dv/LoFTR
