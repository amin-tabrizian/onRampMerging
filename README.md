# Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay

<!-- Original PyTorch implementation of **TACO** from

[TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning](https://arxiv.org/pdf/2306.13229.pdf) by -->

<!-- [Ruijie Zheng](https://ruijiezheng.com), [Xiyao Wang](https://si0wang.github.io)\*, [Yanchao Sun](https://ycsun2017.github.io)\*, [Shuang Ma](https://www.shuangma.me)\*, [Jieyu Zhao](https://jyzhao.net)\*, [Huazhe Xu](http://hxu.rocks)\*, [Hal Daumé III](http://users.umiacs.umd.edu/~hal/)\*, [Furong Huang](https://furong-huang.com)\* -->


<!-- <p align="center">
  <br><img src='media/dmc.gif' width="500"/><br>
   <a href="https://arxiv.org/pdf/2306.13229.pdf">[Paper]</a>&emsp;<a href="https://ruijiezheng.com/project/TACO/index.html">[Website]</a>
</p> -->


## Method

**LISA** is a novel autonomous driving agent designed for safe on-ramp merging in mixed traffic conditions, achieving a 99.67% success rate. It uses latent-state inference to model unobservable factors, like other drivers' intents, enabling the agent to adapt to dynamic environments. An augmented version,**A-LISA**, addresses observation delays, maintaining an impressive 99.37% success rate even with 1-second vehicle-to-vehicle communication delays.

<p align="center">
  <img src='media/policyOptimization.png' width="750"/>
</p>


<!-- ## Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@inproceedings{
zheng2023taco,
title={\${\textbackslash}texttt\{{TACO}\}\$: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning},
author={Ruijie Zheng and Xiyao Wang and Yanchao Sun and Shuang Ma and Jieyu Zhao and Huazhe Xu and Hal Daumé III and Furong Huang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=ezCsMOy1w9}
}

``` -->

## Instructions (macOS)
Assuming that you already have [SUMO](https://sumo.dlr.de/docs/Installing/index.html) installed, install dependencies using `pip3`:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt 
```
After installing dependencies, you can train **LISA** agent by calling:
```
CUDA_VISIBLE_DEVICES=X python3 train.py mode=${TRAINING_MODE} 
```
List of training modes are: `SLSC` for full agent, Plain for LISA without SL and SC netwroks, `SLSCD` for A-LISA, and `PD` for observing LISA's behavior in a delayed environment. There is also another agent for comparison called `Baseline`, for that you need to run `train_baseline.py`.
To test a trained agent, you can call:
```
pyton3 test.py mode=${TEST_MODE} random_seed=${TRAINED_RANDOM_SEED}
```

By providing the random seed, the corrosponding networks will be loaded. For testing the agents behavior on a delayed case, you have to run `test_delay.py`




 <!-- Assuming that you already have [MuJoCo](http://www.mujoco.org) installed, install dependencies using `conda`:

```
conda env create -f environment.yml
conda activate taco
```

After installing dependencies, you can train a **TACO** agent by calling (using quadruped_run as an example):

```
CUDA_VISIBLE_DEVICES=X python train.py agent=taco task=quadruped_run exp_name=${EXP_NAME} 
```

To train a **DrQ-v2** agent:
```
CUDA_VISIBLE_DEVICES=X python train.py agent=drqv2 task=quadruped_run exp_name=${EXP_NAME} 
```

Evaluation videos and model weights can be saved with arguments `save_video=True` and `save_model=True`. Refer to the `cfgs` directory for a full list of options and default hyperparameters. --> -->


## Acknowledgement
<!-- TACO is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the [DrQv2](https://github.com/facebookresearch/drqv2) codebase. Our implementation builds on top of their repository. -->


