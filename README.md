C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion
=====================================================================
By: _David Novotny, Nikhila Ravi, Benjamin Graham, Natalia Neverova, Andrea Vedaldi_

This is the official implementation of **C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion** in PyTorch.

[Link to paper](https://arxiv.org/abs/1909.02533) | 
[Project page](https://research.fb.com/publications/c3dpo-canonical-3d-pose-networks-for-non-rigid-structure-from-motion/)

![alt text](./splash_video.gif "splash")

Dependencies
------------

This is a Python 3.6 package. Required packages can be installed with e.g. `pip` and `conda`:
```
> conda create -n c3dpo python=3.6
> pip install -r requirements.txt
```

The complete list of dependencies:
- pytorch (version==1.1.0)
- numpy
- tqdm
- matplotlib
- visdom
- pyyaml
- tabulate

Demo
----

`demo.py` downloads and runs a pre-trained C3DPO model on a sample skeleton from the Human36m dataset and generates a 3D figure with a video of the predicted 3D skeleton:
```
> python ./demo.py
```
Note that all the outputs are dumped to a local `Visdom` server. You can start a Visdom server with:
```
> python -m visdom.server
```
Images are also stored to the `./data` directory. The video will get exported only if there's a functioning `ffmpeg` callable from the command line.


Downloading data / models
-------------------------

Whenever needed, all datasets / pre-trained models are automatically downloaded to various folders under the `./data` directory. Hence, there's no need to bother with a complicated data setup :). In case you would like to cache all the datasets for your own use, simply run the `evaluate.py` which downloads all the needed data during its run.


Quick start = pre-trained network evaluation
--------------------------------------------

Pre-trained networks can be evaluated by calling `evaluate.py`:
```
> python evaluate.py
```
Note that we provide pre-trained models that will get auto-downloaded during the run of the script to the `./data/exps/` directory.
Furthermore, the datasets will also be automatically downloaded in case they are not stored in `./data/datasets/`.


Network training + evaluation
-----------------------------

Launch `experiment.py` with the argument `cfg_file` set to the yaml file corresponding the relevant dataset., e.g.:
```
> python ./experiment.py --cfg_file ./cfgs/h36m.yaml
```
will train a C3DPO model for the Human3.6m dataset.

Note that the code supports visualisation in `Visdom`. In order to enable Visdom visualisations, first start a visdom server with:
```
> python -m visdom.server
```
The experiment will output learning curves as well as visualisations of the intermediate outputs to the visdom server.

Furthermore, the results of the evaluation will be periodically updated after every training epoch in `./data/exps/c3dpo/<dataset_name>/eval_results.json`. The metrics reported in the paper correspond to 'EVAL_MPJPE_best' and 'EVAL_stress'.

For the list of all possible yaml config files, please see the `./cfgs/` directory. Each config `.yaml` file corresponds to a training on a different dataset (matching the name of the `.yaml` file). Expected quantitative results are the same as for the `evaluate.py` script.


Reference
---------

If you find our work useful, please cite it using the following bibtex reference.
```
@inproceedings{novotny2019c3dpo,
  title={C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion},
  author={Novotny, David and Ravi, Nikhila and Graham, Benjamin and Neverova, Natalia and Vedaldi, Andrea},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```


License
-------

C3DPO is distributed under the MIT license, as found in the LICENSE file.


Expected outputs of `evaluate.py`
----------------------------

Below are the results of the supplied pre-trained models for all datasets:

```
dataset               MPJPE      Stress
--------------  -----------  ----------
h36m             95.6338     41.5864
h36m_hourglass  145.021      84.693
pascal3d_hrnet   56.8909     40.1775
pascal3d         36.6413     31.0768
up3d_79kp         0.0672771   0.0406902
```

Note that the models have better performance than published mainly due to letting the models to train for longer.


Notes for reproducibility
-------------------------

Note that the performance reported above was obtained with PyTorch v1.1. If you notice differences in performance make sure to use PyTorch v1.1.
