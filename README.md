---
title: Real ESRGAN
app_file: app.py
cuda compatibility: check_cuda.py
inference: inference_realesrgan.py
pinned: false
---



> ### "requirements.txt" installs cpu version of pytorch(torch).
In order to install cuda version of pytorch, 
enter below after making sure your conda environment is updated:

First, activate the conda environment, then:
```
conda install anaconda
```
```
conda update --all
```

Second, enter the  below command to install the tested version of pytorch (to avoid problems)
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
The installing process may take a long time.
