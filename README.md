# dance

:construction: This repo is ***WIP***.

|![](./assets/pipeline.png)|![](assets/demo.gif)|
|:---:|:---:|
|*DANCE's Pipeline*| *Illustration* |

## Get started
1. Prepare the environment (the scripts are just examples)
   - gcc & g++ ≥ 5
   - Python 3.6.8 (developed & tested on this version)
     - `conda create --name dance python==3.6.8`
   - PyTorch 1.5.1 with CUDA 10.1
     - `conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch`
2. Clone this project and install sub-module dependency
   - `git clone --recurse-submodules https://github.com/lkevinzc/dance`
   - `cd dance && python -m pip install -e detectron2`