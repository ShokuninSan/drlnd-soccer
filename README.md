[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Deep Reinforcement Learning Nanodegree Challenge: Play Soccer

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Soccer][image2]
In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, we'll need to download a new Unity environment.

### Project Setup

1. You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

2. Place the file in the `environments/` folder, and unzip (or decompress) the file. 

3. [Optional] Create a Conda environment and activate it
```
(base) ➜  drlnd-soccer git:(master) ✗ conda create --name drlnd-soccer python=3.6
(base) ➜  drlnd-soccer git:(master) ✗ conda activate drlnd-soccer
```

4. Change into the `python` folder and execute `pip install .` to install the required dependencies.

5. Create a custom IPython kernel by executing `$ python -m ipykernel install --user --name drlnd --display-name "drlnd"`

### Getting Started

Start a `jupyter notebook` from within the project folder and follow the instructions in `notebooks/Soccer.ipynb` to either
* train your own agent or
* load the model weights and watch the pre-trained agent

__HINT__: make sure to switch from the default Python 3 kernel to "drlnd" (see section Project Setup).

---

Tested on macOS Big Sur (Version 11.0.1) and Ubuntu 20.04.2 LTS.
