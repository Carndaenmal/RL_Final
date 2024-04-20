Description
-------------------------------------------------------

Generative modelling is focused on matching a data distribution and reinforcement learning is focused on finding the optimal policy by maximizing a reward for successful tasks. 
We explore the application of GFlowNets on learning acyclic causal structural models with approaches more oriented to reinforcement learning strategies. Especially from [DynGFN](https://github.com/lazaratan/dyn-gfn) and an [entropy regularization term](https://arxiv.org/abs/1602.01783) to the detailed balance loss. We found this term increases performance on a range of metrics used in DynGFN by increasing exploration.

Installation
------------------------------------------------------
```python

conda create -n myenv python=3.8
conda activate myenv

pip install -r requirements.txt

```
Training
------------------------------------------------------
```
python train.py seed=7 experiment="online_updates.yaml" trainer="gpu"
```
Training on a Cluster
------------------------------------------------------
``` run_10.sh ``` creates the necessary enviornment and python version. It calls ``` train.py``` and passes a seed parameter through it.

```run_all5.sh ``` schedules five jobs of ``` run_10.sh ``` on compute nodes with different seeds.

```
chmod 775 run_10.sh

chmod 775 run_all5.sh

./run_all5.sh
``` 
