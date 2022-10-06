# AssembleRL: Learning to Assemble Furniture from Their Point Clouds

This repo provides the official implementation of the IROS 2022 paper "AssembleRL: Learning to Assemble Furniture from Their Point Clouds" by [Özgür Aslan](https://www.linkedin.com/in/%C3%B6zg%C3%BCr-aslan-44192917b/), [Burak Bolat](https://www.linkedin.com/in/burak-bolat-2579a0177/), [Batuhan Bal](https://batubal.github.io/), [Tuğba Tümer](www.linkedin.com/in/tugbatumer), [Erol Şahin](https://kovan.ceng.metu.edu.tr/~erol/), and [Sinan Kalkan](https://kovan.ceng.metu.edu.tr/~sinan/index.html).

<p align="center"><img src="docs/imgs/intro.png" alt="intro" width="40%"/></p>

## What is AssembleRL?

We propose to use only the fully assembled point cloud of the furniture and the mesh models of its parts, to learn the assembly plan. Specifically, we introduce a novel reward function that evaluates the match between the point cloud of the partially assembled furniture against its fully assembled view using two measures that evaluate the *incorrectness* and *incompleteness*. We train a graph-convolutional neural network with our novel reward signal, combining the incorrectness and incompleteness measures, to learn the assembly plan as a policy that predicts which part pairs need to be connected via which of their connections. 

## Running the Code

### Dependencies
- pytorch
- pytorch-geometric
- open3d
- stable-baselines3
- gym
- numpy
- networkx

### File Structure
```
assembly_learning
│
│─── envs
│    │─── asc_base.py
│    │─── asc_features.py
│    │─── asc_supervised.py
│
│─── objects
│
│─── utils
│    │─── graph.py
│    │─── pointcloud.py
│    │─── pointnet2_utils.py
│    │─── pointnet2.py
│    │─── reg_reward.py
│    │─── transform_utils.py
│
│─── pointnet2.pt

```

## Train and Test
Pretrained Pointnet++ weight can be downloaded from [here](https://drive.google.com/drive/folders/1yD_1t5VYN32fOPhhBxyVxjIDIqtg8Ma0)  
After download put the file under the [assembly_learning](./assembly_learning) folder.  

For training [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) can be used.  
Also simple [train](train.py) and [test](test.py) files using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) are included.
To train a model with spesific furniture and reward implementation:  
```
python train.py -f ivar -r partial_th
```
To test a trained model:
```
python test.py -f ivar -r partial_th
```

## References

The assembly learning environment includes:
- Point cloud manipulation and ICP using open3d: https://github.com/isl-org/Open3D
- Furniture objects and transform utilities (slightly modified) from IKEA Furniture Assembly Environment: https://github.com/clvrai/furniture
- Pointnet++ and GCN implementations with pytorch-geometric: https://github.com/pyg-team/pytorch_geometric 
- RL agent implementation and training using stable-baselines3: https://github.com/DLR-RM/stable-baselines3 
