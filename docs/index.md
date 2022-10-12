<h1 align="center"> AssembleRL: Learning to Assemble Furniture from Their Point Clouds </h1>

<h3 align="center">Abstract</h3>
The rise of simulation environments has enabled learning-based approaches for assembly planning, which is otherwise a labor-intensive and daunting task. Assembling furniture is especially interesting since furniture are intricate and pose challenges for learning-based approaches. Surprisingly, humans can solve furniture assembly mostly given a 2D snapshot of the assembled product. Although recent years have witnessed promising learning-based approaches for furniture assembly, they assume the availability of correct connection labels for each assembly step, which are expensive to obtain in practice. In this paper, we alleviate this assumption and aim to solve furniture assembly with as little human expertise and supervision as possible. To be specific, we assume the availability of the assembled point cloud, and comparing the point cloud of the current assembly and the point cloud of the target product, obtain a novel reward signal based on two measures: Incorrectness and incompleteness. We show that our novel reward signal can train a deep network to successfully assemble different types of furniture. 

<h3 align="center">Network Architecture</h3>
<p align="center">
  <img src="imgs/arch.png" alt="arch" width="60%"/>
</p>

An overview of the system is depicted. Point cloud of the current assembly is processed by graph-convolutional layers followed by fully connected layers. The selected action is rewarded by comparing the updated assembly with the target assembly.

<h3 align="center">Incompleteness and Incorrectness Measures</h3>
<p align="center"><img src="imgs/reward.png" alt="reward" width="60%"/></p>

An illustration showing how the proposed measures and the reward values change over time. (a) Assembly of a table with three steps. (b) How different assembly actions affect the measures.

<h3 align="center">Quantitative Results</h3>

<p align="center"><img src="imgs/results.png" alt="results" width="60%"/> </p>
<strong>SRcon (Connection success rate):</strong> The ratio of correct connections done by the agent to the number of total correct connections.

<strong>SRa (Furniture assembly success rate):</strong> The ratio of correctly assembled furniture to the total number of furniture.

<h3 align="center">References</h3>

The assembly learning environment includes:
- Point cloud manipulation and ICP using open3d: https://github.com/isl-org/Open3D
- Furniture objects and transform utilities (slightly modified) from IKEA Furniture Assembly Environment: https://github.com/clvrai/furniture
- Pointnet++ and GCN implementations with pytorch-geometric: https://github.com/pyg-team/pytorch_geometric 
- RL agent implementation and training using stable-baselines3: https://github.com/DLR-RM/stable-baselines3 
