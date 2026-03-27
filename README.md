# Learning Network Dismantling without Handcrafted Inputs

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.00706-b31b1b.svg)](https://arxiv.org/abs/2508.00706)

<img src="files/fig_auc_bg.svg" alt="MIND Schematic" width="800"><br>

<sub><i>
<b>Network dismantling</b> seeks a sequence of node removals that fragments a network as rapidly as possible into disconnected components. Here, the dismantling of a <a href="https://dl.acm.org/doi/abs/10.1145/2856037">social network</a> is illustrated, where we sequentially remove nodes and record the size of the largest connected component (LCC) after each removal. The objective of network dismantling is to minimize the area under the LCC curve (AUC).
</i></sub>

</div>

---

## Overview

This is the **official implementation** of the MIND algorithm, presented in:

> **Learning Network Dismantling without Handcrafted Inputs**  
> Haozhe Tian, Pietro Ferraro, Robert Shorten, Mahdi Jalili, Homayoun Hamedmoghadam
> *AAAI-26 Main Technical Track (Oral)*
> [[Full Paper]]([https://arxiv.org/abs/2508.00706](https://arxiv.org/abs/2508.00706))

MIND finds the sequence of node removals that most rapidly fragments a network into disconnected components.

---

## Configuration

This implementation was tested on Ubuntu 22.04. To set up the `conda` environment, navigate to the repository directory and run the following commands:

```bash
conda env create -f environment.yml
conda activate mind_nd

# install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# install PyG
pip install torch_geometric
```

---

## Quick visualization

We provide a notebook [`visualize.ipynb`](visualize.ipynb) for visualizing the dismantling process of networks. To visualize your own networks, save your `igraph.Graph` object as a `.pkl` file in a target folder and specify that folder path when initializing the environment in the notebook, as shown below:
```python
# for large networks, set render to False
env = DismantleEnv('your_folder_here', batch_size=5, is_val=True, render='plot')
```

---

## Dismantle Custom Networks

To dismantle custom networks and compute the AUC, place your network files in a directory (e.g., `graphs/example`) and run the following command. The results will be saved as a `.csv` file in the `results` directory.

```bash
python test.py --device cuda --ckpt_pth saved/mind.ckpt --directory graphs/example
```

---

## Dismantle Real Networks

Run the following command to dismantle all real networks in `graphs/real`. Note that this process is computationally intensive (due to large networks with millions of nodes) and may take several days on a GPU.

```bash
python test.py --device cuda --ckpt_pth saved/mind.ckpt --directory graphs/real
```

---

## Citation

If you find the code in this repository useful, please cite:

```bibtex
@inproceedings{tian2026learning,
  title={Learning Network Dismantling Without Handcrafted Inputs},
  author={Tian, Haozhe and Ferraro, Pietro and Shorten, Robert Noel and Jalili, Mahdi and Hamedmoghadam, Homayoun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={25905--25913},
  year={2026}
}
```
