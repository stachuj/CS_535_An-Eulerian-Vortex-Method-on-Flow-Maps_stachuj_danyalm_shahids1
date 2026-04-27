# CS535 An Eulerian Vortex Method on Flow Maps
This is a fork created for a CS535 class project at SUNY Polytechnic Institute  
by Justin Stachurski, Mian Danyal, Shazman Shahid

## Installation
Code is tested on Windows 11 with CUDA 13.1, Python 3.13.3, and Taichi 1.7.4, using Visual Studio Code.

Install the requirements with:

```bash
pip install -r requirements.txt
```

## Simulation
For reproducing the same result in the paper, execute:

```bash
python run_paper.py
```

With the default settings and no code modifications (i.e., the default 3D leapfrog), the expected runtime for the above execution ranges from 30 to 60 minutes, depending on your machine's performance.

An improved version by storing the flow map quantities with the same locations as vorticity can be obtained by running:

```bash
python run_improved.py
```

The improved version enhances the simulation stability and the vorticity preservation ability, e.g., it leads to one more leap in 3D leapfrog.

Hyperparameters can be tuned by changing the values in the file `hyperparameters.py`.  
  
Our modified version is run with:

```bash
python run_cs535.py
```
This version has slightly better run time than run_improved, as it relies on RK2 function rather than RK4, along with small changes to buffer copies.

## Visualization
The results will be stored in `logs/[exp_name]/vtks`. We recommend using ParaView to load these `.vti` files as a sequence and visualize them by selecting **Volume** in the Representation drop-down menu.


## Bibliography
[TOG (SIGGRAPH Asia 2024)] An Eulerian Vortex Method on Flow Maps
by [Sinan Wang](https://sinanw.com), [Yitong Deng](https://yitongdeng.github.io/), [Molin Deng](https://molin7.vercel.app), [Hong-Xing Yu](https://kovenyu.com/), [Junwei Zhou](https://zjw49246.github.io/website/), [Duowen Chen](https://cdwj.github.io), [Taku Komura](https://hku-cg.github.io/author/taku-komura/), [Jiajun Wu](https://jiajunwu.com/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

The paper and video results can be found at [project website](https://evm.sinanw.com/).

This work has been awarded the **[Replicability Stamp](http://www.replicabilitystamp.org#https-github-com-pfm-gatech-an-eulerian-vortex-method-on-flow-maps-git)**.
[![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-pfm-gatech-an-eulerian-vortex-method-on-flow-maps-git)

If you find our paper or code helpful, consider citing:
```bibtex
@article{wang2024eulerian,
  title={An Eulerian Vortex Method on Flow Maps},
  author={Wang, Sinan and Deng, Yitong and Deng, Molin and Yu, Hong-Xing and Zhou, Junwei and Chen, Duowen and Komura, Taku and Wu, Jiajun and Zhu, Bo},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--14},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
