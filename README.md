# Single Node Injection Attack against Graph Neural Networks

**This repository is our Pytorch implementation of our paper:**

**[Single Node Injection Attack against Graph Neural Networks](https://arxiv.org/abs/2007.09647)** 

By Shuchang Tao, Qi Cao, Huawei Shen, Junjie Huang, Yunfan Wu and Xueqi Cheng

**Published at CIKM 2021**



## Introduction





<img src="./imgs/Example.png" />




## G-NIA



<img src="./imgs/Model.png" />



The training and test process of AdvImmune.



## Requirements

- pytorch 

- scipy

- numpy

  



## Usage

***Example Usage***

`python -u main.py --dataset citeseer --scenario rem `

For detailed description of all parameters, you can run

`python -u main.py --help`



## Cite

If you would like to use our code, please cite:

```
@inproceedings{tao2021gnia,
  title={Single Node Injection Attack against Graph Neural Networks},
  author={Shuchang Tao and Qi Cao and Huawei Shen and Junjie Huang and Yunfan Wu and Xueqi Cheng.},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  series={CIKM'21},
  year={2021}
}
```
