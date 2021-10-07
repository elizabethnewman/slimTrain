# slimTrain

A stochastic approximation strategy for training separable DNNs with automatic regularization parameter selection.



### Citation

If you use this code, please cite the following paper:

```latex
@misc{newman2021slimtrain,
      title={slimTrain -- A Stochastic Approximation Method for Training Separable Deep Neural Networks}, 
      author={Elizabeth Newman and Julianne Chung and Matthias Chung and Lars Ruthotto},
      year={2021},
      eprint={2109.14002},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Overview

This repository contains code to train a convolutional MNIST autoencoder using *slimTrain*.  

### Installation
```angular2html
git clone https://github.com/elizabethnewman/slimTrain.git
cd slimTrain
pip install -r requirements.txt
python setup.py install
```

### Organiziation

* **autoencoder**: contains the autoencoder architecture, functions to load the MNIST dataset, and the MNIST autoencoder network used.
* **slimtik_functions**: computes the updated weights using slimTik and applies automatic regularization parameter selection

### References

[Iterative Sampled Methods for Massive and Separable Nonlinear Inverse Problems](https://par.nsf.gov/servlets/purl/10108292) 
(Chung, Chung and Slagel, 2019)

[Sampled Tikhonov Regularization for Large Linear Inverse Problems](https://arxiv.org/abs/1812.06165) 
(Slagel et al., 2019)

[Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection](https://arxiv.org/abs/2007.13171)
(Newman, Ruthotto, Hart, van Bloemen Waanders)



