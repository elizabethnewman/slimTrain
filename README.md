# pytorch-slimtik

### Installation
```angular2html
git clone https://github.com/elizabethnewman/pytorch-slimtik.git
cd pytorch-slimtik
python setup.py install
```

### Organization
There are three experiments implemented in this repository. Each experiment has its own data generation, 
training algorithm, and utilities.  The pinns and autoencoder examples have their own networks as well.

* **autoencoder**: This has a simple autoencoder for MNIST implemented.
  
* **peaks**: This is a function approximation example mapping from the coordinate plane to a single real value. 
    It is a good toy problem to test the code and new ideas.

* **pinns**: Physics-Informed Neural Networks. This has Poisson's equation implemented.   

