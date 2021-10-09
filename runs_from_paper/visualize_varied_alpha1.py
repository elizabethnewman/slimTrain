import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import pylab
import glob, os
import numpy as np

plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'image.interpolation': None})
plt.rcParams['figure.figsize'] = [14, 12]
plt.rcParams['figure.dpi'] = 200

load_dir = '/Users/elizabethnewman/Desktop/autoencoder/'
save_dir = '/Users/elizabethnewman/Desktop/varproSAPaperResults/img/autoencoder/'

#%%
os.chdir(load_dir)
load_flag = lambda fname: (('50000' in fname and 'seed-20' in fname and 'epochs-50' in fname) and
                           ('alpha1-1.00e-10--alpha2-1.00e-10' in fname or
                            'alpha1-1.00e-03--alpha2-1.00e-03' in fname or
                            'alpha1-1.00e-01--alpha2-1.00e-01' in fname or
                            'alpha1-1.00e-05--alpha2-1.00e-05' in fname or
                            'alpha1-1.00e+00--alpha2-1.00e+00' in fname)
                           )


hisADAM, idxADAM, alpha2ADAM = [], [], []
count = 0
for fname in glob.glob("*.pt"):
    if load_flag(fname) and 'adam' in fname:
        print(fname)
        stored_results = pickle.load(open(fname, 'rb'))
        hisADAM.append(stored_results['results'])
        idxADAM.append(count)

        if 'alpha2-1.00e-10' in fname:
            alpha2ADAM.append(1e-10)
        if 'alpha2-1.00e-05' in fname:
            alpha2ADAM.append(1e-5)
        if 'alpha2-1.00e-03' in fname:
            alpha2ADAM.append(1e-3)
        if 'alpha2-1.00e-01' in fname:
            alpha2ADAM.append(1e-1)
        if 'alpha2-1.00e+00' in fname:
            alpha2ADAM.append(1e0)
        count += 1

idxADAM = [i for _, i in sorted(zip(alpha2ADAM, idxADAM))]
idxADAM.reverse()


hisSlimTrain, idxSlimTrain, alpha2SlimTrain = [], [], []
count = 0
for fname in glob.glob("*.pt"):
    if load_flag(fname) and 'slimtrain' in fname:
        print(fname)
        stored_results = pickle.load(open(fname, 'rb'))
        hisSlimTrain.append(stored_results['results'])
        idxSlimTrain.append(count)

        if 'alpha2-1.00e-10' in fname:
            alpha2ADAM.append(1e-10)
        if 'alpha2-1.00e-05' in fname:
            alpha2ADAM.append(1e-5)
        if 'alpha2-1.00e-03' in fname:
            alpha2ADAM.append(1e-3)
        if 'alpha2-1.00e-01' in fname:
            alpha2ADAM.append(1e-1)
        if 'alpha2-1.00e+00' in fname:
            alpha2ADAM.append(1e0)
        count += 1

idxSlimTrain = [i for _, i in sorted(zip(alpha2SlimTrain, idxSlimTrain))]
idxSlimTrain.reverse()
