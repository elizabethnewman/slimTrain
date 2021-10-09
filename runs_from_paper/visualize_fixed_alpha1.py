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
load_flag = lambda fname: (('50000' in fname and 'alpha1-1.00e-10' in fname and 'seed-20' in fname and 'epochs-50' in fname) and
                           ('alpha2-1.00e-10' in fname or 'alpha2-1.00e-01' in fname or  'alpha2-1.00e+00' in fname))


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
    if load_flag(fname) and 'slimtrain' in fname and 'alpha2-1.00e-10' in fname:
        print(fname)
        stored_results = pickle.load(open(fname, 'rb'))
        hisSlimTrain.append(stored_results['results'])
        idxSlimTrain.append(count)

        if 'alpha2-1.00e-10' in fname:
            alpha2SlimTrain.append(1e-10)
        if 'alpha2-1.00e-01' in fname:
            alpha2SlimTrain.append(1e-1)
        if 'alpha2-1.00e+00' in fname:
            alpha2SlimTrain.append(1e0)
        count += 1

idxSlimTrain = [i for _, i in sorted(zip(alpha2SlimTrain, idxSlimTrain))]
idxSlimTrain.reverse()


#%% Plot epochs 1 to 10

x = np.arange(1, 11)
cmap = ['#ff7f0e', '#2ca02c', '#1f77b4']

os.chdir(load_dir)
markers = ['v', 's', 'd']
for a, i in enumerate(idxADAM):
    his = hisADAM[i]
    label = 'ADAM'
    h = plt.plot(x, his['val'][1:11, his['str'].index('train_loss')].numpy(), '-',
                 marker=markers[a], color=cmap[a],
                 label=label + ', $\lambda$=%0.2e' % alpha2ADAM[i],
                 linewidth=6, markersize=20, markeredgewidth=2, fillstyle='none')


count = 0
for i in idxSlimTrain:
    his = hisSlimTrain[i]
    alpha2 = alpha2SlimTrain[i]
    label = 'slimTrain, sGCV'
    h = plt.plot(x, his['val'][1:11, his['str'].index('train_loss')].numpy(), '-o', color='k',
                     label=label + ', $\Lambda_0$=%0.2e' % alpha2, linewidth=6, markersize=20, alpha=1)
    count += 1

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


#%% Plot epochs 0 to 50

x = np.arange(0, 51)
cmap = ['#ff7f0e', '#2ca02c', '#1f77b4']

os.chdir(load_dir)
markers = ['v', 's', 'd']
for a, i in enumerate(idxADAM):
    his = hisADAM[i]
    label = 'ADAM'
    h = plt.semilogy(x, his['val'][:, his['str'].index('train_loss')].numpy(), '-',
                     marker=markers[a], color=cmap[a],
                     label=label + ', $\lambda$=%0.2e' % alpha2ADAM[i], markevery=2,
                     linewidth=6, markersize=20, markeredgewidth=2, fillstyle='none')


count = 0
for i in idxSlimTrain:
    his = hisSlimTrain[i]
    alpha2 = alpha2SlimTrain[i]
    label = 'slimTrain, sGCV'
    h = plt.semilogy(x, his['val'][:, his['str'].index('train_loss')].numpy(), '-o', color='k', markevery=2,
                     label=label + ', $\Lambda_0$=%0.2e' % alpha2, linewidth=6, markersize=20, alpha=1)
    count += 1

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
