#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Libraries
import numpy as np
import pandas as pd
import cv2
import os
import glob
# from scipy import misc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time


# In[2]:


curDir = os.getcwd()
TrainPath = os.path.join(curDir, 'Train')
TestPath = os.path.join(curDir, 'Test')
ValidationPath = os.path.join(curDir, 'Validation')
# Read and Sort Train Data
#os.chdir(TrainPath)
#TrainFiles = glob.glob( '*.jpg')
#TrainFiles = sorted(TrainFiles, key=lambda x: int(os.path.splitext(x)[0]))


# In[3]:


# Read and Sort data files | No need to change directory
# Train Files
TrainFiles = glob.glob(os.path.join(TrainPath, '*.jpg'))
TrainFiles = sorted(TrainFiles, key=lambda x: int(os.path.splitext(os.path.split(x)[1])[0]))
# Test Files
TestFiles = glob.glob(os.path.join(TestPath, '*.jpg'))
TestFiles = sorted(TestFiles, key=lambda x: int(os.path.splitext(os.path.split(x)[1])[0]))
# Val Files
ValFiles = glob.glob(os.path.join(ValidationPath, '*.jpg'))
ValFiles = sorted(ValFiles, key=lambda x: int(os.path.splitext(os.path.split(x)[1])[0]))


# In[4]:


# Checks 1
print('First 5 files in Train\t', [os.path.split(x)[1]  for x in TrainFiles[:5]])
print('File no. 1395 in Train\t', os.path.split(TrainFiles[1358])[1])
print('Last 5 files in Train\t', [os.path.split(x)[1]  for x in TrainFiles[-5:]])
print('First 5 files in Test\t', [os.path.split(x)[1]  for x in TestFiles[:5]])
print('First 5 files in Valid\t', [os.path.split(x)[1]  for x in ValFiles[:5]])


# In[5]:


# Read Train data
X_train = []
with open(os.path.join(TrainPath, 'Training Labels.txt'), 'r', encoding='utf-8') as tlabel:
    Y_train = np.array(tlabel.readlines(), dtype='int')
for f in TrainFiles:
    image = cv2.imread(f, 0) / 255 # dividing by 255 for normalization
    #change dimention to 1 dimentional vector instead of (28x28) array
    dim = image.shape #28x28
    Features = np.prod(dim) # 28*28 = 784
    image=image.reshape(Features,) # reshape to (784,)
    image=np.append(image,1).reshape(Features+1, 1) # appending 1 for bias term and reshape to (785, 1)
    X_train.append(image)
X_train = np.array(X_train)
# checks
print(Y_train[0:5])
print(Y_train.shape)
print(X_train.shape)
print(dim, Features)


# In[6]:


# Read Test data
X_test = []
with open(os.path.join(TestPath, 'Test Labels.txt'), 'r', encoding='utf-8') as tlabel:
    Y_test = np.array(tlabel.readlines(), dtype='int')
for f in TestFiles:
    image = cv2.imread(f, 0) / 255 # dividing by 255 for normalization
    #change dimention to 1 dimentional vector instead of (28x28) array
    dim = image.shape #28x28
    Features = np.prod(dim) # 28*28 = 784
    image=image.reshape(Features,) # reshape to (784,)
    image=np.append(image,1).reshape(785, 1) # appending 1 for bias term and reshape to (785, 1)
    X_test.append(image)
X_test = np.array(X_test)
# checks
print(Y_test.shape)
print(X_test.shape)
print(len(X_test))


# In[7]:


# Read Validation data
X_val = []
with open(os.path.join(ValidationPath, 'Validation Labels.txt'), 'r', encoding='utf-8') as tlabel:
    Y_val = np.array(tlabel.readlines(), dtype='int')
for f in ValFiles:
    image = cv2.imread(f, 0) / 255 # dividing by 255 for normalization
    #change dimention to 1 dimentional vector instead of (28x28) array
    dim = image.shape #28x28
    Features = np.prod(dim) # 28*28 = 784
    image=image.reshape(Features,) # reshape to (784,)
    image=np.append(image,1).reshape(785, 1) # appending 1 for bias term and reshape to (785, 1)
    X_val.append(image)
X_val = np.array(X_val)
# checks
print(Y_val.shape)
print(X_val.shape)
print(len(X_val))


# In[8]:


# Saving data To use in colab
newpath=os.path.join(curDir,'CompressedData')
if not os.path.exists(newpath):
    os.makedirs(newpath)
np.savez_compressed(os.path.join(newpath,'TrainD'), X_train=X_train, Y_train=Y_train)
np.savez_compressed(os.path.join(newpath,'TestD'), X_test=X_test, Y_test=Y_test)
np.savez_compressed(os.path.join(newpath,'ValD'), X_val=X_val, Y_val=Y_val)


# In[9]:


# restore compressed data
path = os.path.join(curDir,'CompressedData')
train = np.load(os.path.join(path, 'TrainD.npz'))
X_train = train['X_train']
Y_train = train['Y_train']
test = np.load(os.path.join(path, 'TestD.npz'))
X_test = test['X_test']
Y_test = test['Y_test']
val = np.load(os.path.join(path, 'ValD.npz'))
X_val = val['X_val']
Y_val = val['Y_val']
X_train = X_train.reshape(X_train.shape[:2])
X_test = X_test.reshape(X_test.shape[:2])
X_val = X_val.reshape(X_val.shape[:2])
dim = [28]*2
Features = X_train.shape[1]-1 # 28*28 = 784


# In[10]:


# checks 2
print(Y_train[0:5])
print(Y_train[::100])
print(Y_train.shape)
print(X_train.shape)
print(Y_test.shape)
print(X_test.shape)
print(Y_val.shape)
print(X_val.shape)
print(dim, Features)


# In[11]:


### Part a ###
#with tf.device('/gpu:0'):
#tf.initializers.global_variables()
# Defining the Perceptron function 2ba
def uniPerceptron(X, Y, region, eta, lr, n_class=10, Features = 28*28, epsilon=10^-4):
    W = np.concatenate((np.ones((n_class, 1)), np.zeros((n_class, Features))), axis=1) # Weights Initialization
    ### In case of disconnected while training
    # if lr == 9: 
        # W = np.load(os.path.join(curDir,'CompressedData','ClassesWeightsbackup.npz'))['W']
    ### W is 10x785 X is 2400x785, Y is 2400x1, region is 2400x10 (region is the t in lectures)
    count = 0
    Y_new = np.zeros(Y.shape)
    ### looping until there is no error or number of iterations exceeds 6600 epochs
    while (Y_new==Y).sum() != np.prod(Y.shape):# and count < 600:# >= epsilon:
        count += 1
        inds = (X @ W.T) * region # 2400x10 ## All the error values
        # getting the lables of all the miss-classified samples
        miss = np.zeros(inds.shape) # 2400x10
        miss[np.nonzero(inds<=0)] = region[np.nonzero(inds<=0)] 
        # updating W
        W = W + eta*(miss.T @ X)
        Y_new = (X @ W.T).argmax(axis=1)
        if lr > 7 and not count%2000: 
            # np.savez_compressed(os.path.join(curDir,'CompressedData','ClassesWeightsNew'), W=W)
            if not count%2000: np.savez_compressed(os.path.join(curDir,'CompressedData','ClassesWeightsbackup'), W=W)
    return W, count


# In[12]
    

# Training
Y_temp = np.zeros((len(Y_train), 10))
for i in range(10):
    Y_temp[:, i] = np.copy(Y_train)
    Y_temp[np.argwhere(i==Y_train), i] = 1
    Y_temp[np.argwhere(i!=Y_train), i] = -1
TotalWeights = np.zeros((10, 10, Features + 1)) # replaced with the most recent weight 
for lr in range(10):
    beg = time.time()
    eta = pow(10, -lr)
    print('Approaching for eta =', eta)
    #Weights = []
    W_final, c = uniPerceptron(X_train, Y_train, Y_temp, eta, lr, 10, Features)
    TotalWeights[lr] = W_final
    print('\tThe 10 classes took %d iterations to approach' %c)
    print('Time taken for eta = {:.0e} is {:.3f} mins\n'.format(eta, (time.time()-beg)/60))
    np.savez_compressed(os.path.join(curDir,'CompressedData','AllWeights-%d' %lr), TotalWeights=TotalWeights)
TotalWeights = [[xx/np.linalg.norm(xx) for xx in x] for x in TotalWeights] # Normalization
TotalWeights = np.array(TotalWeights)


# In[13]:


# Saving Weights
np.savez_compressed(os.path.join(curDir,'CompressedData','AllWeights'), TotalWeights=TotalWeights)


# In[14]:


# Restore Weights
TotalWeights = np.load(os.path.join(curDir,'CompressedData','AllWeights.npz'))['TotalWeights']


# In[15]:


# Function to calculate and convert confusion matrix to image
def cm_analysis(y_true, y_pred, filename, labels, titl = 'Confusion Matrix', figsize=(10,10)):
    """
    This function is copied from 'https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7'
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(titl)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)


# In[16]:


# Testing
TotalWeights = TotalWeights.reshape(10, 10, Features + 1)
newpath=os.path.join(curDir,'Figures')
if not os.path.exists(newpath):
    os.makedirs(newpath)
for lr in range(10):
    eta = pow(10, -lr)
    print('Prediction for eta = {:.0e}'.format(eta))
    Weights = np.copy(TotalWeights[lr]) # 10x785
    Y_predicted = (X_test @ Weights.T).argmax(axis=1).reshape(len(Y_test))
    acc = np.count_nonzero(Y_predicted==Y_test)/len(Y_test)*100
    print('\tTest accuracy = {}%'.format(acc))
    titl = 'Confusion Matrix at eta = 10^{} with accuracy = {}%'.format(-lr, acc)
    cm_analysis(Y_test, Y_predicted, os.path.join(newpath,'Confusion-{}.jpg'.format(lr)), list(range(10)), titl)


# In[17]:


### Part B ###
# Validation
newpath=os.path.join(curDir,'ValFigures')
if not os.path.exists(newpath):
    os.makedirs(newpath)
compare = np.zeros((10,10))
for lr in range(10):
    eta = pow(10, -lr)
    print('Prediction for eta = {:.0e}'.format(eta))
    Y_predicted = []
    Weights = np.copy(TotalWeights[lr])
    Y_predicted = np.array(X_val @ Weights.T).argmax(axis=1).reshape(len(Y_val))
    for i in range(10): # classes
	# get the accuracy of each class for every learning rate value
        compare[i,lr] = (Y_predicted[np.argwhere(i==Y_val)]==Y_val[np.argwhere(i==Y_val)]).sum()
    acc = np.count_nonzero(Y_predicted==Y_val)/len(Y_val)*100
    print('\tVal accuracy = {}%'.format(acc))
BestVal = compare.argmax(axis=1) # the best learning rate for each class
print(BestVal) 


# In[18]:


# Generating the new prediction using best learning rate for every class
WeightsNew = TotalWeights[BestVal,range(10)]
Y_PredNew = np.array(WeightsNew @ X_test.T).argmax(axis=0).reshape(len(Y_val)) # new prediction
acc = np.count_nonzero(Y_PredNew==Y_test)/len(Y_test)*100
print('\tNew drived accuracy = {}%'.format(acc))
titl = 'Part B Confusion Matrix with accuracy = {}%'.format(acc)
cm_analysis(Y_test, Y_predicted, os.path.join('Figures','Confusion-b.jpg'.format(lr)), list(range(10)), titl)


