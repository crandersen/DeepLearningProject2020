"""
Finished on Mon January 11 2021

@author: chrisan@dtu.dk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from skimage.io import imread
from random import randint

# # This code is being used to train and classify structures from
# # simulated TEM images.
# # To generate samples use GenerateSamplesCDN
# # It needs structures of both WZ and ZB type. 


# # Show number of samples:
image_paths = glob.glob("dataCDN/*.jpeg")
print("Total Observations:\t", len(image_paths))

# # Choose how many samples you want to study
datalength = 400 # Use len(image_paths) if you want to use the full dataset
print("Used Observations:\t", datalength)

classes = ('WZ','ZB')
used_classes = range(len(classes))

NWFileNames = []
NWStructeres = []
NWStructuresClass = []
NWLayers = []

idxWZ = 0
idxZB = 0
for i in range(datalength):
    if i % 2 == 1:
        NWFileNames.append("WZ" + str(idxWZ))
        idxWZ += 1
    else:
        NWFileNames.append("ZB" + str(idxZB))
        idxZB += 1
    File = pd.read_csv('dataCDN/' + NWFileNames[i] +'.txt', sep=":", header=None)
    NWLayers.append(File[1][1])
    NWStructeres.append(File[1][0])
    if File[1][0] == ' WZ':
        NWStructuresClass.append(0)
    else:
        NWStructuresClass.append(1)

filenames_images = ['dataCDN/{}.jpeg'.format(i) for i in NWFileNames]


images = np.empty(shape=(datalength,1,512,512))

for i in range(datalength):
  images[i][0] = imread(filenames_images[i], as_gray=True)




# # Plot example
plt.figure(figsize=(8, 8))
indx = randint(0,datalength-1)
plt.imshow(images[indx][0], cmap='gray')
plt.title("Structure: %s \nNumber of layers: %s" % (NWStructeres[indx],NWLayers[indx]))
plt.axis('off')
plt.show()





# # Neural network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from netcdn import Net

num_epochs = 10
batch_size = 5

num_classes = len(classes)

net = Net(num_classes)
print(net)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters()) 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

# # Test the forward pass with dummy data
x = np.random.normal(0,1, (3, 1, 512, 512)).astype('float32')
out = net(Variable(torch.from_numpy(x)))
print(out.size())
print(out)



# # Training 
train_acc, train_loss = [], []
test_acc, test_loss = [], []
valid_acc, valid_loss = [], []
losses = []

# # Percentage for training, test and validation
pct_train = 0.7
pct_valid = 0.15
pct_test = 0.15

x_train, y_train = torch.FloatTensor(images[:int(pct_train*datalength)]),torch.LongTensor(NWStructuresClass[:int(pct_train*datalength)])
x_valid, y_valid = torch.FloatTensor(images[int(pct_train*datalength):int((pct_train+pct_valid)*datalength)]),torch.LongTensor(NWStructuresClass[int(pct_train*datalength):int((pct_train+pct_valid)*datalength)])
x_test, y_test = torch.FloatTensor(images[int((pct_train+pct_valid)*datalength):]),torch.LongTensor(NWStructuresClass[int((pct_train+pct_valid)*datalength):])
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

get_slice = lambda i, size: range(i * size, (i + 1) * size)

for epoch in range(num_epochs):
    # # Forward -> Backprob -> Update params
    # # Train
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(x_train[slce])
        
        # # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss += batch_loss   
    losses.append(cur_loss / batch_size)

    net.eval()
    # # Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        output = net(x_train[slce])
        
        preds = torch.max(output, 1)[1]
        
        train_targs += list(y_train[slce].numpy())
        train_preds += list(preds.data.numpy())
    
    # # Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size)
        
        output = net(x_valid[slce])
        preds = torch.max(output, 1)[1]
        val_targs += list(y_valid[slce].numpy())
        val_preds += list(preds.data.numpy())
        

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
    # # Print progress. 
    # # For small numbers of epochs it will print all
    if num_epochs < 100:
       print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
           epoch+1, losses[-1], train_acc_cur, valid_acc_cur))
    else: 
        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                    epoch+1, losses[-1], train_acc_cur, valid_acc_cur))


# # Plot results
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')

plt.figure()
plt.plot(epoch, losses, 'r')
plt.legend(['Loss'])
plt.xlabel('Updates'), plt.ylabel('Loss')


# # Test
numbers = []
for i in range(4): numbers.append(randint(0,datalength-1))

fig = plt.figure(figsize=(30, 50))

for i in range(4):
  ax = fig.add_subplot(1,4,i+1)
  ax.imshow(images[numbers[i]][0], cmap='gray')
  plt.title("%s" % NWStructeres[numbers[i]])
  ax.axis('off')


print('GroundTruth:  ', ' '.join('%5s' % classes[NWStructuresClass[numbers[j]]] for j in range(4)))
outputs = net(torch.Tensor(images[numbers]))
_, predicted = torch.max(outputs.data, 1)
print('Predicted:    ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


class_total = list(0. for i in range(len(classes)))
class_correct = list(0. for i in range(len(classes)))
c = []


# # # Testing full dataset
# for i in range(datalength):
#     outputs = net(Variable(torch.FloatTensor(images[i:i+1])))
#     _, predicted = torch.max(outputs.data, 1)
#     c.append((predicted == torch.tensor(NWStructuresClass[i])).squeeze())

# for i in range(len(c)):
#     label = NWStructuresClass[i]
#     class_correct[label] += c[i].numpy()
#     class_total[label] += 1

# for i in range(len(classes)):
#     print('Accuracy of {:5s} : {:5.2f} %'.format(
#         classes[i], 100 * class_correct[i] / class_total[i]))


# # # Testing only test dataset
for i in range(len(x_test)):
    outputs = net(x_test[i:i+1])
    _, predicted = torch.max(outputs.data, 1)
    c.append((predicted == y_test[i]).squeeze())

for i in range(len(c)):
    label = y_test[i]
    class_correct[label] += c[i].numpy()
    class_total[label] += 1

for i in range(len(classes)):
    print('Accuracy of {:5s} : {:5.2f} %'.format(
        classes[i], 100 * class_correct[i] / class_total[i]))