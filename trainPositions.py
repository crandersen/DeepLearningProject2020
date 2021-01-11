"""
Finished on Mon January 11 2021

@author: chrisan@dtu.dk
"""

from glob import glob
from filesFCN.dataset import DataSet, DataEntry
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abtem.transfer import CTF

def load(data_dir):
    
    waves = glob(data_dir+"wave/wave_*.hdf5")
    points = glob(data_dir+"points/points_*.npz")

    entries = [DataEntry(wave=w, points=p) for w,p in zip(waves,points)]
    
    return DataSet(entries)

def show_examples(data, size, n=3,test=False):
    
    image,label=next_example(data,size)
    
    if test:
        fig,axarr=plt.subplots(3,n)
        outputs = net(Variable(torch.from_numpy(image)).unsqueeze_(0).unsqueeze_(0))
        predicted = torch.max(outputs, 1)[1]
        
    else:
        fig,axarr=plt.subplots(2,n)
        
    
    for i in range(n):
        
        im = axarr[0,i].imshow(image, interpolation='nearest', cmap='gray')
        
        divider = make_axes_locatable(axarr[0,i])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax1)

        im = axarr[1,i].imshow(label[0,:,:], cmap='jet',vmin=0, vmax=2)
        
        divider = make_axes_locatable(axarr[1,i])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)
        if test:
            im = axarr[2,i].imshow(predicted[0].numpy(), cmap='jet',vmin=0, vmax=2)
        
            divider = make_axes_locatable(axarr[2,i])
            cax3 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax = cax3)
            
        if i < n - 1:
            image, label = next_example(data, size)
            if test:
                outputs = net(Variable(torch.from_numpy(image)).unsqueeze_(0).unsqueeze_(0))
                predicted = torch.max(outputs, 1)[1]
                
    
    plt.tight_layout()
    plt.show()

def next_example(data,size):

    sampling = np.random.uniform(.084,.09)
    Cs = -7 * 10**4
    defocus = -85.34
    focal_spread = 0
    semiangle_cutoff = 20
    
    dose = 10**np.random.uniform(2,4)
    
    poisson_noise = 5000
    
    entry=data.next_batch(1)[0]
    
    entry.load()
    
    ctf=CTF(defocus=defocus,Cs=Cs,focal_spread=focal_spread,semiangle_cutoff=semiangle_cutoff)
    
    entry.create_image(ctf,sampling,dose,poisson_noise)
    
    entry.normalize()
    
    entry.create_label(sampling, width = int(.4/sampling))

    image,label=entry.as_tensors()
    entry.reset()
    
    return image,label

data_dir = "dataFCN/"
summary_dir = "summaries/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
net_dir = datetime.now().strftime("%Y%m%d-%H%M%S") + "FCN.pt"

data = load(data_dir)

batch_size = 1
image_size = (256,256) # spatial dimensions of input/output
image_features = 1 # depth of input data
num_classes = 2 # number of predicted class labels
kernel_num = 32 # number at the first level
num_epochs = 50 # number of training epochs
loss_type = 'cross_entropy' # mse or cross_entropy
optimizer_type = 'SDG' # Adam or SDG
nonlinearity = 'softmax' # sigmoid or softmax
weight_decay = 0.0001 # weight decay scale

num_iterations = num_epochs*data.num_examples//batch_size

show_examples(data, image_size, n=3)


# # Neural network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from netfcn import Net

net = Net(num_classes+1) # Including background
print(net)

train_acc, train_loss = [], []
test_acc, test_loss = [], []
valid_acc, valid_loss = [], []
losses = []

if loss_type == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()
    
if optimizer_type == 'Adam':
    optimizer = optim.Adam(net.parameters()) 
else:
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)

# # Test the forward pass with dummy data
x = np.random.normal(0,1, (3, 1, 256, 256)).astype('float32')
out = net(Variable(torch.from_numpy(x)))
print(out.size())
print(out)

data_testandvalid,data_train = data.split(int(data.num_examples*0.7))
data_test,data_valid = data_testandvalid.split(int(data_testandvalid.num_examples*0.5))

num_samples_train = data_train.num_examples
num_batches_train = num_samples_train // batch_size
num_samples_valid = data_valid.num_examples
num_batches_valid = num_samples_valid // batch_size

for epoch in range(num_epochs):
    # # Forward -> Backprob -> Update params
    # # Train
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        image,label = next_example(data_train,image_size)
        output = net(Variable(torch.from_numpy(image)).unsqueeze_(0).unsqueeze_(0))
        
        # # compute gradients given loss
        target_batch = Variable(torch.from_numpy(label)).long()
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        cur_loss += batch_loss   
    losses.append(cur_loss / batch_size)

    net.eval()
    # # Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        image,label = next_example(data_train,image_size)
        output = net(Variable(torch.from_numpy(image)).unsqueeze_(0).unsqueeze_(0))
        
        preds = torch.max(output, 1)[1]
        
        train_targs += list(label.flatten())
        train_preds += list(preds.data.numpy().flatten())
    
    # # Evaluate validation
    val_preds, val_targs = [], []
    for i in range(num_batches_valid):
        image,label = next_example(data_train,image_size)
        
        output = net(Variable(torch.from_numpy(image)).unsqueeze_(0).unsqueeze_(0))
        preds = torch.max(output, 1)[1]
        val_targs += list(label.flatten())
        val_preds += list(preds.data.numpy().flatten())
        

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)
    
    print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
        epoch+1, losses[-1], train_acc_cur, valid_acc_cur))


epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Accucary','Validation Accuracy'])
plt.xlabel('Updates'), plt.ylabel('Acc')

torch.save(net,net_dir)

# class_total = list(0. for i in range(num_classes))
# class_correct = list(0. for i in range(num_classes))
# c = []

# # # # Testing only test dataset
# for i in range(data_test.num_examples):
#     image,label = next_example(data_test,image_size)
    
#     outputs = net(image)
#     _, predicted = torch.max(outputs.data, 1)
#     c.append((predicted == label).squeeze())

# for i in range(len(c)):
#     _,ylabel = next_example(data_test,image_size)
#     label = ylabel
#     class_correct[label] += c[i].numpy()
#     class_total[label] += 1

# for i in range(num_classes):
#     print('Accuracy of {:5s} : {:5.2f} %'.format(
#         classes[i], 100 * class_correct[i] / class_total[i]))