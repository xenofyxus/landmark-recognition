import torch
import matplotlib.pyplot as plt
import numpy as np
import Net2 as initializeNet
import torch.optim as optim
import time
import os
import Dataset as ds
import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_directory_path = 'models/'
model_path = model_directory_path + '/test1.pt'

batch_size = 32
epochs = 30
learning_rate = 0.001
momentum = 0.9


# This is the two-step process used to prepare the
# data for use with the convolutional neural network.

# First step is to convert Python Image Library (PIL) format
# to PyTorch tensors.

# Second step is used to normalize the data by specifying a 
# mean and standard deviation for each of the three channels.
# This will convert the data from [0,1] to [-1,1]

# Normalization of data should help speed up conversion and
# reduce the chance of vanishing gradients with certain 
# activation functions.


#Load the dataset
GoogleLandmarksDataset = ds.GoogleLandmarks('data/labels2.csv')

train_size = int(0.60 * len(GoogleLandmarksDataset))
rest_size = len(GoogleLandmarksDataset) - train_size
train_dataset, rest_dataset = torch.utils.data.random_split(GoogleLandmarksDataset, [train_size, rest_size])

test_size = int(0.50 * len(rest_dataset))
valid_size = len(rest_dataset) - test_size
test_dataset, valid_dataset = torch.utils.data.random_split(rest_dataset, [test_size, valid_size])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

classes = np.arange(99)

# Initialize network with structure specified in Net.py 
net = initializeNet.Net2()

#use GPU if available
if torch.cuda.is_available():
    net.to(device)

# Defining the loss function used
criterion = torch.nn.CrossEntropyLoss().to(device)

# Defining the optimizer used
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

if not os.path.exists(model_directory_path):
    os.makedirs(model_directory_path)

if os.path.isfile(model_path):
    # load trained model parameters from disk
    net.load_state_dict(torch.load(model_path))
    print('Loaded model parameters from disk.')
else:
    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
			
            #use GPU if available
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 1000 mini-batches
                print(time.time()-start)
                net.lossOverTime.append(running_loss / 100)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        '''
        total_correct = 0
        total_images = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        model_accuracy = total_correct / total_images * 100
        net.model_accuracy.append(model_accuracy)
        '''
    print('Finished Training.')
    torch.save(net.state_dict(), model_path)
    print('Saved model parameters to disk.')
    plt.plot(net.lossOverTime)
    plt.savefig('loss.png', bbox_inches='tight')
    '''
    plt.plot(net.model_accuracy)
    plt.savefig('accuracy.png', bbox_inches='tight')
    '''



'''
# Load four images from the test set
dataiter = iter(test_loader)
images, labels = dataiter.next()

fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))

for idx, image in enumerate(images):
    print("trying to add image")
    image = image / 2 + 0.5
    image = image.numpy()
    axes[idx].imshow(image.transpose(1, 2, 0))
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

# Show four images from test set with corresponding labels
plt.show()

images = images.to(device)
# Get predictions for the four images
outputs = net(images)

sm = torch.nn.Softmax(dim=1)
sm_outputs = sm(outputs)

probs, index = torch.max(sm_outputs, dim=1)



for p, i in zip(probs, index):
    print('{0} - {1:.4f}'.format(classes[i], p))

# Calculate and print model for 10000 images from the test set
total_correct = 0
total_images = 0
confusion_matrix = np.zeros([99, 99], int)
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_images += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        for i, l in enumerate(labels):
            confusion_matrix[l.item(), predicted[i].item()] += 1 

model_accuracy = total_correct / total_images * 100
print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))


# Computing the confusion matrix for the test set.
print('{0:99s} - {1}'.format('Category', 'Accuracy'))
for i, r in enumerate(confusion_matrix):
    print('{0:99d} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))

# Visualize the confusion matrix for the test set.
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
plt.ylabel('Actual Category')
plt.yticks(range(99), classes)
plt.xlabel('Predicted Category')
plt.xticks(range(99), classes)
plt.show()

'''


