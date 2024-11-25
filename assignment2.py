from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# control constants
PLOT = False


##########################          Task a)         ########################## 


# split data into training set and test set
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=302)

# split training set into proper training set and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, random_state=17)

median_income = X[:, 0]
house_age = X[:, 1]
ave_rooms = X[:, 2]
ave_bedrms = X[:, 3]
population = X[:, 4]
ave_occup = X[:, 5]
latitude = X[:, 6]
longitude = X[:, 7]


if(PLOT):
    # Boxplots
    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(median_income, patch_artist = True, notch ='True', vert = 0)
    plt.title('Median Income Values Boxplot')
    plt.ylabel('MedInc')
    plt.savefig('median_income_boxplot.png', dpi=300)

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(house_age, patch_artist = True, notch ='True', vert = 0)
    plt.title('House Age Values Boxplot')
    plt.ylabel('HouseAge')
    plt.savefig('house_age_boxplot.png', dpi=300)

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(ave_rooms, patch_artist = True, notch ='True', vert = 0)
    plt.title('Average Rooms Values Boxplot')
    plt.ylabel('AveRooms')
    plt.savefig('average_rooms_boxplot.png')

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(ave_bedrms, patch_artist = True, notch ='True', vert = 0)
    plt.title('Average Bedrooms Values Boxplot')
    plt.ylabel('AveBedrms')
    plt.savefig('average_bedrooms_boxplot.png')

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(population, patch_artist = True, notch ='True', vert = 0)
    plt.title('Population Values Boxplot')
    plt.ylabel('Population')
    plt.savefig('population_boxplot.png', dpi=300)

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(ave_occup, patch_artist = True, notch ='True', vert = 0)
    plt.title('Average Occupants Values Boxplot')
    plt.ylabel('AveOccup')
    plt.savefig('average_occupants_boxplot.png', dpi=300)

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(latitude, patch_artist = True, notch ='True', vert = 0)
    plt.title('Latitude Values Boxplot')
    plt.ylabel('Latitude')
    plt.savefig('latitude_boxplot.png')

    fig = plt.figure(figsize =(10, 10))
    plt.boxplot(longitude, patch_artist = True, notch ='True', vert = 0)
    plt.title('Latitude Values Boxplot')
    plt.ylabel('Longitude')
    plt.savefig('longitude_boxplot.png')


##########################          Task b)         ########################## 


# normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

# convert data to tensor
X_train_tensor = torch.tensor(X_train_normalized).float()
y_train_tensor = torch.tensor(y_train).float()
X_val_tensor = torch.tensor(X_val_normalized).float()
y_val_tensor = torch.tensor(y_val).float()

# create mini batches
dataset_training = TensorDataset(X_train_tensor, y_train_tensor) 
batch_size = 1
train_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)

# mean squared error function
def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


class NeuralNet(nn.Module):
    def __init__(self):
      super(NeuralNet, self).__init__()
      self.fc1 = nn.Linear(in_features=8, out_features=64) 
      self.fc2 = nn.Linear(in_features=64, out_features=64)
      self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
    

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.001
num_epochs = 10

model = NeuralNet()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(num_epochs):
    print(f'Epoch number {epoch}')

    # train with mini batches
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        outputs = model(data)
        loss = loss_fn(outputs, target)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\nAverage train loss in epoch {epoch}: {np.mean(losses[-len(train_loader):])}')


    # validate without batches
    model.eval()
    valdiation_loss = 0
    
    with torch.no_grad():
        for data, target in zip(X_val_tensor, y_val_tensor):
            output = model(data)
            valdiation_loss += loss_fn(output, target)

    valdiation_loss /= len(X_val_normalized)
    print(f'Validation set: Average loss: {valdiation_loss}')
 

