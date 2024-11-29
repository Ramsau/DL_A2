from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from threading import Thread


##########################          Task a)         ########################## 

def get_data():
    # split data into training set and test set
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=302)

    # split test set into proper test set and validation set (evenly)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=17)

    return X, X_train, X_val, X_test, y, y_train, y_val, y_test


def plot_data(X):
    median_income = X[:, 0]
    house_age = X[:, 1]
    ave_rooms = X[:, 2]
    ave_bedrms = X[:, 3]
    population = X[:, 4]
    ave_occup = X[:, 5]
    latitude = X[:, 6]
    longitude = X[:, 7]


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

def normalize_data(X_train, X_val, X_test):
    # normalize data based on train set
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_normalized = scaler.transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    X_test_normalized = scaler.transform(X_test)

    return X_train_normalized, X_val_normalized, X_test_normalized

def get_loader(X_train, y_train, batch_size):

    # convert data to tensor
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).float()

    # create mini batches
    dataset_training = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)

    return train_loader

# mean squared error function
def loss_fn(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


class NeuralNet(nn.Module):
    def __init__(self, dimensions):
      super(NeuralNet, self).__init__()

      # build architecture: input and output size are fixed to 8 and 1
      # input: 8 features
      layer_setup = [
          nn.Linear(8, dimensions[0]),
          nn.ReLU(),
      ]
      # build layers dynamically
      for i in range(1, len(dimensions)):
          layer_setup.append(nn.Linear(dimensions[i-1], dimensions[i]))
          layer_setup.append(nn.ReLU())
      # output: 1 feature
      layer_setup.append(nn.Linear(dimensions[-1], 1))

      self.layers = nn.Sequential(
          *layer_setup
      )

    def forward(self,x):
        return self.layers(x)

def train(model,
          optimizer,
          loss_fn,
          train_loader,
          num_epochs,
          X_val_tensor,
          y_val_tensor,
          label=None,
          scheduler=None,
          early_stopping=False,
          patience=10):
    epoch_losses_train = []
    epoch_losses_val = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # try-except block to let user exit learning with Ctrl-C
    try:
        best_validation_loss = float("inf")
        best_model = None
        improved_epoch = 0
        for epoch in range(num_epochs):
            print(f'{label}: Epoch number {epoch}')

            # train with mini batches
            model.train()
            losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                outputs = model(data).flatten()
                loss = loss_fn(outputs, target)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            train_epoch_loss = np.mean(losses)
            print(f'\nAverage train loss in epoch {epoch}: {train_epoch_loss}')
            epoch_losses_train.append(train_epoch_loss)
            if scheduler is not None:
                scheduler.step()


            # validate without batches
            model.eval()
            valdiation_loss = 0

            with torch.no_grad():
                for data, target in zip(X_val_tensor, y_val_tensor):
                    output = model(data)
                    valdiation_loss += loss_fn(output, target)

            valdiation_loss /= len(X_val_tensor)
            print(f'Validation set: Average loss: {valdiation_loss}')
            epoch_losses_val.append(valdiation_loss)

            # early stopping
            if early_stopping:
                if valdiation_loss < best_validation_loss:
                    best_validation_loss = valdiation_loss
                    best_model = model.state_dict()
                    improved_epoch = epoch
                else:
                    if epoch - improved_epoch > patience:
                        model.load_state_dict(best_model)
                        break


            # if epoch % 20 == 0:
            #     plot_losses(epoch_losses_train, epoch_losses_val, label + f" Epoch {epoch}")

    except KeyboardInterrupt:
        pass

    if label is not None:
        plot_losses(epoch_losses_train, epoch_losses_val, label)

        with open(f"output/{label}.npy", "wb+") as f:
            np.save(f, epoch_losses_train)
            np.save(f, epoch_losses_val)

    return epoch_losses_train, epoch_losses_val
 

def plot_losses(epoch_losses_train, epoch_losses_val, title):
    plt.plot(epoch_losses_train, 'r-', label='Training Loss')
    plt.plot(epoch_losses_val, 'b-', label='Validation loss')

    # Add titles and labels
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.savefig(f"output/{title}.png")
    plt.show()


if __name__ == "__main__":
    # ensure deterministic behaviour
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)

    # hyperparameters
    learning_rate = 0.003
    num_epochs = 200
    batch_size = 100
    models = [
        lambda: NeuralNet([1024]),
        lambda: NeuralNet([64, 64]),
        lambda: NeuralNet([64, 64, 64, 64]),
        lambda: NeuralNet([64, 128, 128, 128, 64]),
        lambda: NeuralNet([16] * 10),
    ]
    optimizers = [
        lambda params, learning_rate: torch.optim.SGD(params, lr=learning_rate),
        lambda params, learning_rate: torch.optim.SGD(params, lr=learning_rate, momentum=0.9),
        lambda params, learning_rate: torch.optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov=True),
        lambda params, learning_rate: torch.optim.Adam(params, lr=learning_rate),
        lambda params, learning_rate: torch.optim.RMSprop(params, lr=learning_rate),
    ]
    schedulers = [
        lambda optimizer: StepLR(optimizer, step_size=5, gamma=0.1),
        lambda optimizer: MultiStepLR(optimizer, milestones=[5, 20, 40], gamma=0.1),
        lambda optimizer: CosineAnnealingLR(optimizer, T_max=num_epochs),
        lambda optimizer: CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6),
    ]

    model_selected = 3
    optimizer_selected = 3
    scheduler_selected = 3

    # setup
    X, X_train, X_val, X_test, y, y_train, y_val, y_test = get_data()


    # plot_data(X)

    X_train_normalized, X_val_normalized, X_test_normalized = normalize_data(X_train, X_val, X_test)
    train_loader = get_loader(X_train_normalized, y_train, batch_size)

    X_val_tensor = torch.tensor(X_val_normalized).float()
    y_val_tensor = torch.tensor(y_val).float()
    X_test_tensor = torch.tensor(X_test_normalized).float()
    y_test_tensor = torch.tensor(y_test).float()

    # train all 5 models
    # for model_selected_opt in range(len(models)):
    #     model = models[model_selected_opt]()
    #     losses_train, losses_val = train(
    #             model,
    #             optimizers[optimizer_selected](model.parameters(), learning_rate),
    #             loss_fn,
    #             train_loader,
    #             num_epochs,
    #             X_val_tensor,
    #             y_val_tensor,
    #             f"Model {model_selected_opt}"
    #     )
    #     print(f"Model {model_selected_opt}: {losses_train[-1]}/{losses_val[-1]}")
    '''
    Comparison of models:
    lr = 0.003
    num_epochs = 50
    batch_size = 100
    optimizer = SGD
    
    Final Loss: Train/Validation
    Model 0: 0.3739/0.3998
    Model 1: 0.3924/0.4058
    Model 2: 0.3595/0.3753
    Model 3: 0.3559/0.3667 *
    Model 4: 1.3381/1.3211
    '''

    # for optimizer_selected_opt in range(len(optimizers)):
    #     model = models[model_selected]()
    #     losses_train, losses_val = train(
    #         model,
    #         optimizers[optimizer_selected_opt](model.parameters(), learning_rate),
    #         loss_fn,
    #         train_loader,
    #         num_epochs,
    #         X_val_tensor,
    #         y_val_tensor,
    #         f"Optimizer {optimizer_selected_opt}"
    #     )
    #
    #     print(f"Optimizer {optimizer_selected_opt}: {losses_train[-1]}/{losses_val[-1]}")

    '''
    
    Comparison of optimizers:
    lr = 0.003
    num_epochs = 50
    batch_size = 100
    model = 3
    
    Final Loss: Train/Validation
    Optimizer 0: 0.3545/0.3601
    Optimizer 1: 0.2555/0.2908
    Optimizer 2: 0.2520/0.3014
    Optimizer 3: 0.2053/0.2796 *
    Optimizer 4: 0.1958/0.3469
    '''

    # for learning_rate_opt in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]:
    #     model = models[model_selected]()
    #     losses_train, losses_val = train(
    #         model,
    #         optimizers[optimizer_selected](model.parameters(), learning_rate),
    #         loss_fn,
    #         train_loader,
    #         num_epochs,
    #         X_val_tensor,
    #         y_val_tensor,
    #         f"Learning rate {learning_rate_opt}"
    #     )
    #
    #     print(f"Learning rate {learning_rate_opt}: {losses_train[-1]:.4f}/{losses_val[-1]:.4f}")


    '''
    Comparison of learning rates:
    num_epochs = 50
    batch_size = 100
    model = 3
    optimizer = Adam
    
    Final Loss: Train/Validation
    Learning rate 0.5: 0.2176/0.2945
    Learning rate 0.2: 0.2046/0.2816
    Learning rate 0.1: 0.2030/0.2768
    Learning rate 0.05: 0.2053/0.2796
    Learning rate 0.02: 0.2177/0.2945
    Learning rate 0.01: 0.2046/0.2816
    Learning rate 0.005: 0.2030/0.2768
    Learning rate 0.002: 0.2053/0.2796
    Learning rate 0.001: 0.2012/0.2776
    Learning rate 0.0005: 0.2056/0.2751
    Learning rate 0.0002: 0.2172/0.3007
    Learning rate 0.0001: 0.2322/0.2725
    Learning rate 5e-05: 0.2176/0.2945
    Learning rate 2e-05: 0.2046/0.2816
    Learning rate 1e-05: 0.2029/0.2768
    '''

    # for scheduler_opt in range(len(schedulers)):
    #     model = models[model_selected]()
    #     optimizer = optimizers[optimizer_selected](model.parameters(), learning_rate)
    #     scheduler = schedulers[scheduler_opt](optimizer)
    #     losses_train, losses_val = train(
    #         model,
    #         optimizer,
    #         loss_fn,
    #         train_loader,
    #         num_epochs,
    #         X_val_tensor,
    #         y_val_tensor,
    #         f"Scheduler {scheduler_opt}",
    #         scheduler = scheduler
    #     )
    #
    #     print(f"Scheduler {scheduler_opt}: {losses_train[-1]:.4f}/{losses_val[-1]:.4f}")

    '''
    Comparison of Schedulers:
    num_epochs = 50
    batch_size = 100
    model = 3
    optimizer = Adam
    
    Final Loss: Train/Validation
    Scheduler 0: 0.2871/0.3082
    Scheduler 1: 0.2526/0.2865
    Scheduler 2: 0.1843/0.2709
    Scheduler 3: 0.1988/0.2581 *
    '''

    model = models[model_selected]()
    optimizer = optimizers[optimizer_selected](model.parameters(), learning_rate=learning_rate)
    scheduler = schedulers[scheduler_selected](optimizer)

    train_loader = get_loader(ConcatDataset([X_train_normalized, X_val_normalized]), ConcatDataset([y_train, y_val]), batch_size)


    # You can load the saved model state here
    model.load_state_dict(torch.load("output/Final Weights.npy"))

    # losses_train, losses_val =  train(
    #     model,
    #     optimizer,
    #     loss_fn,
    #     train_loader,
    #     num_epochs,
    #     X_test_tensor,
    #     y_test_tensor,
    #     f"Final Model",
    #     scheduler = scheduler,
    #     early_stopping = True,
    #     patience = 15,
    # )
    # print(f"Final Model: {losses_train[-1]:.4f}/{losses_val[-1]:.4f}" )
    # torch.save(model.state_dict(), "output/Final Weights.npy")

    '''
    Final Model:
    num_epochs = 75 (early stopping)
    batch_size = 100
    model = 3
    optimizer = Adam
    scheduler = Cosine Annealing
    
    Final Loss (Train/Test): 0.1842/0.2486
    '''

    model.eval()
    estimations = model(X_test_tensor).flatten().detach()

    plt.scatter(estimations, y_test)
    plt.xlabel("Estimated House Value")
    plt.ylabel("Actual House Value")
    plt.title("Evaluation of Final Model")

    plt.savefig("output/Evaluation.png")
    plt.show()




##########################          Task f)         ########################## 

# transforming above architecture to binary classification architecture within
# least possible steps

def get_data_binary():
    # split data into training set and test set
    X, y = fetch_california_housing(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=302)
    # assign labels to target variables
    y_train[y_train < 2], y_test[y_test < 2] = 0, 0
    y_train[y_train >= 2], y_test[y_test >= 2] = 1, 1


    return X, X_train, X_test, y, y_train, y_test


class NeuralNetBinary(nn.Module):
    def __init__(self, dimensions):
      super(NeuralNetBinary, self).__init__()

      # build architecture: input and output size are fixed to 8 and 2
      # input: 8 features
      layer_setup = [
          nn.Linear(8, dimensions[0]),
          nn.ReLU(),
      ]
      # build layers dynamically
      for i in range(1, len(dimensions)):
          layer_setup.append(nn.Linear(dimensions[i-1], dimensions[i]))
          layer_setup.append(nn.ReLU())
      # output: Still 1 output feature because of binary classification
      layer_setup.append(nn.Linear(dimensions[-1], 1))

      # added Sigmoid activation function for classification
      layer_setup.append(nn.Sigmoid())
      
      self.layers = nn.Sequential(
          *layer_setup
      )

    def forward(self,x):
        return self.layers(x)

print(f'Training Binary Model now')

# setup
X, X_train, X_test, y, y_train, y_test = get_data_binary()

X_train_normalized, X_val_normalized, X_test_normalized = normalize_data(X_train, X_val, X_test)
train_loader = get_loader(X_train_normalized, y_train, batch_size)

X_test_tensor = torch.tensor(X_test_normalized).float()
y_test_tensor = torch.tensor(y_test).float().unsqueeze(1)


# loss function that is usable for binary classification task
loss_function = nn.BCELoss()

# reconstructing the Final Model from above task

num_epochs = 75
early_stopping = True
model = NeuralNetBinary([64, 128, 128, 128, 64])
optimizer = optimizers[optimizer_selected](model.parameters(), learning_rate=0.003)
label = f'Final Model Binary Classification'
scheduler = schedulers[scheduler_selected](optimizer)
patience = 15

# train single model for  binary classification task using the whole training set and evaluate on the test set
losses_train, losses_val = train(
    model,
    optimizer,
    loss_function,
    train_loader,
    num_epochs,
    X_test_tensor,
    y_test_tensor,
    label,
    scheduler,
    early_stopping,
    patience
)

