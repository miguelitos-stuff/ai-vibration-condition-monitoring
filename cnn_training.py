from cnn_architecture import CNN
import torch
import torch.nn as nn
from torch import optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size =
num_classes =
learning_rate =
batch_size =
num_epochs =

# Load Data
train_dataset = #import data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = #Import data
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialze network
model = CNN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# adapt optimizer if needed

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print(training data)
    else:
        print(test data)


    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct}/{num_samples} correct samples with an accuracy of {(num_correct/num_samples)*100}%')
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


