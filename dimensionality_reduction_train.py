from __future__ import print_function
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('diabetes_last4.csv')  #we can choose top4 or last4 data
data_tensor = torch.tensor(data.values, dtype=torch.float32)
x_train = data_tensor[0:500, 0:4]
y_train = data_tensor[0:500, 4]
y_train = y_train.reshape((500,1))
x_test = data_tensor[500:700, 0:4]
y_test = data_tensor[500:700, 4]
y_test = y_test.reshape((200,1))
print(x_train.shape)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1 = nn.Linear(input_size, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fn2 = nn.Linear(10, 1)
        self.bn2 = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fn1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fn2(out)
        out = self.bn2(out)
        out = torch.sigmoid(out)
        return out
learn_rate = 0.01
num_epochs = 2500
input_size = 4
model = Net()
criterion = torch.nn.BCELoss() #二元交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2200, gamma=0.1)
for epoch in range(num_epochs):
    outputs = model(x_train)
    outputs_test = model(x_test)
    loss = criterion(outputs, y_train)
    loss_test = criterion(outputs_test, y_test)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Loss_test: {loss_test.item():.4f}')
print("Training finished!")
torch.save(model.state_dict(), 'model_temp.pth')
with torch.no_grad():
    output_test = model(x_test)
    output_train = model(x_train)
    predicted_class_test = (output_test >= 0.5).float()
    predicted_class_train = (output_train >= 0.5).float()
    predicted_class_test.reshape((200,1))
    predicted_class_train.reshape((500, 1))
    predicted_class_test = predicted_class_test.numpy()
    predicted_class_train = predicted_class_train.numpy()
    y_test = y_test.numpy()
    y_train=y_train.numpy()
    error_test = predicted_class_test- y_test
    error_test=(error_test*error_test).mean()
    error_train = predicted_class_train - y_train
    error_train = (error_train * error_train).mean()
    print(f'Error_train: {1 - error_train}')
    print(f'Error_test: {1 - error_test}')