from __future__ import print_function
import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv('diabetes.csv')
data_tensor = torch.tensor(data.values, dtype=torch.float32)
x_train = data_tensor[0:500, 0:8]
y_train = data_tensor[0:500, 8]
y_train = y_train.reshape((500,1))
x_test = data_tensor[500:700, 0:8]
y_test = data_tensor[500:700, 8]
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
num_epochs = 2000
input_size = 8
model = Net()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=0.001)
for epoch in range(num_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
print("Training finished!")
torch.save(model.state_dict(), 'model1.pth')
torch.save(model, 'model.pth')
with torch.no_grad():  
    output = model(x_test)
    predicted_class = (output >= 0.5).float()
    predicted_class.reshape((200,1))
    predicted_class = predicted_class.numpy()
    y_test = y_test.numpy()
    error = predicted_class - y_test
    error=(error*error).mean()
    print(f'Error: {1-error}')






