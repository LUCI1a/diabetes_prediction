from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('diabetes.csv')
data_tensor = torch.tensor(data.values, dtype=torch.float32)
total_input_data = data_tensor[0:700, 0:8]
print(total_input_data.shape)
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1 = nn.Linear(8, 10)
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
model = Net()
model.load_state_dict(torch.load('model1.pth'))
model.eval()
hidden_layer_output = None
total_norms = torch.zeros(8)
for row in total_input_data:
  row=row.reshape(1, 8)
  print(row.shape)
  with torch.no_grad():
    def get_output_hook(module, input, output):
      global f_output
      f_output = output
    hook = model.fn1.register_forward_hook(get_output_hook)
    output = model(row)
    output_with_no_noise=f_output
    perturbation = 0.01
    perturbed_tensor = torch.zeros(8, 8)
    for i in range(8):
        perturbed_tensor[i] = row
        perturbed_tensor[i, i] += perturbation*perturbed_tensor[i, i]
    hook.remove()
    hook = model.fn1.register_forward_hook(get_output_hook)
    output = model(perturbed_tensor)
    output_with_noise = f_output
    # print (output_with_no_noise)
    # print (output_with_noise)

  norms = torch.zeros(8)
  for i in range(8):
    diff = output_with_noise[i] - output_with_no_noise
    norms[i] = torch.norm(diff, p=2)
  total_norms+=norms
total_norms = total_norms/700
labels = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
plt.figure(figsize=(8, 6))
plt.bar(range(8), total_norms.numpy(), color='b')
plt.xlabel('Element Index')
plt.ylabel('sensitive index')
plt.title('S1 in first hidden layer')
plt.xticks(range(8), labels, rotation=45, ha='right')
plt.tight_layout()
plt.show()
