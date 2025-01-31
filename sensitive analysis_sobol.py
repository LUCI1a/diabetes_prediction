from __future__ import print_function
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('diabetes.csv')
data_tensor = torch.tensor(data.values, dtype=torch.float32)
original_data = data_tensor[0:700, 0:8].numpy()
min_vals = original_data.min(axis=0)
max_vals = original_data.max(axis=0)
problem = {
    'num_vars': 8,
    'names': ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
    # 'names': ['Беременности','Глюкоза','Артериальное давление','Толщина кожной складки','Инсулин','ИМТ','Наследственная предрасположенность к диабету','Возраст'],
    'bounds': [[min_vals[0], max_vals[0]],
               [min_vals[1], max_vals[1]],
               [min_vals[2], max_vals[2]],
               [min_vals[3], max_vals[3]],
               [min_vals[4], max_vals[4]],
               [min_vals[5], max_vals[5]],
               [min_vals[6], max_vals[6]],
               [min_vals[7], max_vals[7]],]
}
# problem = {
#     'num_vars': 8,
#     'names': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
#     'bounds': [[min_val, max_val] for min_val, max_val in zip(min_vals, max_vals)]
# }

param_values = saltelli.sample(problem, 65536)
inputs = torch.Tensor(param_values)
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
Y = np.zeros(param_values.shape[0])
model.eval()

with torch.no_grad():
    for i, sample in enumerate(param_values):
        input_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        Y[i] = model(input_tensor).item()
Si = sobol.analyze(problem, Y)
Si['S2'][Si['S2'] < 0] = 0
print(Si['S1'])  # 一阶敏感度
print(Si['ST'])  # 总敏感度
print(Si['S2'])  # 二阶敏感度

plt.figure(figsize=(12, 6))
x = np.arange(len(problem["names"]))
plt.bar(x - 0.2, Si['S1'], width=0.4, label='First-order Sensitivity (S1)', color='b', alpha=0.7)
plt.bar(x + 0.2, Si['ST'], width=0.4, label='Total Sensitivity (ST)', color='r', alpha=0.7)
plt.xlabel('Input Variables')
plt.ylabel('Sensitivity Index')
plt.title('Sensitivity Analysis: S1 and ST')
plt.xticks(x, problem["names"], rotation=45)
plt.legend()
plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(Si['S2'], annot=True, fmt=".2f", cmap='viridis',
            xticklabels=problem["names"], yticklabels=problem["names"])
plt.title('Second-order Sensitivity Analysis (S2)')
plt.xlabel('Input Variables')
plt.ylabel('Input Variables')
plt.savefig('second_order_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()





#置信区间的生成
# from __future__ import print_function
# import numpy as np
# import pandas as pd
# from SALib.sample import saltelli
# from SALib.analyze import sobol
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('diabetes.csv')
# data_tensor = torch.tensor(data.values, dtype=torch.float32)
# original_data = data_tensor[0:700, 0:8].numpy()
# min_vals = original_data.min(axis=0)
# max_vals = original_data.max(axis=0)
#
# problem = {
#     'num_vars': 8,
#     'names': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age'],
#     'bounds': [[min_vals[0], max_vals[0]],
#                [min_vals[1], max_vals[1]],
#                [min_vals[2], max_vals[2]],
#                [min_vals[3], max_vals[3]],
#                [min_vals[4], max_vals[4]],
#                [min_vals[5], max_vals[5]],
#                [min_vals[6], max_vals[6]],
#                [min_vals[7], max_vals[7]]]
# }
#
# # 采样点数
# sample_sizes = [128, 256, 512, 1024, 2048, 4096, 8192,16384,32768, 65536]
# all_S1 = {name: [] for name in problem['names']}
# all_ST = {name: [] for name in problem['names']}
# all_S1_conf = {name: [] for name in problem['names']}
# all_ST_conf = {name: [] for name in problem['names']}
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fn1 = nn.Linear(8, 10)
#         self.bn1 = nn.BatchNorm1d(10)
#         self.fn2 = nn.Linear(10, 1)
#         self.bn2 = nn.BatchNorm1d(1)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.fn1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.fn2(out)
#         out = self.bn2(out)
#         out = torch.sigmoid(out)
#         return out
#
# model = Net()
# model.load_state_dict(torch.load('model1.pth'))
# model.eval()
#
# for N in sample_sizes:
#     param_values = saltelli.sample(problem, N)
#     Y = np.zeros(param_values.shape[0])
#     with torch.no_grad():
#         for i, sample in enumerate(param_values):
#             input_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
#             Y[i] = model(input_tensor).item()
#     Si = sobol.analyze(problem, Y)
#
#     for name, s1, st, s1_conf, st_conf in zip(problem['names'], Si['S1'], Si['ST'], Si['S1_conf'], Si['ST_conf']):
#         all_S1[name].append(s1)
#         all_ST[name].append(st)
#         all_S1_conf[name].append(s1_conf)
#         all_ST_conf[name].append(st_conf)
#
# plt.figure(figsize=(12, 8))
#
# for idx, name in enumerate(problem['names']):
#     plt.subplot(2, 4, idx + 1)
#     #S1
#     plt.plot(sample_sizes, all_S1[name], 'o-', label=f'S1 ({name})', alpha=0.7)
#     plt.fill_between(sample_sizes, np.array(all_S1[name]) - np.array(all_S1_conf[name]),
#                      np.array(all_S1[name]) + np.array(all_S1_conf[name]), color='blue', alpha=0.2)
#     #ST
#     plt.plot(sample_sizes, all_ST[name], 'o-', label=f'ST ({name})', alpha=0.7)
#     plt.fill_between(sample_sizes, np.array(all_ST[name]) - np.array(all_ST_conf[name]),
#                      np.array(all_ST[name]) + np.array(all_ST_conf[name]), color='red', alpha=0.2)
#
#     plt.xlabel('Sample Size (N)')
#     plt.ylabel('Sensitivity Index')
#     plt.title(f'{name}')
#     plt.legend(loc='best')
#
# plt.tight_layout()
# plt.savefig('parameter_sensitivity_vs_N_with_confidence.png', dpi=300, bbox_inches='tight')
# plt.show()








