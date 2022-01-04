
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

ReluNet = [0.044848754, 0.044077049, 0.216295385]  # 0.810611179843544]

LinReg = [0.0389291506260633, 0.0401787117123603, 0.214463660120964]  # 0.753881526738405]

PolyNet = [0.028300693, 0.031916652, 0.220772359]  # 0.749413399]

#tread = []

plt.figure(figsize=[14.8, 9.6])

xlabels = ['Housing - Price', 'Housing - Price + NOX', 'Forest Fires'] #, 'cross_1', 'cross_2']

x = np.arange(len(xlabels))

width = 0.2
dist = width+0.05

cmap = get_cmap(name='viridis')

p1 = plt.bar(x - dist, ReluNet, width, label='ReluNet', color=cmap(0.1))
p2 = plt.bar(x, LinReg, width, label='LinReg', color=cmap(0.45))
p3 = plt.bar(x + dist, PolyNet, width, label='PolyNet', color=cmap(0.85))

# put rounded average mse loss on top of the bars
for idx, rect in enumerate(p1):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, 1.005 * height, round(ReluNet[idx], 3), fontsize=16, ha='center', va='bottom')

for idx, rect in enumerate(p2):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, 1.005 * height, round(LinReg[idx], 3), fontsize=16, ha='center', va='bottom')

for idx, rect in enumerate(p3):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, 1.005 * height, round(PolyNet[idx], 3), fontsize=16, ha='center', va='bottom')

# use log scale
plt.yscale('log')
plt.ylabel('Avg MSE-Loss', fontsize=20)
plt.xticks(x, xlabels, fontsize=20)
# plt.title('', fontsize=26)
plt.legend(fontsize=20, loc='upper left')
# save figure?
plt.savefig('models_barplots_v4_log.png', dpi=300)
# plt.show()
