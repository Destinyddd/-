from dataload import read_images,read_labels
from model import *
from config import config
from train import optimize_model_parameters
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读数据
X_train = read_images("data/train-images-idx3-ubyte")
y_train = read_labels("data/train-labels-idx1-ubyte")
y_train = encode_one_hot(y_train.T,10)
X_test = read_images("data/t10k-images-idx3-ubyte")
y_test = read_labels("data/t10k-labels-idx1-ubyte")
y_test = encode_one_hot(y_test.T,10)

# 读取超参
input_dim = config.input_dim
hidden_dim = config.hidden_dim
output_dim = config.output_dim
learning_rate = config.learning_rate
decay_rate = config.decay_rate
beta = config.beta
batch_size = config.batch_size
num_epochs = config.num_epochs
activation = config.activation

# 训练模型并保存最好的模型
model = setup_model(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim)
model, best_model, history_train_losses, history_train_accuracies, history_val_losses, history_val_accuracies = optimize_model_parameters(model, X_train, y_train, X_test, y_test, learning_rate, decay_rate, beta, num_epochs, batch_size,activation)
store_model_parameters(best_model,"best_model")

# 准确率和损失绘制
#fig, ax = plt.subplots(2, 1, figsize=(12, 8))
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history_train_losses, label='Training loss')
ax1.plot(history_val_losses, label='Testing loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(history_train_accuracies, label='Training accuracy')
ax2.plot(history_val_accuracies, label='Testing accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('myplot.png')