import matplotlib.pyplot as plt
import seaborn as sns
from model import retrieve_model_parameters
# Generate heatmap for weights of a layer
def render_heatmap(weights_matrix, layer_label, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights_matrix, annot=False, cmap='viridis', center=0)
    plt.title(f"Weight Heatmap for Layer {layer_label}")
    plt.xlabel("Input Neurons")
    plt.ylabel("Output Neurons")
    plt.savefig(save_path)
    plt.show()

# Generate histogram for weights of a layer
def render_histogram(weights_matrix, layer_label, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(weights_matrix.flatten(), bins=30, alpha=0.75, color='blue')
    plt.title(f"Weight Distribution Histogram for Layer {layer_label}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# Generate a bar chart for biases of a layer
def render_bias_chart(biases, layer_label, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(biases)), biases, color='magenta')
    plt.title(f"Bias Values for Layer {layer_label}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Bias Value")
    plt.savefig(save_path)
    plt.show()

# Load models and plot parameter visualizations
relu_model = retrieve_model_parameters("best_model_relu")
sigmoid_model = retrieve_model_parameters("best_model_sigmoid")

# Heatmaps
render_heatmap(relu_model["W1"], "W1", "output/relu/heatmap/W1.png")
render_heatmap(sigmoid_model["W1"], "W1", "output/sigmoid/heatmap/W1.png")

render_heatmap(relu_model["W2"], "W2", "output/relu/heatmap/W2.png")
render_heatmap(sigmoid_model["W2"], "W2", "output/sigmoid/heatmap/W2.png")

render_heatmap(relu_model["W3"], "W3", "output/relu/heatmap/W3.png")
render_heatmap(sigmoid_model["W3"], "W3", "output/sigmoid/heatmap/W3.png")

# Histograms
render_histogram(relu_model["W1"], "W1", "output/relu/histogram/W1.png")
render_histogram(sigmoid_model["W1"], "W1", "output/sigmoid/histogram/W1.png")

render_histogram(relu_model["W2"], "W2", "output/relu/histogram/W2.png")
render_histogram(sigmoid_model["W2"], "W2", "output/sigmoid/histogram/W2.png")

render_histogram(relu_model["W3"], "W3", "output/relu/histogram/W3.png")
render_histogram(sigmoid_model["W3"], "W3", "output/sigmoid/histogram/W3.png")

# Bias charts
render_bias_chart(relu_model["b1"], "b1", "output/relu/biases/b1.png")
render_bias_chart(sigmoid_model["b1"], "b1", "output/sigmoid/biases/b1.png")

render_bias_chart(relu_model["b2"], "b2", "output/relu/biases/b2.png")
render_bias_chart(sigmoid_model["b2"], "b2", "output/sigmoid/biases/b2.png")

render_bias_chart(relu_model["b3"], "b3", "output/relu/biases/b3.png")
render_bias_chart(sigmoid_model["b3"], "b3", "output/sigmoid/biases/b3.png")