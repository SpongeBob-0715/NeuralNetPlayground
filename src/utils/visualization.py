import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from src.models.neural_net import ThreeLayerNet

class ModelVisualizer:
    CLASS_NAMES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, model_file, X_test, y_test, output_dir='/data/dell/lty/作业/作业/深度学习/hw1/pic'):
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        data = np.load(model_file)
        self.params = {k: data[k] for k in data.files if k not in ['hidden_size', 'activation', 'dropout_rate']}
        self.hidden_size = data['hidden_size'].item()
        self.activation = data.get('activation', 'relu').item()
        
        self.model = ThreeLayerNet(X_test.shape[1], self.hidden_size, 10, self.activation)
        self.model.params = self.params
        self.model.set_training(False)

    def visualize_layer1_weights(self):
        plt.figure(figsize=(16, 8))
        plt.suptitle('First Layer Weight Visualizations', y=0.95)
        W1 = self.params['W1']
        
        for i in range(min(32, self.hidden_size)):
            plt.subplot(4, 8, i+1)
            weights = W1[:, i].reshape(32, 32, 3)
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            plt.imshow(weights)
            plt.axis('off')
            plt.title(f'Neuron {i+1}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/layer1_weights.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_layer2_weights(self):
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.params['W2'].T, cmap='coolwarm', center=0,
                   xticklabels=range(1, self.hidden_size+1, max(1, self.hidden_size//10)),
                   yticklabels=self.CLASS_NAMES)
        plt.xlabel('Hidden Neuron Index')
        plt.ylabel('CIFAR-10 Classes')
        plt.title('Second Layer Weight Matrix')
        plt.savefig(f'{self.output_dir}/layer2_weights_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_biases(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(self.params['b1'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Hidden Layer Bias Distribution')
        
        plt.subplot(1, 2, 2)
        plt.bar(self.CLASS_NAMES, self.params['b2'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Output Layer Biases per Class')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bias_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_activations(self):
        sample_idx = np.random.choice(len(self.X_test))
        self.model.forward(self.X_test[sample_idx:sample_idx+1])
        activations = self.model.a1
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(self.hidden_size), activations.squeeze(), color='purple', alpha=0.7)
        plt.xlabel('Hidden Neuron Index')
        plt.ylabel('Activation Strength')
        plt.title(f'Hidden Activations for Sample {sample_idx}\nTrue Class: {self.CLASS_NAMES[self.y_test[sample_idx]]}')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'{self.output_dir}/sample_activations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_tsne(self):
        print("Computing t-SNE embeddings...")
        sample_indices = np.random.choice(len(self.X_test), min(500, len(self.X_test)), replace=False)
        hidden_reps = [self.model.forward(self.X_test[idx:idx+1])[0] for idx in sample_indices]
        labels = self.y_test[sample_indices]
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings = tsne.fit_transform(np.array(hidden_reps))
        
        plt.figure(figsize=(10, 8))
        for c in range(10):
            idx = labels == c
            plt.scatter(embeddings[idx, 0], embeddings[idx, 1],
                       label=self.CLASS_NAMES[c], alpha=0.7, s=40)
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('t-SNE of Hidden Layer Representations')
        plt.grid(alpha=0.2)
        plt.savefig(f'{self.output_dir}/tsne_hidden.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_all(self):
        self.visualize_layer1_weights()
        self.visualize_layer2_weights()
        self.visualize_biases()
        self.visualize_activations()
        self.visualize_tsne()
        print(f"Visualizations saved to {self.output_dir}/")