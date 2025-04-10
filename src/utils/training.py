import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.neural_net import ThreeLayerNet

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, output_dir='/data/dell/lty/作业/作业/深度学习/hw1/best'):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.output_dir = output_dir
        self.best_params = None
        self.best_val_acc = 0.0

    def compute_loss(self, X, y, reg):
        probs = self.model.forward(X)
        data_loss = -np.log(probs[range(len(y)), y]).mean()
        reg_loss = 0.5 * reg * (np.sum(self.model.params['W1']**2) + np.sum(self.model.params['W2']**2))
        return data_loss + reg_loss

    def evaluate(self, X, y):
        self.model.set_training(False)
        probs = self.model.forward(X)
        predictions = np.argmax(probs, axis=1)
        self.model.set_training(True)
        return np.mean(predictions == y)

    def train(self, hidden_size=124, activation='relu', reg=0.01, learning_rate=1e-3,
              epochs=1000, batch_size=200, lr_decay=0.9, lr_decay_every=5,
              early_stop_step=20, dropout_rate=0.2, trial_id=0):
        
        self.model.set_training(True)
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        recorded_epochs = []
        no_improvement_count = 0
        
        for epoch in range(epochs):
            if epoch % lr_decay_every == 0 and epoch > 0:
                learning_rate *= lr_decay
                
            shuffle_idx = np.random.permutation(self.X_train.shape[0])
            X_shuffled = self.X_train[shuffle_idx]
            y_shuffled = self.y_train[shuffle_idx]
            
            for i in range(0, self.X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                probs = self.model.forward(X_batch)
                grads = self.model.backward(y_batch, reg)
                for param in self.model.params:
                    self.model.params[param] -= learning_rate * grads[param]
            
            val_acc = self.evaluate(self.X_val, self.y_val)
            print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                train_loss = self.compute_loss(self.X_train, self.y_train, reg)
                val_loss = self.compute_loss(self.X_val, self.y_val, reg)
                train_acc = self.evaluate(self.X_train, self.y_train)
                
                train_loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                recorded_epochs.append(epoch + 1)
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                no_improvement_count = 0
                self.best_val_acc = val_acc
                self.best_params = {k: v.copy() for k, v in self.model.params.items()}
                np.savez(f'{self.output_dir}/best_model.npz',
                         **self.best_params, hidden_size=hidden_size, activation=activation,
                         dropout_rate=dropout_rate)
            else:
                no_improvement_count += 1
                if no_improvement_count > early_stop_step:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        self._plot_training_curves(recorded_epochs, train_loss_history, val_loss_history,
                                 train_acc_history, val_acc_history, trial_id)
        return self.best_params, self.best_val_acc

    def _plot_training_curves(self, epochs, train_loss, val_loss, train_acc, val_acc, trial_id):
        sns.set_style("whitegrid")
        plt.style.use('ggplot')
        plt.rcParams.update({
            'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
            'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 8
        })

        plt.figure(num=trial_id, figsize=(12, 4), facecolor='white')
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'o-', color='#E24A33', label='Train Loss')
        plt.plot(epochs, val_loss, 's--', color='#348ABD', label='Validation Loss')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Loss', fontweight='bold')
        plt.title('Training and Validation Loss', pad=20)
        plt.legend(framealpha=0.8, edgecolor='black')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'o-', color='#8EBA42', label='Train Accuracy')
        plt.plot(epochs, val_acc, 's--', color='#988ED5', label='Validation Accuracy')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('Training and Validation Accuracy', pad=20)
        plt.legend(framealpha=0.8, edgecolor='black')

        plt.tight_layout(pad=3)
        plt.savefig(f'{self.output_dir}/training_curves_{trial_id}.png', dpi=300, bbox_inches='tight')
        plt.close()

def parameter_search(X_train, y_train, X_val, y_val):
    param_grid = {
        'hidden_size': [256, 512, 1024],
        'learning_rate': [0.1, 0.01, 0.001],
        'reg': [0.01, 0.1],
        'dropout_rate': [0.2, 0.5],
        'activation': ['relu', 'sigmoid']
    }
    
    best_acc = 0
    best_params = {}
    trial_id = 0
    
    for values in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), values))
        print(f"Trial {trial_id}: {current_params}")
        
        model = ThreeLayerNet(X_train.shape[1], current_params['hidden_size'], 10,
                            current_params['activation'], current_params['dropout_rate'])
        trainer = Trainer(model, X_train, y_train, X_val, y_val)
        model_params, val_acc = trainer.train(**current_params, trial_id=trial_id)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = current_params.copy()
            np.savez('/data/dell/lty/作业/作业/深度学习/hw1/best/cv_best_model.npz',
                     **model_params, hidden_size=best_params['hidden_size'])
        
        trial_id += 1
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_acc}")
    return best_params, best_acc