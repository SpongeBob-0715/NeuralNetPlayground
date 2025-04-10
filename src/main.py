import argparse
from src.data.data_loader import CIFAR10Loader
from src.models.neural_net import ThreeLayerNet
from src.utils.training import Trainer, parameter_search
from src.utils.visualization import ModelVisualizer

def main():
    parser = argparse.ArgumentParser(description="Deep Learning HW1")
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--test', action='store_true', help="Run testing")
    parser.add_argument('--param_search', action='store_true', help="Run parameter search")
    parser.add_argument('--draw', action='store_true', help="Visualize model parameters")
    args = parser.parse_args()

    dataset = CIFAR10Loader.load_dataset([f"data_batch_{i}" for i in range(1, 6)])
    X_train, y_train = dataset['train_X'], dataset['train_y']
    X_test, y_test = dataset['test_X'], dataset['test_y']

    if args.train:
        model = ThreeLayerNet(X_train.shape[1], 126, 10)
        trainer = Trainer(model, X_train, y_train, X_test, y_test)
        trainer.train(hidden_size=126, learning_rate=0.01, reg=0.01, dropout_rate=0)

    elif args.test:
        model = ThreeLayerNet(X_train.shape[1], 126, 10)
        trainer = Trainer(model, X_train, y_train, X_test, y_test)
        model.params = np.load('/data/dell/lty/作业/作业/深度学习/hw1/best/cv_best_model.npz')
        acc = trainer.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")

    elif args.param_search:
        parameter_search(X_train, y_train, X_test, y_test)

    elif args.draw:
        visualizer = ModelVisualizer('/data/dell/lty/作业/作业/深度学习/hw1/best/cv_best_model.npz',
                                   X_test, y_test)
        visualizer.visualize_all()

if __name__ == "__main__":
    main()
    