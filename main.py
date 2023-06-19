# main.py
from data.data_loader import load_data
from models.model_builder import build_model, train_model
from utils.evaluation import evaluate_model, plot_training_history

def main():
    X_train, X_val, y_train, y_val, test_images, test_labels = load_data()
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128)
    plot_training_history(history)
    evaluate_model(model, test_images, test_labels)
    model.save('fashion_mnist_model.h5')

if __name__ == '__main__':
    main()
