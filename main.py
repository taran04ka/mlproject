from data.data_loader import DataLoader
from models.model_builder import ModelBuilder
from utils.evaluation import Evaluator
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def main():
    if tf.test.is_gpu_available():
        print("GPU available")
    else:
        print("GPU not available")
    tf.config.list_physical_devices('GPU')

    data_loader = DataLoader()
    X_train, X_val, y_train, y_val, test_images, test_labels = data_loader.load_data()
    model_builder = ModelBuilder()

    model = model_builder.build_model()
    history = model_builder.train_model(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=128)
    evaluator = Evaluator()
    evaluator.plot_training_history(history)
    evaluator.evaluate_model(model, test_images, test_labels)
    model.save('fashion_mnist_model.h5')

    pruned_model = model_builder.build_pruned_model()
    pruned_model.summary()
    pruned_model.fit(X_train, y_train, batch_size=128, epochs=3, validation_data=(X_val, y_val),
                     callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
    pruned_model.save('pruned_fashion_mnist_model.h5')

    pruned_history = model_builder.train_model(pruned_model, X_train, y_train, X_val, y_val, epochs=3, batch_size=128)
    evaluator = Evaluator()
    evaluator.plot_training_history(pruned_history)
    evaluator.evaluate_model(pruned_model, test_images, test_labels)
    pruned_model.save('fine_tuned_fashion_mnist_model.h5')

if __name__ == '__main__':
    main()
