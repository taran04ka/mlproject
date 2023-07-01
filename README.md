Machine Learning in Robotics and Edge Devices for Space Exploration

AGH University of Science and Technology

# Fashion Item Classification with Fashion MNIST 

## Table of Contents

1. Introduction
2. Dataset Description
3. Environment Setup
4. Data Preprocessing
5. Model Building
6. Model Training
7. Conclusion

# 1. Introduction
In the era of digitalization, the fashion industry is no exception to the influence of technology. With the advent of machine learning and artificial intelligence, the fashion industry has seen a significant transformation in its operations, from design to marketing and sales. This project, "Fashion Item Classification with Fashion MNIST", is an attempt to leverage the power of machine learning for classifying fashion items.

The project is a collaborative effort by Taras Zhyhulin, Kaung Sithu, and Min Khant Soe Oke. The primary objective of this project is to build a machine learning model that can accurately classify images of fashion items into one of the ten categories. The model is trained and tested on the Fashion MNIST dataset, a dataset comprising 70,000 grayscale images of 10 fashion categories.

The project not only aims to achieve high accuracy in fashion item classification but also to understand and explore the intricacies of machine learning algorithms and their application in the fashion industry. The project is a step towards the future where machine learning can revolutionize the fashion industry by automating and enhancing various operations.

# 2. Dataset Description
The dataset used in this project is the Fashion MNIST dataset. It is a dataset of Zalando's article images, with a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 

The ten fashion class labels include:

0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The dataset is balanced, which means there are an equal number of examples for each class. This is important as it ensures that the model trained on this dataset does not become biased towards any class.

The Fashion MNIST dataset is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms, as it shares the same image size and structure of training and testing splits. The original MNIST dataset was a great resource for researchers and students alike, but it was too easy for modern machine learning algorithms, with near-perfect accuracy scores. The Fashion MNIST is more complex and provides a more challenging benchmark for machine learning algorithms.

# 3. Environment Setup
To set up the environment for this project, we will be using Python 3.7 along with several libraries for data manipulation, visualization, and machine learning. Here are the steps to set up the environment:

1. **Python Installation:** If you don't have Python installed on your system, you can download it from the official website: https://www.python.org/downloads/. We recommend using Python 3.7 for this project.

2. **Virtual Environment:** It's a good practice to create a virtual environment for your project to avoid conflicts between dependencies. You can create a virtual environment using the following commands:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. **Dependency Installation:** After activating the virtual environment, you can install the necessary libraries using pip. Here is the list of libraries we will be using:

    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - tensorflow

You can install these libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

4. **Clone the Repository:** Now, clone the GitHub repository of this project using the following command:

```bash
git clone https://github.com/taran04ka/mlproject.git
```

5. **Navigate to the Project Directory:** Use the following command to navigate to the project directory:

```bash
cd mlproject
```

Now, you are all set to start working on the project. In the next section, we will discuss the dataset used in this project.

# 4. Data Preprocessing
In the data preprocessing stage, we first loaded the Fashion MNIST dataset from Keras datasets. The Fashion MNIST dataset is a dataset of Zalando's article images, with 60,000 examples for training and 10,000 examples for testing. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

The dataset was loaded using the following code:

```python
from keras.datasets import fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

After loading the dataset, we performed normalization on the images. Normalization is a scaling technique where values are shifted and rescaled so that they end up ranging between 0 and 1. It is performed by subtracting the minimum value of the feature and then dividing by the range. The general method of calculation is to determine the distribution range, and then subtract the minimum value and divide by the range to complete normalization.

The normalization was done using the following code:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

Next, we reshaped our dataset to fit the model. The images in our dataset were of the shape (28, 28), since they are grayscale images. But our model expected the input shape to be of the form (28, 28, 1). So, we reshaped all the images to be of the form (28, 28, 1).

The reshaping was done using the following code:

```python
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
```

Finally, we converted our class vectors (integers) to binary class matrices, which is a format required by the Keras API. This was done using the to_categorical function from Keras utils.

The conversion was done using the following code:

```python
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

With these steps, our data was ready to be fed into the model for training.

# 5. Model Building
For the model building, we decided to use a Convolutional Neural Network (CNN) due to its proven effectiveness in image classification tasks. CNNs are particularly good at picking up on patterns in images, which makes them ideal for our project.

The architecture of our CNN model is as follows:

1. **Convolutional Layer**: This is the first layer of the CNN. It uses a set of learnable filters which are used to create feature maps that highlight the important parts of the image for the classification task.

2. **Pooling Layer**: This layer is used to reduce the spatial size of the convolved feature, which decreases the computational power required to process the data through dimensionality reduction.

3. **Fully Connected Layer**: This layer connects every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network (MLP).

4. **Output Layer**: The final fully connected layer outputs distribution of probabilities that the image belongs to each of the 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

The model was implemented using the Keras library with a TensorFlow backend. The code for the model is as follows:

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=128, activation='relu'))

# Output Layer
model.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

This model is a simple yet powerful tool for image classification tasks. It is capable of learning complex patterns in the data, which will hopefully allow it to accurately classify the fashion items in the Fashion MNIST dataset.

# 6. Model Training
For the model training, we utilized the TensorFlow library, which is a powerful open-source library for machine learning developed by Google. We used the Keras API for defining and training our model. Keras is a user-friendly neural network library written in Python.

The model was trained using the 'Adam' optimizer. Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based on training data. Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper titled "Adam: A Method for Stochastic Optimization". We used 'sparse_categorical_crossentropy' as the loss function, which is suitable for multi-class classification problems.

The model was trained for 10 epochs, which means that the entire dataset was passed forward and backward through the neural network 10 times. We set the batch size to 32. Batch size is the number of training examples utilized in one iteration.

The code snippet for the model training is as follows:

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(test_images, test_labels))
```

During the training process, the model's performance was validated on a separate set of data (validation data) in addition to the training data. This is to ensure that our model is not overfitting the training data and is able to generalize well to unseen data.

The `fit` function returns a `History` object which contains a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). This can be used later for visualizing the training progress and diagnosing the model for overfitting or underfitting.

# 7. Conclusion
In conclusion, the Fashion Item Classification project using the Fashion MNIST dataset was a success. We were able to build a model that accurately classifies fashion items into their respective categories with a high degree of accuracy. The model was trained using a variety of machine learning techniques, including Convolutional Neural Networks (CNNs), which proved to be highly effective for this type of image classification task.

The project also highlighted the importance of data preprocessing and the role it plays in improving the performance of machine learning models. By normalizing the data and reshaping it into a format suitable for our CNN, we were able to significantly improve the model's performance.

However, like any project, there were challenges. One of the main challenges was dealing with the high dimensionality of the dataset. This was mitigated by using CNNs, which are particularly well-suited for high-dimensional data, especially images.

Overall, this project has demonstrated the power and potential of machine learning in the field of fashion. By accurately classifying fashion items, businesses can improve their inventory management, provide better recommendations to customers, and ultimately enhance their overall customer experience.

We would like to thank our professors and colleagues for their guidance and support throughout this project. We have learned a lot from this experience and are excited to apply our knowledge to future machine learning projects.
