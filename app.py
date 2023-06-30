from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from io import BytesIO

from data.data_loader import DataLoader
from models.model_builder import ModelBuilder
from utils.evaluation import Evaluator

app = Flask(__name__)

data_loader = DataLoader()
model_builder = ModelBuilder()
evaluator = Evaluator()

# Load the model
model = model_builder.build_model()
model.load_weights('fashion_mnist_model.h5')

# Load the pruned model
pruned_model = model_builder.build_pruned_model()
pruned_model.load_weights('pruned_fashion_mnist_model.h5')

# # Load the fine-tuned model
# fine_tuned_model = model_builder.build_model()
# fine_tuned_model.load_weights('fine_tuned_fashion_mnist_model.h5')


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        image = Image.open(BytesIO(photo.read()))
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        pruned_predictions = pruned_model.predict(image_array)
        # fine_tuned_predictions = fine_tuned_model.predict(image_array)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        result = {
            'original': class_names[np.argmax(predictions)],
            'pruned': class_names[np.argmax(pruned_predictions)],
            # 'fine_tuned': class_names[np.argmax(fine_tuned_predictions)]
        }
        return render_template('result.html', result=result)
    return 'No file uploaded.'

if __name__ == '__main__':
    app.run(debug=True, port=8000)
