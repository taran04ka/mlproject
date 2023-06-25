let model;

async function loadModel() {
    console.log('Loading model...');
    model = await tf.loadLayersModel('../tfjs_model/model.json');
    console.log('Model loaded successfully.');
}  

// Add canvas functionality
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    const x = e.offsetX;
    const y = e.offsetY;
    ctx.fillStyle = '#000';
    ctx.fillRect(x, y, 1, 1);
}

function stopDrawing() {
    isDrawing = false;
}

loadModel();

document.getElementById('classify').addEventListener('click', async () => {
    const imageData = ctx.getImageData(0, 0, 1, 1);
    const inputTensor = preprocessImage(imageData);
    const prediction = model.predict(inputTensor);
    const predictedClass = prediction.argMax(-1).dataSync()[0];
    document.getElementById('prediction').innerText = `Predicted class: ${predictedClass}`;
});

function preprocessImage(imageData) {
    const tensor = tf.browser.fromPixels(imageData, 1)
        .toFloat()
        .resizeNearestNeighbor([28, 28])
        .expandDims();

    return tensor.div(255.0);
}
