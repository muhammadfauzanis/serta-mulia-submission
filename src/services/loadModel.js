const tf = require('@tensorflow/tfjs-node');

async function loadModel() {
  return await tf.loadGraphModel(process.env.MODEL_URL);
}

module.exports = loadModel;
