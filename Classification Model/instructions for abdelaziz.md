# Node.js NEO Classifier Implementation Guide

This guide assumes you have the following folder structure:
```
project_root/
├── NEOsClassification_Model/
│   └── tfjs_model/
├── scaler_params.json
├── class_labels.json
└── your_nodejs_files/
```

## Setup

1. Navigate to your project root and initialize a new Node.js project (if not already done):
   ```
   npm init -y
   ```

2. Install required dependencies:
   ```
   npm install @tensorflow/tfjs-node
   ```

## Implementation

1. Create `neoClassifier.js` in your Node.js files directory:

```javascript
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.join(__dirname, '..');

async function loadModel() {
  const modelPath = path.join(PROJECT_ROOT, 'NEOsClassification_Model', 'tfjs_model', 'model.json');
  return await tf.loadLayersModel(`file://${modelPath}`);
}

function preprocessData(inputData) {
  const scalerPath = path.join(PROJECT_ROOT, 'scaler_params.json');
  const scaler = JSON.parse(fs.readFileSync(scalerPath, 'utf8'));
  return inputData.map(row => 
    row.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i])
  );
}

async function classifyNEOs(inputData) {
  try {
    const model = await loadModel();
    const processedData = tf.tensor2d(preprocessData(inputData));
    
    const predictions = model.predict(processedData);
    const classes = predictions.argMax(1).dataSync();
    
    const labelsPath = path.join(PROJECT_ROOT, 'class_labels.json');
    const classLabels = JSON.parse(fs.readFileSync(labelsPath, 'utf8'));
    return classes.map(c => classLabels[c]);
  } catch (error) {
    console.error('Classification error:', error);
    throw error;
  }
}

module.exports = { classifyNEOs };
```

2. Create `testModel.js` in the same directory:

```javascript
const { classifyNEOs } = require('./neoClassifier');

async function test() {
  const testData = [
  // Example of a small, fast-rotating NEO
  [22.1, 0.15, 0.140, 0.20, 2.5, 0.22, 1.5, 1.17, 5.2, 180.4, 318.9, 340.2, 0.531, 680.5, 0.05],
  
  // Example of a larger, slower-rotating NEO
  [17.3, 0.10, 1.2, 0.08, 15.7, 0.35, 2.2, 1.43, 12.8, 210.3, 28.7, 15.8, 0.439, 823.9, 0.15]
];

  try {
    const results = await classifyNEOs(testData);
    console.log('Predictions:', results);
  } catch (error) {
    console.error('Test failed:', error);
  }
}

test();
```

## Usage

1. Run the test script from your project root:
   ```
   node your_nodejs_files/testModel.js
   ```

2. To use in your application, import and use the `classifyNEOs` function:

```javascript
const { classifyNEOs } = require('./neoClassifier');

async function yourFunction(neoData) {
  try {
    const classifications = await classifyNEOs(neoData);
    // Process classifications...
  } catch (error) {
    // Handle error...
  }
}
```

## Input Format

The `classifyNEOs` function expects an array of arrays, where each inner array represents a single NEO and contains 15 numerical features in this order:

[H, G, diameter, albedo, rot_per, e, a, q, i, om, w, ma, n, per, moid]

Ensure your input data follows this format for accurate predictions.

## Notes for the Backend Developer

1. The `classifyNEOs` function is the main entry point for classification. It handles model loading, data preprocessing, and prediction.

2. The implementation uses relative paths to locate the model files and JSON files. Ensure the folder structure is maintained as described at the top of this guide.

3. The `PROJECT_ROOT` constant is used to correctly reference files in the root directory. Make sure your Node.js files are in a subdirectory of the project root for this to work correctly.

4. Error handling is included, but you may want to adjust it based on your application's specific needs.

5. The test script provides a basic example of how to use the `classifyNEOs` function. You can expand on this for more comprehensive testing or integration into your API routes.

6. If you need to deploy this model, ensure that the entire `NEOsClassification_Model` folder, along with `scaler_params.json` and `class_labels.json` in the root directory, are included in your deployment package.

7. If you encounter any "file not found" errors, double-check the paths and ensure all files are in their correct locations.