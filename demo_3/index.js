/**
* @license
* Copyright 2018 Google LLC. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http:// www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==============================================================================
*/

// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.


function loadMyImages() {
  return tf.randomUniform([100, 224, 224, 3], 0, 64, 'float32')
}

function loadMyStrings() {
  return tf.preprocessing.stringTensor([['hello', 'world'], ['こんにちは', '世界'], ['thanks', 'David']]);
}

function loadPreprocModel() {
  zeroMeanLayer = tf.layers.zeroMean(
    {
       optimizer: new tf.preprocessing.ZeroMeanOptimizer(),
       inputShape: [224, 224, 3]
    });

  unitVarLayer = tf.layers.unitVariance(
    {
      optimizer: new tf.preprocessing.UnitVarianceOptimizer()
    });

  return tf.sequential({ layers: [zeroMeanLayer, unitVarLayer] });
}
