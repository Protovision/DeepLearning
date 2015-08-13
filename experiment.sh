#!/bin/bash
mkdir -p "experiments"
loadDatabase="TinyImageNet"
loadModel="TinyImageNetModel"
loadWeights="experiments/TinyImageNetWeights4.table"
saveWeights="experiments/TinyImageNetWeights5.table"
saveResults="experiments/TinyImageNetResults5.table"
targets="1-200"
batchSize="100"
testBatchSize="256"
learningRate="0.001"
weightDecay="0.0009"
momentum="0.9"
maxEpochs="40"
epochsPerSave="5"
epochsPerTest="3"

./run.lua -loadDatabase "$loadDatabase" -loadModel "$loadModel" -loadWeights "$loadWeights" -saveWeights "$saveWeights" -targets "$targets" -batchSize "$batchSize" -testBatchSize "$testBatchSize" -learningRate "$learningRate" -weightDecay "$weightDecay" -momentum "$momentum" -maxEpochs "$maxEpochs" -epochsPerSave "$epochsPerSave" -epochsPerTest "$epochsPerTest"

