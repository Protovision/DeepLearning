#!/usr/bin/env th

require "Classifier"

cmd = torch.CmdLine()
cmd:text( "Classifier Experiment" )
cmd:text( "Options: " )
cmd:option( "-maxEpochs", 1/0, "Number of epochs to perform" )
cmd:option( "-maxSeconds", 1/0, "Stop before next epoch if experiment runs over maxSeconds" )
cmd:option( "-earlyStop", false, "Stop if validation loss get higher" )
cmd:option( "-learningRateAccel", 0, "Increase learning rate by this percentage if loss gets lower" )
cmd:option( "-learningRateDecel", 0, "Decrease learning rate by this perentage if loss gets higher" )
cmd:option( "-targets", "", "Ranges of classifications to use" )
cmd:option( "-loadDatabase", "", "File to load database from" )
cmd:option( "-loadModel", "", "File to load model from" )
cmd:option( "-loadWeights", "", "File to load weights from" )
cmd:option( "-saveWeights", "", "File to save weights to" )
cmd:option( "-saveResults", "", "File to save results to" )
cmd:option( "-epochsPerSave", 10, "Epochs to run before saving" )
cmd:option( "-epochsPerTest", 5, "Epochs to train before each test" )
cmd:option( "-cuda", true, "Enable CUDA acceleration" )
cmd:option( "-batchSize", 10, "Batch size" )
cmd:option( "-testBatchSize", 100, "Batch size for validation" )
cmd:option( "-learningRate", 0.01, "Learning rate" )
cmd:option( "-learningRateDecay", 0, "Learning rate decay" )
cmd:option( "-momentum", 0, "Momentum" )
cmd:option( "-weightDecay", 0.00005, "Weight decay" )
cmd:option( "-weightInit", "", "Weight Initialization method" )

params = cmd:parse( arg )

print( "Initializing classifier..." )
classifier = Classifier()
classifier.trainBatchSize = params.batchSize
classifier.testBatchSize = params.testBatchSize
classifier.learningRate = params.learningRate
classifier.learningRateDecay = params.learningRateDecay
classifier.momentum = params.momentum
classifier.weightDecay = params.weightDecay
print( "Loading neural network..." )
model = require( params.loadModel )
if params.loadWeights ~= "" then
	savedWeights = torch.load( params.loadWeights )
	weights, _ = model:parameters()
	for i = 1, #weights, 1 do
		weights[i]:copy( savedWeights[i] )
	end
	savedWeights = nil
end
if params.weightInit ~= "" and params.loadWeights == "" then
	classifier:setModel( require("w-init")(model, params.weightInit) )
else
	classifier:setModel( model )
end
classifier:setCUDA( params.cuda )
print( "Opening database..." )
targets = Array.ranges( params.targets )
database = require( params.loadDatabase )
print( "Loading training set..." )
trainingSet = Dataset.selectTargets( database:TrainingSet(), targets, true )
print( "Loading validation set..." )
validationSet = Dataset.selectTargets( database:ValidationSet(), targets, true )

collectgarbage()
results = {}
results.summary = {
	totalTests = 0,
	totalTrainTime = 0,
	totalTrainTestTime = 0,
	totalValTestTime = 0,
	peakTrainAcc = 0,
	peakValAcc = 0
}
results.epochs = {}
results.config = {
	batchSize = params.batchSize,
	testBatchSize = params.testBatchSize,
	learningRate = params.learningRate,
	learningRateDecay = params.learningRateDecay,
	momentum = params.momentum,
	weightDecay = params.weightDecay,
	targets = params.targets
}
print( results.config )
results.classifier = {
	model = tostring( classifier:getModel() ),
	numParameters = classifier:getParameters():size(1)
}
print( "Training on " .. results.classifier.numParameters .. " parameters..." )

timer = torch.Timer()
io.write( string.format(
	"%-11s%-11s%-11s%-11s%-11s%-11s%-11s\n",
	"Epoch",
	"TrainTime",
	"ValTime",
	"TrainAcc",
	"ValAcc",
	"TrainLoss",
	"ValLoss"
) )
lowestValLoss = 1/0
for epoch = #results.epochs + 1, params.maxEpochs, 1 do
	if timer:time().real >= params.maxSeconds then
		break
	end
	train = classifier:train( trainingSet )
	results.summary.totalTrainTime = results.summary.totalTrainTime + train.time
	results.summary.avgTrainTime = results.summary.totalTrainTime / epoch
	if epoch % params.epochsPerTest == 0 then
		trainTest = classifier:test( trainingSet )
		valTest = classifier:test( validationSet )
		results.summary.totalTrainTestTime = results.summary.totalTrainTestTime + trainTest.time
		results.summary.totalValTestTime = results.summary.totalValTestTime + valTest.time
		results.summary.totalTests = results.summary.totalTests + 1
		results.summary.avgTrainTestTime = results.summary.totalTrainTestTime / results.summary.totalTests
		results.summary.avgValTestTime = results.summary.totalValTestTime / results.summary.totalTests
		results.summary.finalTrainAcc = trainTest.accuracy
		results.summary.finalValAcc = valTest.accuracy
		results.summary.finalTrainLoss = trainTest.loss
		results.summary.finalValLoss = valTest.loss
	else
		trainTest = nil
		valTest = nil
	end
	results.summary.totalTime = timer:time().real
	io.write( string.format(
		"%-11d%-11.4f%-11.4f%-11.4f%-11.4f%-11.6f%-11.6f\n",
		epoch,
		train.time,
		(trainTest and trainTest.time + valTest.time) or 0/0,
		(trainTest and trainTest.accuracy) or 0/0,
		(valTest and valTest.accuracy) or 0/0,
		(trainTest and trainTest.loss) or train.loss,
		(valTest and valTest.loss) or 0/0
	) )
	table.insert( results.epochs, {
		epoch = epoch,
		train = train,
		trainTest = trainTest,
		valTest = valTest
	} )
	if trainTest ~= nil then
		if trainTest.accuracy > results.summary.peakTrainAcc then
			results.summary.peakTrainAcc = trainTest.accuracy
		end
	end
	if valTest ~= nil then
		if valTest.accuracy >= results.summary.peakValAcc then
			results.summary.peakValAcc = valTest.accuracy
		end
		if valTest.loss <= lowestValLoss then
			lowestValLoss = valTest.loss
			if params.earlyStop == true or params.learningRateDecel > 0 then
				print( "Backing up parameters..." )
				classifier:backupParameters()
			end
			if params.learningRateAccel > 0 then
				classifier.learningRate = classifier.learningRate + (classifier.learningRate * params.learningRateAccel)
				print( "Learning rate increased to " .. classifier.learningRate )
			end
		else
			if params.earlyStop == true then
				print( "Early stop. Restoring parameters..." )
				classifier:restoreParameters()
				stopNow = true
			elseif params.learningRateDecel > 0 then
				print( "Restoring parameters..." )
				classifier:restoreParameters()
				classifier.learningRate = classifier.learningRate - (classifier.learningRate * params.learningRateDecel)
				print( "Learning rate set to " .. classifier.learningRate )
			end
		end
	end
	if (epoch % params.epochsPerSave == 0) or (timer:time().real >= params.maxSeconds) or (stopNow == true) then
		if params.saveResults ~= "" then
			print( "Saving results..." )
			torch.save( params.saveResults, results )
			print( "Results saved." )
		end
		if params.saveWeights ~= "" then
			print( "Saving parameters..." )
			parameters, _ = classifier:getModel():parameters()
			torch.save( params.saveWeights, parameters )
			print( "Parameters saved." )
		end
		if timer:time().real >= params.maxSeconds or stopNow == true then
			break
		end
		io.write( string.format(
			"%-11s%-11s%-11s%-11s%-11s%-11s%-11s\n",
			"Epoch",
			"TrainTime",
			"ValTime",
			"TrainAcc",
			"ValAcc",
			"TrainLoss",
			"ValLoss"
		) )
	end
end


