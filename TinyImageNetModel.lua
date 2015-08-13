require "cunn"
require "cudnn"

BigImageNetModel = function()
	local model
	model = nn.Sequential()
	model:add( nn.Reshape(3,64,64) )

	model:add( cudnn.SpatialConvolution(3,64,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(64,64,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(64,128,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(128,128,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(128,256,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(256,256,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(256,256,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(256,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(512,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(512,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(512,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(512,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	model:add( cudnn.SpatialConvolution(512,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( nn.Reshape(512*2*2) )

	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(512*2*2, 2048) )
	model:add( cudnn.ReLU(true) )

	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(2048, 2048) )
	model:add( cudnn.ReLU(true) )

	model:add( nn.Linear(2048, 200) )
	model:add( nn.LogSoftMax() )

	return model
end

ImageNetModel = function()
	local model
	model = nn.Sequential()
	model:add( nn.Reshape(3,64,64) )

	model:add( cudnn.SpatialConvolution(3,32,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(32,64,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(64,128,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(128,256,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )

	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( cudnn.SpatialConvolution(256,512,3,3,1,1,1,1) )
	model:add( cudnn.ReLU(true) )
	
	model:add( cudnn.SpatialMaxPooling(2,2,2,2) )

	model:add( nn.Reshape(512*2*2) )

	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(512*2*2, 2048) )
	model:add( cudnn.ReLU(true) )

	model:add( nn.Dropout(0.5) )
	model:add( nn.Linear(2048, 4096) )
	model:add( cudnn.ReLU(true) )

	model:add( nn.Linear(4096, 200) )
	model:add( nn.LogSoftMax() )

	return model
end
return BigImageNetModel()

