#!/usr/bin/env th

r = torch.load( arg[1] )

io.write( string.format("%11s%11s%11s%11s%11s\n",
	"Epoch", "TrainAcc", "ValAcc", "TrainLoss", "ValLoss") )
for i = 1, #r.epochs, 1 do
	if r.epochs[i].trainTest ~= nil and r.epochs[i].valTest ~= nil then
		epoch = r.epochs[i]
		io.write( string.format("%11d%11.4f%11.4f%11.6f%11.6f\n",
			i,
			epoch.trainTest.accuracy,
			epoch.valTest.accuracy,
			epoch.trainTest.loss,
			epoch.valTest.loss)
		)
	end
end
