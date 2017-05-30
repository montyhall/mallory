--
-- User: peyman
-- Date: 12/1/16
-- Time: 3:32 PM
-- To change this template use File | Settings | File Templates.
--




require 'image'
require 'paths'
require 'xlua'
require 'pl.stringx'
require 'lfs'
posix = require 'posix'

require 'nn'

opt = dofile('opts.lua').parse(arg)

function run(batchSize,numClasses,bconvTrick)
    local bsize = batchSize or 1
    local nClasses = numClasses or 10
    local convTrick = convTrick or false
    --[[
    --nn.Sequential {
          [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
          (1): nn.SpatialConvolutionMM(3 -> 8, 3x3, 1,1, 1,1)
          (2): nn.SpatialBatchNormalization
          (3): nn.ReLU
          (4): nn.SpatialMaxPooling(2x2, 2,2)
          (5): nn.SpatialConvolutionMM(8 -> 16, 3x3, 1,1, 1,1)
          (6): nn.SpatialBatchNormalization
          (7): nn.ReLU
          (8): nn.SpatialMaxPooling(2x2, 2,2)
          (9): nn.SpatialConvolutionMM(16 -> 2, 7x5, 1,1, 3,2)
          (10): nn.Transpose
          (11): nn.Reshape(-1x2)
          (12): nn.LogSoftMax
        }
     ]]
    input = torch.randn(bsize,3,256,256)

    local features = nn.Sequential()
    features:add(nn.SpatialConvolution(3, 8, 3,3, 1,1, 1,1)) --10,8,256,256
    features:add(nn.SpatialBatchNormalization(8)) --10,8,256,256
    features:add(nn.ReLU()) --10,8,256,256
    features:add(nn.SpatialMaxPooling(2,2,2,2)) --10,8,128,128
    features:add(nn.SpatialConvolution(8, 16, 3,3, 1,1, 1,1)) --10,16,128,128
    features:add(nn.SpatialBatchNormalization(16)) --10,16,128,128
    features:add(nn.ReLU()) --10,16,128,128
    features:add(nn.SpatialMaxPooling(2,2,2,2)) --10,16,64,64

    local classifier = nn.Sequential()

    if convTrick then
        --if NLL
        features:add(nn.SpatialConvolution(16, 2, 7, 5, 1, 1, 3, 2)) --10,2,64,64
        features:add(nn.Transpose({ 2, 3 }, { 3, 4 })) --10,64,64,2
        features:add(nn.Reshape(-1,2,false)) -- 4096,2
        -- features:add(nn.LogSoftMax()) -- 4096,2

        classifier:add(nn.View(4096*2))
        classifier:add(nn.Linear(4096*2, 3072))
    else
        classifier:add(nn.View(16*64*64))
        classifier:add(nn.Linear(16*64*64, 3072))
    end

    classifier:add(nn.Threshold(0, 1e-6))

    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(3072, 4096))
    classifier:add(nn.Threshold(0, 1e-6))

    classifier:add(nn.Linear(4096, nClasses))
    classifier:add(nn.LogSoftMax())

    -- 1.4. Combine 1.2 and 1.3 to produce final model
    local model = nn.Sequential():add(features):add(classifier)

    print('prediction')
    print(model:forward(input))
end

function t()
    dataTable={}
    dataTable.input = {}
    dataTable.class = {}
    setmetatable(dataTable,
        {__index = function(t, i)
            return {t.input[i], t.class[i]}
        end}
    )
    for i=1, 389 do
        table.insert(dataTable.input,torch.rand(2000))
        table.insert(dataTable.class,torch.random(1,10))
    end

    function dataTable:size() return 389 end

    print(dataTable[1])

    mod = nn.Sequential()
    mod:add(nn.Linear(2000, 1000))
    mod:add(nn.Tanh())
    mod:add(nn.Linear(1000, 500))
    mod:add(nn.Tanh())
    mod:add(nn.Linear(500, 10))
    mod:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    trainer = nn.StochasticGradient(mod, criterion)
    trainer.learningRate = 0.01
    trainer.maxIteration = 2

    trainer:train(dataTable)


    correct =0
    for i=1,10 do
        local groundtruth = dataTable.class[i]
        local prediction = mod:forward(dataTable.input[i])

        local confidences, indices = torch.sort(prediction, true)
        if groundtruth == indices[1] then
            correct = correct + 1
        else
            print("Incorrect! "..tostring(i))
        end
        print("Input size: ",dataTable.input[i]:size())
        print("predictions: ",prediction)
        print("Correct pred: "..correct)
    end

    prediction =  mod:forward(dataTable.input[100])
    print(prediction)
end

-- understanding ClassNLLCriterion
function classes()
    data = torch.Tensor( 16, 10 ):bernoulli() --16 numbers represented in 4 bits
    class = torch.Tensor( 16, 1 ):bernoulli():add(1)

    network = nn.Sequential()
    network:add( nn.Linear( 10, 8 ) )
    network:add( nn.ReLU() )
    network:add( nn.Linear( 8, 2 ) )
    network:add( nn.LogSoftMax() )

    criterion = nn.ClassNLLCriterion()

    print(data)
    print()
    print(class:t())

    for i = 1, 10000 do
        prediction = network:forward( data )
        -- nn.ClassNLLCriterion expects target to be a 1D tensor of size batch_size or a scalar.
        -- class is a 2D one (batch_size x 1) but class[i] is 1D, that's why
        -- non-batch version will work.

        loss = criterion:forward( prediction, class )

        network:zeroGradParameters()

        grad = criterion:backward( prediction, class )
        network:backward( data, grad )

        network:updateParameters( 0.1 )
    end
    --test = torch.Tensor( 1, 10 ):bernoulli()
    test = data[1]
    prediction = network:forward(test)
    local confidences, indices = torch.sort(prediction, true)

    print('test',test)
    print('prediction',prediction)
    print('confidences',confidences)
    print('indices',indices)

end
