--
-- User: peyman
-- Date: 12/1/16
-- Time: 12:39 PM
-- To change this template use File | Settings | File Templates.
--

--
-- User: peyman
-- Date: 11/2/16
-- Time: 10:10 AM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'image'

-- number of channels in input (RGB=3):
local inputSize = opt.inputPixels[1]

local ok,savedNet = pcall(torch.load, paths.concat(opt.save, opt.network))

if not ok then

    print('==> building model')

    local features = nn.Sequential()

    -- convolutional and pooling layers
    local depth = 1
    local finalChannelSize
    for i = 1, #opt.channelSize do
        if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
            features:add(nn.SpatialDropout(opt.dropoutProb[depth]))
        end
        --SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
        features:add(nn.SpatialConvolution(inputSize, opt.channelSize[i],
            opt.kernelSize[i], opt.kernelSize[i],
            opt.kernelStride[i], opt.kernelStride[i],
            math.floor(opt.kernelSize[i]/2), math.floor(opt.kernelSize[i]/2)))

        if opt.batchNorm then
            features:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
        end

        features:add(nn[opt.activation]())

        if opt.poolSize[i] and opt.poolSize[i] > 0 then
            features:add(nn.SpatialMaxPooling(opt.poolSize[i], opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i]))
        end
        inputSize = opt.channelSize[i]
        finalChannelSize = opt.channelSize[i]
        depth = depth + 1
    end
    -- features = 64,16,16
    local model
    local classifier = nn.Sequential()
    local sample = torch.rand(1, opt.inputPixels[1], opt.inputPixels[2],
                              opt.inputPixels[3])
    local temp = features:forward(sample) --to get batcsize x 64,16,16
    if opt.convOnly then
        classifer:add(nn.SpatialConvolution(opt.channelSize[#opt.channelSize],
                                            nClasses, 1, 1))
        classifier:add(nn[opt.activation]())
        local poolHeight = temp:size(3)
        local poolWidth = temp:size(4)
        classifier:add(nn.SpatialMaxPooling(poolWidth, poolHeight))
        classifier:add(nn.Reshape(nClasses))
        classifier:add(nn.LogSoftMax())
        model = nn.Sequential():add(features):add(classifier)
    else
        local linearFeats = temp:nElement() -- batchsize x 64 x 16 x 16 = batchsize * 16384

        -- 1.3. Create Classifier (fully connected layers)

        classifier:add(nn.Reshape(linearFeats)) -- 16 x 64 x 64 = 65536
        classifier:add(nn.Dropout(0.5))
        classifier:add(nn.Linear(linearFeats, opt.hiddensize1))
        classifier:add(nn[opt.activation]())

        classifier:add(nn.Dropout(0.5))
        classifier:add(nn.Linear(opt.hiddensize1, opt.hiddensize2))
        classifier:add(nn[opt.activation]())

        classifier:add(nn.Linear(opt.hiddensize2, nClasses))
        classifier:add(nn.LogSoftMax())

        -- 1.4. Combine 1.2 and 1.3 to produce final model
        model = nn.Sequential():add(features):add(classifier)
    end

    -- define model + params
    net={
            model = model,
            inputPixels = opt.inputPixels,
            labelPixels = opt.labelPixels,
            channels = opt.channels,
            channelSize = opt.channelSize,
            kernelSize = opt.kernelSize,
            kernelStride = opt.kernelStride,
            poolSize = opt.poolSize,
            poolStride = opt.poolStride,
            dropout = opt.dropout,
            dropoutProb = opt.dropoutProb,
            batchNorm = opt.batchNorm,
            activation =opt.activation,
            loss = opt.loss,
            bceThresh = opt.bceThresh,
            labelThresh = opt.labelThresh,
            convOnly = opt.convOnly
        }

else
    print('==> loading model')
    net = savedNet
end

if not opt.silent then
    print(net.model)
end

if opt.gpu then
    net.model:cuda()
end

collectgarbage()
