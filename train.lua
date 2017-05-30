--
-- User: peyman
-- Date: 12/1/16
-- Time: 1:26 PM
--
----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

if opt.gpu then require 'cunn' end

----------------------------------------------------------------------
-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = net.model:getParameters()

----------------------------------------------------------------------
local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.lrDecay
}
-----------------------------------------------------
-- vars
-----------------------------------------------------
local timer = torch.Timer()
local batchNumber
local loss_epoch

local current_loss=math.huge
local min_loss=math.huge
local current_accuracy,max_accuracy=0,0
local best_valid_epoch=0
local ntrial = 0
local bestModel

local inputsCPU = torch.Tensor(opt.batchSize, opt.inputPixels[1], opt.inputPixels[2], opt.inputPixels[3])
local labelsCPU = torch.LongTensor(opt.batchSize)

local inputs,labels=nil,nil

if opt.gpu then
    inputs = torch.Tensor(opt.batchSize, opt.inputPixels[1], opt.inputPixels[2], opt.inputPixels[3])
    labels = torch.Tensor(opt.batchSize)

    inputs  = inputs:cuda()
    labels  = labels:cuda()

    net.model:cuda()
end

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-----------------------------------------------------
-- Train all batchsize for 1 epoc
-----------------------------------------------------
function train()
    print('\n==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. '/' .. opt.nEpochs)

    -- set the regularizers to training mode
    net.model:training()

    -- wait for the operations to finish
    if opt.gpu then cutorch.synchronize() end

    local tm = torch.Timer()
    loss_epoch = 0
    batchNumber = 0
    confusion:zero()

    -- epochSize = Number of batches per epoch
    for i=1,opt.epochSize do
        -- queue jobs to data-workers
        donkeys:addjob(
        -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels = trainLoader:sample(opt.batchSize)
                local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
                local l_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(labels:storage())))
                inputs:cdata().storage = nil
                labels:cdata().storage = nil
                return i_stg, l_stg
            end,
            -- the end callback (runs in the main thread)
            trainBatch
        )
    end
    donkeys:synchronize()

    if opt.gpu then cutorch.synchronize() end

    loss_epoch = loss_epoch / opt.epochSize

    trainLogger:add{ ['avg loss (train set)'] = loss_epoch }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
            .. 'average loss (per batch): %.2f ', epoch, tm:time().real, loss_epoch))

    confusion:updateValids()
    print(confusion)

    collectgarbage()
end

-----------------------------------------------------
-- train on a single batch after the data is loaded.
-----------------------------------------------------
function trainBatch(dataPointer, labelPointer)

        if opt.gpu then cutorch.synchronize() end

        timer:reset()

        -- set the data and labels to the main thread tensor buffers (free any existing storage)
        setFloatStorage(inputsCPU, dataPointer)
        setLongStorage(labelsCPU, labelPointer)

        -- transfer over to GPU
        if opt.gpu then
            inputs:copy(inputsCPU)
            labels:copy(labelsCPU)
        else
            inputs=inputsCPU
            labels = labelsCPU
        end

        if opt.gpu then cutorch.synchronize() end

        -- create closure to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            dE_dw:zero()
            local E,y,dE_dy

            -- evaluate function for complete mini batch
            y = net.model:forward(inputs)
            E = loss:forward(y,labels)

            -- estimate df/dW
            dE_dy = loss:backward(y,labels)

            net.model:backward(inputs,dE_dy)

            confusion:batchAdd(y, labels)

            -- return f and df/dX
            return E,dE_dw
        end
        -- optimize on current mini-batch
        _,fs = optim.sgd(eval_E, w, optimState)
        local err = fs[1]

        if opt.gpu then cutorch.synchronize() end

        io.write(string.format("%sEpoch: [%d][%d/%d]\tTime %.3f  Loss %.4f",
            '\r',epoch, batchNumber, opt.epochSize, timer:time().real, err))
        io.flush()

        batchNumber = batchNumber + 1
        loss_epoch = loss_epoch + err
end