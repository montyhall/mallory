--
-- User: peyman
-- Date: 12/7/16
-- Time: 10:28 AM
--

if opt.gpu then require 'cunn' end

validateLogger = optim.Logger(paths.concat(opt.save, 'validate.log'))

local validateDataIterator = function()
    validateLoader:reset()
    return function() return validateLoader:get_batch(false) end
end

-- Save light network tools:
function nilling(module)
    module.gradBias   = nil
    if module.finput then module.finput = torch.Tensor() end
    module.gradWeight = nil
    module.output     = torch.Tensor()
    if module.fgradInput then module.fgradInput = torch.Tensor() end
    module.gradInput  = nil
end

function netLighter(network)
    nilling(network)
    if network.modules then
        for _,a in ipairs(network.modules) do
            netLighter(a)
        end
    end
end

-----------------------------------------------------
-- vars
-----------------------------------------------------
local batchNumber
local loss_epoch
local top1_10crop, top5_10crop
local current_loss=math.huge
local min_loss=math.huge
local current_accuracy,max_accuracy=0,0
local best_valid_epoch=0
local ntrial = 0
bestModel=nil

local timer = torch.Timer()
local inputsCPU = torch.Tensor(opt.validateBatchSize, opt.inputPixels[1], opt.inputPixels[2], opt.inputPixels[3])
local labelsCPU = torch.LongTensor(opt.validateBatchSize)

local inputs,labels=nil,nil

if opt.gpu then
    inputs = torch.Tensor(opt.validateBatchSize, opt.inputPixels[1], opt.inputPixels[2], opt.inputPixels[3])
    labels = torch.Tensor(opt.validateBatchSize)

    inputs  = inputs:cuda()
    labels  = labels:cuda()

    net.model:cuda()
end

local confusion = optim.ConfusionMatrix(classes)

-----------------------------------------------------
-- validation all batchsize for 1 epoc
-----------------------------------------------------
function validate()
    print('\n==> doing epoch on validation data:')
    print("==> online epoch # " .. epoch)

    -- set the dropouts to evaluate mode
    net.model:evaluate()

    if opt.gpu then cutorch.synchronize() end

    timer:reset()

    batchNumber = 0
    loss_epoch = 0
    current_accuracy = 0
    confusion:zero()
    ntrial = ntrial + 1

    print('nValidate: ' .. nValidate) -- see data.lua
    print('validateBatchSize: ' .. opt.validateBatchSize)

    for i=1,nValidate/opt.validateBatchSize do -- nValidate is set in 1_data.lua
        local indexStart = (i-1) * opt.validateBatchSize + 1
        local indexEnd = (indexStart + opt.validateBatchSize - 1)
        donkeys:addjob(
        -- work to be done by donkey thread
            function()
                local inputs, labels = validateLoader:get(indexStart, indexEnd)
                return sendTensor(inputs), sendTensor(labels)
            end,
            -- callback that is run in the main thread once the work is done
            validateBatch
        )
        if i % 5 == 0 then
            donkeys:synchronize()
            collectgarbage()
        end
    end

    donkeys:synchronize()

    if opt.gpu then cutorch.synchronize() end

    confusion:updateValids()
    current_accuracy = confusion.totalValid * 100
    loss_epoch = loss_epoch / (nValidate/opt.validateBatchSize) -- because loss_epoch is calculated per batch

    validateLogger:add{ ['avg loss (validate set)'] = loss_epoch }

    -- print to console
    print(string.format('Epoch: [%d][VALIDATING SUMMARY] Total Time(s): %.2f \t'
            .. 'average loss (per batch): %.2f \t ', epoch, timer:time().real, loss_epoch))
    print(confusion)
    print("ntrial           = " .. ntrial)
    print("current accuracy = " .. current_accuracy)
    print("max accuracy     = " .. max_accuracy)
    print("current loss     = " .. loss_epoch)
    print("min average loss = " .. min_loss)

    -- early stopping
    if current_accuracy > max_accuracy then
        max_accuracy = current_accuracy
        best_valid_epoch = epoch

        bestModel = net.model:clone()
        netLighter(bestModel)

        if loss_epoch < min_loss then
            min_loss = loss_epoch
        end

        ntrial = 0
        stop = false

    elseif ntrial >= opt.earlystop  then
        print("\nEARLY STOP.\nNo new minima found after "..ntrial.." epochs.")
        print('Best accuracy was: ' ..  max_accuracy .. 'at epoch: ' .. best_valid_epoch)
        stop=true
    end
    return stop
end -- of validate()
---------------------------------------------------------------------------

function validateBatch(inputsThread, labelsThread)
    if opt.gpu then cutorch.synchronize() end

    batchNumber = batchNumber + opt.validateBatchSize

    receiveTensor(inputsThread, inputsCPU)
    receiveTensor(labelsThread, labelsCPU)

    if opt.gpu then
        inputs:copy(inputsCPU)
        labels:copy(labelsCPU)
    else
        inputs=inputsCPU
        labels = labelsCPU
    end

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local y = net.model:forward(inputs)
    local E = loss:forward(y, labels)

    confusion:batchAdd(y, labels)

    if opt.gpu then cutorch.synchronize() end

    local pred = y:float()

    loss_epoch = loss_epoch + E
end