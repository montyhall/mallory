--
-- User: peyman
-- Date: 12/8/16
-- Time: 9:32 AM
-- To change this template use File | Settings | File Templates.
--

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
    testLoader:reset()
    return function() return testLoader:get_batch(false) end
end

local function loadModel()
    -- load learnt model
    --torch.load(opt.serializedModel)
    ok,model = pcall(torch.load, opt.serializedModel)
    if not ok then
        print(sys.COLORS.red .. '... could not load model -> ',opt.serializedModel)
        return
    end
    return model
end

local net  = loadModel()

net.model:evaluate()

channels=net.channels
c=net.inputPixels[1]
h=net.inputPixels[2]
w=net.inputPixels[3]
target_h=net.labelPixels[1]
target_w=net.labelPixels[2]

local batchNumber
local timer = torch.Timer()
filename2label = {}
local predFile=paths.concat(opt.save,opt.testRoot .. '_predictions.txt')
local outputHandle = assert(io.open(predFile, 'w'))
local inputsCPU = torch.Tensor()
local inputs=nil
local correct=0

if opt.gpu then
    inputs = torch.Tensor(opt.testBatchSize,
        opt.inputPixels[1],
        opt.inputPixels[2],
        opt.inputPixels[3])
    inputs  = inputs:cuda()
    net.model:cuda()
end
-----------------------------------------------------
-- test
-----------------------------------------------------
function test()
    print('==> doing epoch on test data:')

    batchNumber = 0
    if opt.gpu then cutorch.synchronize() end
    timer:reset()

    -- set the dropouts to evaluate mode
    net.model:evaluate()

    print('nTest: ' .. nTest)
    print('testBatchSize: ' .. opt.testBatchSize)
    for i=1, nTest/opt.testBatchSize do -- nTest is set in 1_data.lua
        local indexStart = (i-1) * opt.testBatchSize + 1
        local indexEnd = (indexStart + opt.testBatchSize - 1)
        donkeys:addjob(
        -- work to be done by donkey thread
            function()
                local inputs, names = testLoader:get(indexStart, indexEnd)
                -- local i_stg = tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
                -- local l_stg =  torch.pointer(names:storage())
                -- inputs:cdata().storage = nil
                return sendTensor(inputs), names
            end,
            -- callback that is run in the main thread once the work is done
            testBatch
        )
        donkeys:synchronize()
        collectgarbage()
    end

    donkeys:synchronize()

    if opt.gpu then cutorch.synchronize() end

    print('test finished.')
    print('% correct ',(correct/nTest)*100)
    print('predictions written to',predFile)
    --sort -t $'\t' -k 2,2 /home/faratin/git/kagglefish/CNN/results/test_stg1_predictions.txt > /home/faratin/git/kagglefish/CNN/results/test_stg1_predictions_sorted.txt

end -- of test()
-----------------------------------------------------------------------------

function testBatch(inputsThread, names)

    if opt.gpu then cutorch.synchronize() end

    batchNumber = batchNumber + opt.testBatchSize

    --[[names =
    1 : "/home/faratin/git/mallorypoc/Data/malimg/test/Alueron.gen!J/060d135c6dfba5c129dea27045e67504.png"
    2 : "/home/faratin/git/mallorypoc/Data/malimg/test/Alueron.gen!J/00c8d9d6149199f4dde4a64279c63216.png
    --]]

    io.write(string.format("%sBatch: [%d/%d]",'\r', batchNumber,nTest))
    io.flush()

    receiveTensor(inputsThread, inputsCPU)

    if opt.gpu then
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
    else
        inputs=inputsCPU
    end

    local outputs = net.model:forward(inputs)

    if opt.gpu then cutorch.synchronize() end

    local pred = outputs:float()

    local function updateTestResult(prob, name)
        local classOf = paths.basename(name:split(paths.basename(name))[1]) -- get target class name from path name
        _,p = prob:sort(1, true)
        local predictedLabel = classes[p[1]]
        outputHandle:write(name .. '\t' .. classOf .. '\t' .. predictedLabel .. '\n')

        if classOf == predictedLabel then
            correct = correct + 1
        end

        --print('predictedLabel index: ' .. p[1])
        _, filename = string.match(name, "(.-)([^//]-([^%.]+))$")
        filename2label[filename] = predictedLabel
    end

    if opt.augment ~= 'none' then
        for i=1,pred:size(1),30 do
            local porbs = pred[{{i, i+29}, {}}]
            local tencrop = porbs:sum(1)[1]
            updateTestResult(tencrop, names[i])
        end
    else
        for i=1,pred:size(1) do
            updateTestResult(pred[i], names[i])
        end
    end
end