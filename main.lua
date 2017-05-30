--
-- User: peyman
-- Date: 11/25/16
-- Time: 11:42 PM
-- to run: th main.lua

require 'torch'
require 'paths'
require 'xlua'
require 'optim'

virusClassifier = {}
virusClassifier.version = 1

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
opt = dofile('opts.lua').parse(arg)

if opt.gpu then
    require 'cutorch'
    require 'cunn'
end

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data/data.lua')
paths.dofile('criteria.lua')
paths.dofile('model.lua')
paths.dofile('utils.lua')

if opt.testMode then
    paths.dofile('test.lua')
    test()
else

    paths.dofile('train.lua')
    paths.dofile('validate.lua')

    epoch = 1
    stop=false

    -- nEpochs= Number of total epochs to run
    for i=1,opt.nEpochs do
        train()
        stop = validate()
        if stop then
            break
        else
            epoch = epoch + 1
        end
    end
    -- save
    print("\n\nSaving model at epoch stage: " .. epoch)
    local filename = paths.concat(opt.save, opt.network)
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('saving final model to ' .. filename)
    if bestModel ~= nil then
        print('found best model')
        net.model = bestModel
    end
    torch.save(filename, net)
    saveWs(net)
end

print('done')
--end
