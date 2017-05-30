--
-- User: peyman
-- Date: 12/7/16
-- Time: 11:03 AM
-- Split the train data into test, train and validation
--
require 'torch'
require 'sys'
require '../utils'
local dir = require 'pl.dir'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Split the train data into train and validation. Dumps meta data to logfile')
cmd:text('Example:')
cmd:text("th -i split.lua -splitTrain 0.8")
cmd:text('Options:')
cmd:option('-datasetRoot', '/home/faratin/git/mallorypoc/Data/malimg', 'root path to training data')
cmd:option('-trainzip', 'malimg_dataset.zip', 'path to train.zip file')
cmd:option('-trainRoot', 'train', 'path to training data')
cmd:option('-validationRoot', 'validation', 'path to validation data')
cmd:option('-testRoot', 'test', 'path to validation data')
cmd:option('-splitTrain', 0.8, 'Portion of split to go to training. Default: 0.8')
cmd:option('-splits', '{0.2,0.8}', 'Test-train, train-valid splits: Default: {0.2,0.8}')
cmd:option('-logfile', 'meta.txt', 'Meta file to record data sizes')
cmd:text()
opt = cmd:parse(arg or {})

opt.splits = tbl2num(opt.splits)

local unzip = 'unzip'
local logfile = assert(io.open(paths.concat(opt.datasetRoot,opt.logfile), 'w'))

local trainZipFile=paths.concat(opt.datasetRoot,opt.trainzip)
local testPath=paths.concat(opt.datasetRoot,opt.testRoot)
local trainPath=paths.concat(opt.datasetRoot,opt.trainRoot)
local validPath=paths.concat(opt.datasetRoot,opt.validationRoot)

-- unzip train.zip
print('...unzipping ',trainZipFile)
os.execute(unzip .. ' -q ' .. trainZipFile .. ' -d ' .. trainPath)
os.execute('mv ' .. trainPath .. '/malimg_paper_dataset_imgs/* ' .. trainPath)
os.execute('rm -rf ' .. trainPath .. '/malimg_paper_dataset_imgs')

-- create validation dir
os.execute('mkdir -p ' .. validPath)
os.execute('mkdir -p ' .. testPath)

-- split
print('...building splits from data at ',trainPath)
local dirs = dir.getdirectories(trainPath)

for k,dirpath in ipairs(dirs) do
    local class = paths.basename(dirpath)
    local classValidPath = paths.concat(validPath,class)
    local classTestPath = paths.concat(testPath,class)

    os.execute('mkdir -p ' .. classValidPath)
    os.execute('mkdir -p ' .. classTestPath)

    local imgs={}
    for f in paths.files(dirpath) do
        if paths.extname(f) == 'png' then
            table.insert(imgs,paths.concat(dirpath,f))
        end
    end

    local traintestShuffle = torch.randperm(#imgs)
    local testsize = torch.floor(traintestShuffle:size(1)*(opt.splits[1]))
    local trsize = traintestShuffle:size(1) - testsize
    local valsize = torch.floor(trsize*(1-opt.splits[2]))
    trsize = trsize - valsize

    for i=1,testsize do
        local img = imgs[traintestShuffle[i]]
        os.execute('mv ' .. img .. ' ' .. classTestPath)
    end

    -- we dont go anything here since the train data is already in right place
    for i=testsize+1,trsize do
    end

    for i=trsize+1,trsize+valsize do
        local img = imgs[traintestShuffle[i]]
        os.execute('mv ' .. img .. ' ' .. classValidPath)
    end

    print('\n' .. class)
    print('#images ',#imgs)
    print('#test ',testsize)
    print('#train ',trsize)
    print('#validation ',valsize)

    logfile:write('\n')
    logfile:write(class)
    logfile:write('\n' .. 'original #images ',#imgs .. '\t' .. dirpath)
    logfile:write('\n' .. '#test ',testsize .. '\t' .. classTestPath)
    logfile:write('\n' .. '#train ',trsize .. '\t' .. dirpath)
    logfile:write('\n' .. '#validation ',valsize .. '\t' .. classValidPath)
    logfile:write('\n')
end
io.close(logfile)

