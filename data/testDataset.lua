--
-- User: peyman
-- Date: 11/28/16
-- Time: 1:15 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local testDataset = torch.class('testDataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
     A testDataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large testDatasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
    {check=function(paths)
        local out = true;
        for k,v in ipairs(paths) do
            if type(v) ~= 'string' then
                print('paths can only be of string input');
                out = false
            end
        end
        return out
    end,
        name="paths",
        type="table",
        help="Multiple paths of directories with images"},

    {name="sampleSize",
        type="table",
        help="a consistent sample size to resize the images"},

    {name="samplingMode",
        type="string",
        help="Sampling mode: random | balanced ",
        default = "balanced"},

    {name="verbose",
        type="boolean",
        help="Verbose mode during initialization",
        default = false},

    {name="loadSize",
        type="table",
        help="a size to load the images to, initially",
        opt = true},

    {name="forceClasses",
        type="table",
        help="If you want this loader to map certain classes to certain indices, "
                .. "pass a classes table that has {classname : classindex} pairs."
                .. " For example: {3 : 'dog', 5 : 'cat'}"
                .. "This function is very useful when you want two loaders to have the same "
                .. "class indices (trainLoader/validateLoader for example)",
        opt = true},

    {name="sampleHookTrain",
        type="function",
        help="applied to sample during training(ex: for lighting jitter). "
                .. "It takes the image path as input",
        opt = true},

    {name="sampleHookTest",
        type="function",
        help="applied to sample during testing",
        opt = true},
}

function testDataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    --print(args)
    for k,v in pairs(args) do self[k] = v end

    if not self.loadSize then self.loadSize = self.sampleSize; end

    if not self.sampleHookTrain then
        -- print('sampleHookTrain is null')
        self.sampleHookTrain = self.defaultSampleHook
    end
    if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

    local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end

    -- define command-line tools, try your best to maintain OSX compatibility
    local wc = 'wc'
    local cut = 'cut'
    local find = 'find'
    if jit and jit.os == 'OSX' then
        wc = 'gwc'
        cut = 'cut'
        find = 'find'
    end
    ----------------------------------------------------------------------
    -- Options for the GNU find command
    local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
    local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
    for i=2,#extensionList do
        findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
    end

    -- find the image path names
    self.imagePath = torch.CharTensor()  -- path to each image in testDataset

    --print('running "find" on each class directory, and concatenate all'
    --        .. ' those filenames into a single file containing all image paths for a given class')
    -- so, generates one file per class
    local tmpfile = os.tmpname()
    local tmphandle = assert(io.open(tmpfile, 'w'))
    local allPaths = os.tmpname()
    -- iterate over classes
    local path = self.paths[1]
    local command = find .. ' "' .. path .. '" ' .. findOptions
            .. ' >>"' .. allPaths .. '" \n'
    tmphandle:write(command)

    io.close(tmphandle)
    os.execute('bash ' .. tmpfile)
    os.execute('rm -f ' .. tmpfile)

    --==========================================================================
    --print('load the large concatenated list of sample paths to self.imagePath')
    local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
            .. allPaths .. "' |"
            .. cut .. " -f1 -d' '")) + 1
    local length = tonumber(sys.fexecute(wc .. " -l '"
            .. allPaths .. "' |"
            .. cut .. " -f1 -d' '"))
    assert(length > 0, "Could not find any image file in the given input paths")
    assert(maxPathLength > 0, "paths of files are length 0?")
    -- print("maxPathLength: " .. maxPathLength)
    print("image list length: " .. length)
    self.imagePath:resize(length, maxPathLength):fill(0)
    local s_data = self.imagePath:data()
    local count = 0
    for line in io.lines(allPaths) do
        ffi.copy(s_data, line)
        s_data = s_data + maxPathLength
        if self.verbose and count % 100 == 0 then
            --xlua.progress(count, length)
        end;
        count = count + 1
    end

    self.numSamples = self.imagePath:size(1)
    print(self.numSamples ..  ' samples found.')

    os.execute('rm -f "' .. allPaths .. '"')
end

-- size(), size(class)
function testDataset:size()
    return self.numSamples
end

-- by default, just load the image and return it
function testDataset:defaultSampleHook(imgpath)
    local out = image.load(imgpath, self.loadSize[1])
    out = image.scale(out, self.sampleSize[3], self.sampleSize[2],'bilinear')
    return out
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, namesTable)
    local data, scalarNames
    local quantity = #namesTable
    local samplesPerDraw
    if dataTable[1]:dim() == 3 then samplesPerDraw = 1
    else samplesPerDraw = dataTable[1]:size(1) end
    -- print('samplesPerDraw: ' .. samplesPerDraw)
    if quantity == 1 and samplesPerDraw == 1 then
        data = dataTable[1]
        scalarNames = namesTable[1]
    else
        data = torch.Tensor(quantity * samplesPerDraw,
            self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
        scalarNames = {}

        for i=1,#dataTable do
            local idx = (i-1)*samplesPerDraw
            data[{{idx+1,idx+samplesPerDraw}}]:copy(dataTable[i])
            local filename = ffi.string(torch.data(namesTable[i]))
            for j = 1,samplesPerDraw do
                table.insert(scalarNames, filename)
            end
        end
    end
    return data, scalarNames
end

-- sampler, samples from the training set.
-- function testDataset:sample(quantity)
--    if self.split == 0 then
--       error('No training mode when split is set to 0')
--    end
--    quantity = quantity or 1
--    local dataTable = {}
--    local scalarTable = {}
--    for i=1,quantity do
--       local class = torch.random(1, #self.classes)
--       local out = self:getByClass(class)
--       table.insert(dataTable, out)
--       table.insert(scalarTable, class)
--    end
--    local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
--    return data, scalarLabels, labels
-- end

function testDataset:get(i1, i2)
    local indices, quantity
    indices = torch.range(i1, i2);
    quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local namesTable = {}
    for i=1,quantity do
        -- load the sample
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
        local status, err = pcall(function()
            local out = self:sampleHookTest(imgpath)
            table.insert(dataTable, out)
        end)
        if not status then
            local out = opt.augment ~= 'none' and torch.Tensor(30, self.sampleSize[1], self.sampleSize[3], self.sampleSize[2])
                    or torch.Tensor(self.sampleSize[1], self.sampleSize[3], self.sampleSize[2])
            table.insert(dataTable, out)
            print('imgpath not loaded successfully: ' .. imgpath)
            print(err)
        end
        table.insert(namesTable, self.imagePath[indices[i]])
    end
    local data, scalarNames = tableToOutput(self, dataTable, namesTable)
    return data, scalarNames
end

function testDataset:validate(quantity)
    if self.split == 100 then
        error('No validate mode when you are not splitting the data')
    end
    local i = 1
    local n = self.validateIndicesSize
    local qty = quantity or 1
    return function ()
        if i+qty-1 <= n then
            local data, scalarLabelss, labels = self:get(i, i+qty-1)
            i = i + qty
            return data, scalarLabelss, labels
        end
    end
end

return testDataset