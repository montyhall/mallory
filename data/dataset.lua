--
-- User: peyman
-- Date: 12/1/16
-- Time: 8:08 AM
-- To change this template use File | Settings | File Templates.
--
--
-- User: peyman
-- Date: 11/2/16
-- Time: 11:58 AM
-- To change this template use File | Settings | File Templates.
--

--
-- Ara2012 and Ara2013(Cannon) 7 megapixel camera -> width × height: 3108×2324 pixels
-- Ara2013 (Rasp. Pi) Raspberry Pi7 with 5 megapixel camera -> width × height: 2592×1944 pixels

-- 2655 files

--require 'image'
--require 'paths'
--require 'xlua'
--stringx = require 'pl.stringx'
--require 'lfs'
--
--local class = require('pl.class')
--local dir = require 'pl.dir'
--local tablex = require 'pl.tablex'
--local argcheck = require 'argcheck'
--require 'sys'

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

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
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

    {name="split",
        type="number",
        help="Percentage of split to go to Training"
    },

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

function dataset:__init(...)
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

    -- find class names
    self.classes = {}
    local classPaths = {}
    if self.forceClasses then
        for k,v in pairs(self.forceClasses) do
            --print('k: ' .. k .. ' v: ' .. v)
            self.classes[k] = v
            classPaths[k] = {}
        end
    end
    local function tableFind(t, o) for k,v in pairs(t) do if v == o then return k end end end
    -- loop over each paths folder, get list of unique class names,
    -- also store the directory paths per class
    -- for each class,
    for k,path in ipairs(self.paths) do
        local dirs = dir.getdirectories(path);
        for k,dirpath in ipairs(dirs) do
            local class = paths.basename(dirpath)
            local idx = tableFind(self.classes, class)
            if not idx then
                table.insert(self.classes, class)
                idx = #self.classes
                classPaths[idx] = {}
            end
            if not tableFind(classPaths[idx], dirpath) then
                table.insert(classPaths[idx], dirpath);
            end
        end
    end
    -- classes[1]=SHARK
    -- classPaths[1]=/Users/peyman/git/kagglefish/data/train/SHARK
    -- classIndices[SHARK]=1

    self.classIndices = {}
    for k,v in ipairs(self.classes) do
        --print('k: ' .. k .. ' v: ' .. v)
        self.classIndices[v] = k
    end

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
    local findOptions = ' -iname "*.' .. extensionList[1] .. '"' -- - iname "*.jpg"
    for i=2,#extensionList do
        findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
    end
    -- -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPG" -o -iname "*.PNG" -o -iname "*.JPEG" -o -iname "*.ppm" -o -iname "*.PPM" -o -iname "*.bmp" -o -iname "*.BMP"

    -- find the image path names
    self.imagePath = torch.CharTensor()  -- path to each image in dataset
    self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
    self.classList = {}                  -- index of imageList to each image of a particular class
    self.classListSample = self.classList -- the main list used when sampling data

    -- imagePath = numImages x maxLengthOfFileName byte array
    -- imageClass:narrow(1,1,10) = 1 1 1 ..1 <- 10 1s for class 1 (shark)
    -- classList[1] = 1 2 3 .. 176 <- 176 files for class 1 (shark)

    --print('running "find" on each class directory, and concatenate all'
    --        .. ' those filenames into a single file containing all image paths for a given class')
    -- so, generates one file per class
    local classFindFiles = {}
    for i=1,#self.classes do
        classFindFiles[i] = os.tmpname()
    end

    local tmpfile = os.tmpname()
    local tmphandle = assert(io.open(tmpfile, 'w'))
    -- iterate over classes
    for i, class in ipairs(self.classes) do
        -- iterate over classPaths
        for j,path in ipairs(classPaths[i]) do
            --find "/Users/peyman/git/kagglefish/data/train/SHARK"  -iname "*.jpg" -o -iname "*.png" -o -iname "*.JPG" -o -iname "*.PNG" -o -iname "*.JPEG" -o -iname "*.ppm" -o -iname "*.PPM" -o -iname "*.bmp" -o -iname "*.BMP" >>"/tmp/lua_UFGkZH"
            -- classFindFiles[i] =
             --/Users/peyman/git/kagglefish/data/train/SHARK/img_00096.jpg
             --/Users/peyman/git/kagglefish/data/train/SHARK/img_00235.jpg
            local command = find .. ' "' .. path .. '" ' .. findOptions
                    .. ' >>"' .. classFindFiles[i] .. '" \n'
            tmphandle:write(command)
        end
    end
    io.close(tmphandle)
    os.execute('bash ' .. tmpfile)
    os.execute('rm -f ' .. tmpfile)

    --print('now combine all the files to a single large file')
    local combinedFindList = os.tmpname();
    local tmpfile = os.tmpname()
    local tmphandle = assert(io.open(tmpfile, 'w'))
    -- concat all finds to a single large file in the order of self.classes
    for i=1,#self.classes do
        local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
        tmphandle:write(command)
    end
    io.close(tmphandle)
    os.execute('bash ' .. tmpfile)
    os.execute('rm -f ' .. tmpfile)

    --==========================================================================
    --print('load the large concatenated list of sample paths to self.imagePath')
    -- length = total number of files
    -- maxPathLength = maxLengthOfFileName
    local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
            .. combinedFindList .. "' |"
            .. cut .. " -f1 -d' '")) + 1
    local length = tonumber(sys.fexecute(wc .. " -l '"
            .. combinedFindList .. "' |"
            .. cut .. " -f1 -d' '"))
    assert(length > 0, "Could not find any image file in the given input paths")
    assert(maxPathLength > 0, "paths of files are length 0?")


    self.imagePath:resize(length, maxPathLength):fill(0) -- numImages x maxLengthOfFileName
    local s_data = self.imagePath:data() --Returns a LuaJIT FFI pointer to the raw data of the tensor
    local count = 0
    for line in io.lines(combinedFindList) do
        --copy bytes of the path string (line) plus a zero-terminator to s_data
        ffi.copy(s_data, line)
        s_data = s_data + maxPathLength
        if self.verbose and count % 10000 == 0 then
            --xlua.progress(count, length)
        end;
        count = count + 1
    end
    --for i=1,self.imagePath:size(1) do
    --    print(ffi.string((torch.data(self.imagePath[i]))))
    --end
    self.numSamples = self.imagePath:size(1)
    print(self.numSamples ..  ' Total instances found.')
    --==========================================================================
    --print('Updating classList and imageClass appropriately')
    -- classList[1] = 1 2 3 .. 176 <- 176 files for class 1 (shark)
    -- imageClass:narrow(1,1,10) = 1 1 1 ..1 <- 10 1s for class 1 (shark)
    self.imageClass:resize(self.numSamples)
    local runningIndex = 0
    for i=1,#self.classes do
        --if self.verbose then xlua.progress(i, #(self.classes)) end
        local length = tonumber(sys.fexecute(wc .. " -l '"
                .. classFindFiles[i] .. "' |"
                .. cut .. " -f1 -d' '"))
        --print(string.format('class: %s has %d instances',self.classes[i],length))
        if length == 0 then
            error('Class has zero samples')
        else
            self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
            self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
        end
        runningIndex = runningIndex + length
    end
    --print('imageClass: ',self.imageClass:narrow(1,1,10))
    --==========================================================================
    -- clean up temporary files
    --print('Cleaning up temporary files')
    local tmpfilelistall = ''
    for i=1,#(classFindFiles) do
        tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
        if i % 1000 == 0 then
            os.execute('rm -f ' .. tmpfilelistall)
            tmpfilelistall = ''
        end
    end
    os.execute('rm -f '  .. tmpfilelistall)
    os.execute('rm -f "' .. combinedFindList .. '"')
    --==========================================================================
    --print('self.split: ' .. self.split)
    if self.split == 100 then
        self.validateIndicesSize = 0
    else
        print('Splitting training and validate sets to a ratio of '
                .. self.split .. '/' .. (100-self.split))
        self.classListTrain = {}
        self.classListValidate  = {}
        self.classListSample = self.classListTrain
        local totalValidateSamples = 0
        -- split the classList into classListTrain and classListValidate
        for i=1,#self.classes do
            local list = self.classList[i]
            local count = self.classList[i]:size(1)
            local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
            local perm = torch.randperm(count)
            self.classListTrain[i] = torch.LongTensor(splitidx)
            for j=1,splitidx do
                self.classListTrain[i][j] = list[perm[j]]
            end
            if splitidx == count then -- all samples were allocated to train set
                self.classListValidate[i]  = torch.LongTensor()
            else
                self.classListValidate[i]  = torch.LongTensor(count-splitidx)
                totalValidateSamples = totalValidateSamples + self.classListValidate[i]:size(1)
                local idx = 1
                for j=splitidx+1,count do
                    self.classListValidate[i][idx] = list[perm[j]]
                    idx = idx + 1
                end
            end
        end
        -- Now combine classListValidate into a single tensor
        self.validateIndices = torch.LongTensor(totalValidateSamples)
        self.validateIndicesSize = totalValidateSamples
        local tdata = self.validateIndices:data()
        local tidx = 0
        for i=1,#self.classes do
            local list = self.classListValidate[i]
            if list:dim() ~= 0 then
                local ldata = list:data()
                for j=0,list:size(1)-1 do
                    tdata[tidx] = ldata[j]
                    tidx = tidx + 1
                end
            end
        end
    end
end

-- size(), size(class)
function dataset:size(class, list)
    list = list or self.classList
    if not class then
        return self.numSamples
    elseif type(class) == 'string' then
        return list[self.classIndices[class]]:size(1)
    elseif type(class) == 'number' then
        return list[class]:size(1)
    end
end

-- size(), size(class)
function dataset:sizeTrain(class)
    if self.split == 0 then
        return 0;
    end
    if class then
        return self:size(class, self.classListTrain)
    else
        return self.numSamples - self.validateIndicesSize
    end
end

-- size(), size(class)
function dataset:sizeValidate(class)
    if self.split == 100 then
        return 0
    end
    if class then
        return self:size(class, self.classListValidate)
    else
        return self.validateIndicesSize
    end
end

-- by default, just load the image and return it
function dataset:defaultSampleHook(imgpath)
    local out = image.load(imgpath, opt.channels,'float')
    out = image.scale(out, self.sampleSize[3], self.sampleSize[2],'bilinear')
    return out
end

-- getByClass
function dataset:getByClass(class)
    local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
    local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
    return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
    local data, scalarLabels, labels
    local quantity = #scalarTable
    local samplesPerDraw
    if dataTable[1]:dim() == 3 then samplesPerDraw = 1
    else samplesPerDraw = dataTable[1]:size(1) end
    if quantity == 1 and samplesPerDraw == 1 then
        data = dataTable[1]
        scalarLabels = scalarTable[1]
        labels = torch.LongTensor(#(self.classes)):fill(-1)
        labels[scalarLabels] = 1
    else
        --print(quantity,samplesPerDraw,self.sampleSize[1], self.sampleSize[2], self.sampleSize[3],dataTable[1]:size())
        data = torch.Tensor(quantity * samplesPerDraw,
            self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
        scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
        labels = torch.LongTensor(quantity * samplesPerDraw, #(self.classes)):fill(-1)
        for i=1,#dataTable do
            local idx = (i-1)*samplesPerDraw
            data[{{idx+1,idx+samplesPerDraw}}]:copy(dataTable[i])
            scalarLabels[{{idx+1,idx+samplesPerDraw}}]:fill(scalarTable[i])
            labels[{{idx+1,idx+samplesPerDraw},{scalarTable[i]}}]:fill(1)
        end
    end
    return data, scalarLabels, labels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
    if self.split == 0 then
        error('No training mode when split is set to 0')
    end
    quantity = quantity or 1
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        local class = torch.random(1, #self.classes)
        local out = self:getByClass(class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
    return data, scalarLabels, labels
end

function dataset:get(i1, i2)
    local indices, quantity
    if type(i1) == 'number' then
        if type(i2) == 'number' then -- range of indices
            indices = torch.range(i1, i2);
            quantity = i2 - i1 + 1;
        else -- single index
            indices = {i1}; quantity = 1
        end
    elseif type(i1) == 'table' then
        indices = i1; quantity = #i1;         -- table
    elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
        indices = i1; quantity = (#i1)[1];    -- tensor
    else
        error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
    end
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
        local out = self:sampleHookTest(imgpath)
        table.insert(dataTable, out)
        table.insert(scalarTable, self.imageClass[indices[i]])
    end
    local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
    return data, scalarLabels, labels
end

function dataset:validate(quantity)
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

return dataset

