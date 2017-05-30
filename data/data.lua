--
-- User: peyman
-- Date: 11/28/16
-- Time: 12:29 AM
--
--  Copyright (c) 2018, Robustlinks, LLC.
--  All rights reserved.

local ffi = require 'ffi'
local Threads = require 'threads'

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-- see https://github.com/torch/threads#examples for threads examples
-------------------------------------------------------------------------------

nClasses = nil
classes = nil
nTrain = 0
nValidate = 0
nTest = 0

--local nproc = 'nproc'
--if jit.os == 'OSX' then
--    nproc = 'sysctl -n hw.ncpu'
--end
--local ncpus = tonumber(sys.fexecute(nproc))

do -- start K datathreads (donkeys)
    if opt.threads > 0 then
        local options = opt -- make an upvalue to serialize over to donkey threads
        donkeys = Threads(
            opt.threads,
            function()
                gsdl = require 'sdl2'
                require 'torch'
            end,
            function(idx)
                opt = options -- pass to all donkeys via upvalue
                tid = idx
                local seed = idx
                torch.manualSeed(seed)
                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                paths.dofile('donkey.lua')
            end
        );
    else -- single threaded data loading. useful for debugging
        paths.dofile('donkey.lua')
        donkeys = {}
        function donkeys:addjob(f1, f2) f2(f1()) end
        function donkeys:synchronize() end
    end
end

--call dojob until all callbacks and endcallbacks are executed on the queue and main threads, respectively
donkeys:synchronize()

donkeys:addjob(
    function()
        return trainLoader:size()
    end,
    function(c)
        nTrain = c
    end)

donkeys:synchronize()
print('nTrain: ',nTrain)

donkeys:addjob(
    function()
        return testLoader:size()
    end,
    function(c)
        nTest = c
    end)

donkeys:synchronize()
print('nTest: ',nTest)

donkeys:addjob(
-- callback function that will be executed in each queue thread with the optional ... arguments
-- threads call this function and return classes, which become 'c' input in next (endcallback)
-- function run in main thread
    function()
        return trainLoader.classes
    end,
    -- endcallback function that will be executed in this main thread
    function(c)
        classes = c
    end)

donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

donkeys:addjob(function()
    return validateLoader:sizeValidate()
end,
    function(c)
        nValidate = c
    end)

donkeys:synchronize()
assert(nValidate > 0, "Failed to get nValidate")
print('nValidate: ', nValidate)

