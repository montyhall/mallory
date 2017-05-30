local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function setFloatStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
    local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THFloatStorage_free'](cstorage)
    end
    local storage = ffi.cast('THFloatStorage*', storage_p)
    tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
    local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THLongStorage_free'](cstorage)
    end
    local storage = ffi.cast('THLongStorage*', storage_p)
    tensor:cdata().storage = storage
end

function sendTensor(inputs)
    local size = inputs:size()
    local ttype = inputs:type()
    local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
    inputs:cdata().storage = nil
    return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
    local pointer = obj[1]
    local size = obj[2]
    local ttype = obj[3]
    if buffer then
        buffer:resize(size)
        assert(buffer:type() == ttype, 'Buffer is wrong type')
    else
        buffer = torch[ttype].new():resize(size)
    end
    if ttype == 'torch.FloatTensor' then
        setFloatStorage(buffer, pointer)
    elseif ttype == 'torch.LongTensor' then
        setLongStorage(buffer, pointer)
    else
        error('Unknown type')
    end
    return buffer
end

-- save feature maps
saveWs= function(net)
    print('saving weights to ' .. opt.save)
    -- save weights
    -- https://github.com/nicholas-leonard/dp/blob/master/scripts/showfilters.lua
    for i,layer in pairs(net.model:findModules('nn.SpatialConvolution')) do
        wi = layer.weight

        if wi:size(2) == 3 then
            wi = wi:view(layer.nOutputPlane, layer.nInputPlane, layer.kW, layer.kH)
        else
            wi = wi:view(-1, layer.kW, layer.kH) --filters grey
        end
        local filters = image.toDisplayTensor{input=wi, nrow=layer.nOutputPlane,padding=1, scaleeach=false }
        image.save(opt.save .. '/layer_' .. i .. '_weights.png', filters)
    end
end

tbl2num = function(tbl)
    tbl = tbl:gsub('[{}]',''):split(',')
    for i,_ in pairs(tbl) do
        tbl[i]=tonumber(tbl[i])
    end
    return tbl
end
