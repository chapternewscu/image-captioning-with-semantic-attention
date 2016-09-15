function stringToByteTensor(str, vecLength)
    -- str: string, table of strings, or tensor of ascii codes
    -- vecLength: vector length
    -- output: 2D ASCII byte tensor
    local strByteTensor = torch.ByteTensor()
    if type(str) == 'string' then
        str = {str}
    end
    torch.IntTensor().nn.Utils_stringToByteTensor(str, vecLength, strByteTensor)
    return strByteTensor
end

function stringToVec(str, vecLength)
    -- str: string, table of strings, or tensor of ascii codes
    -- vecLength: vector length
    -- strVec: 2D-tensor of vectorized strings
    local strVec = torch.IntTensor()
    if type(str) == 'string' or type(str) == 'table' then
        if type(str) == 'string' then
            str = {str}
        end
        strVec.nn.Utils_stringToVec(str, vecLength, strVec)
    elseif str:type() == 'torch.ByteTensor' then
        strVec.nn.Utils_byteTensorToVec(str, vecLength, strVec)
    else
        error('Unrecognized input type')
    end
    return strVec
end


function vecToString(vec)
    if vec:dim() == 1 then
        vec = vec:view(1, vec:size(1))
    end
    local strings = vec.nn.Utils_vecToString(vec)
    return strings
end


function oneHotEmbedding(target, yDim, pruneT)
    -- target: [n x maxT] IntTensor, input label sequences
    -- embedding: [n x T' x Dy] Tensor, output embedded tensor
    pruneT = pruneT or false
    local n, T = target:size(1), target:size(2)
    local embedding = torch.Tensor(n, T, yDim):fill(0)
    local maxT = 0
    for i = 1, n do
        for t = 1, T do
            local label = target[i][t]
            if label ~= 0 then
                embedding[i][t][label] = 1.0
                maxT = math.max(maxT, t)
            end
        end
    end
    if pruneT then
        embedding = embedding:narrow(2, 1, maxT):clone()
    end
    return embedding
end


function setupLogger(fpath)
    local fileMode = 'w'
    if paths.filep(fpath) then
        local input = nil
        while not input do
            print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
            input = io.read()
            if input == 'o' then
                fileMode = 'w'
            elseif input == 'a' then
                fileMode = 'a'
            elseif input == 'q' then
                os.exit()
            else
                fileMode = nil
            end
        end
    end
    gLoggerFile = io.open(fpath, fileMode)
end


function tensorInfo(x, name)
    local name = name or ''
    local sizeStr = ''
    for i = 1, #x:size() do
        sizeStr = sizeStr .. string.format('%d', x:size(i))
        if i < #x:size() then
            sizeStr = sizeStr .. 'x'
        end
    end
    infoStr = string.format('[%15s] size: %12s, min: %+.2e, max: %+.2e', name, sizeStr, x:min(), x:max())
    return infoStr
end


function shutdownLogger()
    if gLoggerFile then
        gLoggerFile:close()
    end
end


function logging(message, mute)
    mute = mute or false
    local timeStamp = os.date('%x %X')
    local msgFormatted = string.format('[%s]  %s', timeStamp, message)
    if not mute then
        print(msgFormatted)
    end
    if gLoggerFile then
        gLoggerFile:write(msgFormatted .. '\n')
        gLoggerFile:flush()
    end
end

function modelSize(model)
    -- calculate the number of parameters in a model
    local params = model:parameters()
    local count = 0
    local countForEach = {}
    for i = 1, #params do
        local nParam = params[i]:numel()
        count = count + nParam
        countForEach[i] = nParam
    end
    return count, torch.LongTensor(countForEach)
end


function cloneList(tensorList, setZero)
    local out = {}
    for k, v in pairs(tensorList) do
        out[k] = v:clone()
        if setZero then out[k]:zero() end
    end
    return out
end


function cloneManyTimes(module, T)
    local clones = {}
    local params, gradParams = module:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(module)
    for t = 1, T do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end


function diagnoseGradients(params, gradParams)
    for i = 1, #params do
        local pMin = params[i]:min()
        local pMax = params[i]:max()
        local gpMin = gradParams[i]:min()
        local gpMax = gradParams[i]:max()
        local normRatio = gradParams[i]:norm() / params[i]:norm()
        logging(string.format('%02d - params [%+.2e, %+.2e] gradParams [%+.2e, %+.2e], norm gp/p %+.2e',
            i, pMin, pMax, gpMin, gpMax, normRatio), true)
    end
end


function dumpModelState(model)
    local state = model:parameters()
    local bnLayers = model:findModules('nn.BatchNormalization')
    for i = 1, #bnLayers do
        table.insert(state, bnLayers[i].running_mean)
        table.insert(state, bnLayers[i].running_std)
    end
    local sbnLayers = model:findModules('nn.SpatialBatchNormalization')
    for i = 1, #sbnLayers do
        table.insert(state, sbnLayers[i].running_mean)
        table.insert(state, sbnLayers[i].running_std)
    end
    return state
end


function loadModelState(model, stateToLoad)
    local state = dumpModelState(model)
    assert(#state == #stateToLoad)
    for i = 1, #state do
        state[i]:copy(stateToLoad[i])
    end
end
