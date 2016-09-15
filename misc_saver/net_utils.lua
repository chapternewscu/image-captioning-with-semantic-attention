local utils = require 'misc_saver.utils'
require 'misc_saver.LinearTensorD3'

local net_utils = {}

 -- version 1: no attention model, input the image at every time step, use LSTM_Armour.lstm directy as self.core in LanguageModel.lua
-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn_v1(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38) 
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 128)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end
   
  cnn_part:add(nn.Linear(4096,encoding_size))
  cnn_part:add(backend.ReLU(true))
  
  return cnn_part
end


-- version 2: combined with attention model, output will be the fourth convoluational layer 
-- before maxpooling
-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
-- output will be: batch_size * 512 * 14 * 14
function net_utils.build_cnn_v2(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 30) 
  local backend_string = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 128)
 
  local backend = nil 
  if backend_string == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend_string == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    --[[ no need to set below, cause already to be true even without it 
    if torch.typename(layer) == backend_string .. '.ReLU' then 
        print(i .. '-th layer')
        print(layer)
        -- set the inplace varibale of cudnn.ReLU to be true
        layer.inplace = true
    end
    ---]]
    
    cnn_part:add(layer)
  end
  -- output: bs * L * h * w(bs * 512 * 14 * 14)
  -- view the tensor in format: bs * L * D (L is the number of features for each image, D is the feature vector size at each position)
  cnn_part:add(nn.View(512, -1):setNumInputDims(3))
  cnn_part:add(nn.Transpose({2, 3}))
  --- output: bs * 196 * 512
  --print('done!')
  cnn_part:add(nn.LinearTensorD3(512, encoding_size))
  print(cnn_part)
  return cnn_part
end


--load googlenet 
function net_utils.build_goolenet(cnn, opt)

end 

-- just return a linear transform model 
-- thus we can use features which are already extracted
-- here, we suppose we extract vgg 4096 features 
-- for googlenet, feature size will be 1024 size features 
function net_utils.build_linear_trans(opt) 
    local backend = utils.getopt(opt, 'backend', 'cudnn')
    local encoding_size = utils.getopt(opt, 'encoding_size', 512) 

    if backend == 'cudnn' then 
        require 'cudnn'
        backend = cudnn
    elseif backend == 'nn' then 
        require 'nn'
        backend = nn 
    else 
        error(string.format('Unrecognized backend %s', backend))
    end 

    local linear_transform = nn.Sequential() 
    linear_transform:add(nn.Linear(4096, encoding_size))
    linear_transform:add(backend.ReLU(true)) 

    return linear_transform 
end 




-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  return imgs
end

-- load glove vector 
-- Parses and loads the GloVe word vectors into a hash table: 
--   glove_table['word'] = vector 
function net_utils.load_glove(glove_path, inputDim)
    local glove_file = io.open(glove_path)
    local glove_table = {} 
    
    local line = glove_file:read("*l")
    while line do 
        -- read GloVe text file one line at a time, break at EOF
        local k = 1 
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if k == 1 then 
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case 
                if string.len(word) > 0 then 
                    -- may just use 
                    -- glove_table[word] = torch.zeros(inputDim)
                    glove_table[word] = torch.zeros(inputDim) -- padded with an extra dimention for convolution 
                else 
                    break 
                end 
            else
                -- read off and store each word vector element
                glove_table[word][k-1] = tonumber(entry)
            end 
            k = k+1
        end
        line = glove_file:read("*l")
    end 

    return glove_table
end 

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end


-- layer that expands features map  out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.Feat2DExpander', 'nn.Module')
function layer:__init(n)
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 3)
  local nc, fd = input:size(2), input:size(3)
  self.output:resize(input:size(1)*self.n, nc, fd)

  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {}, {}}]:expand(self.n, nc, fd) -- copy over
  end
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local nc, fd = input:size(2), input:size(3)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end


function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end

function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

-- allow nested nn.gModule(depth = 2, ie, nn.gModule in nn.gModule)
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    -- allow two level  nn.gModule
    if torch.type(m) == 'nn.gModule' then 
        local nested_moduleList = net_utils.listModules(m)
        for nested_k, nested_m in ipairs(nested_moduleList) do 
            if nested_m.weight and nested_m.gradWeight then
                --print('sanitizing gradWeight in of size ' .. nested_m.gradWeight:nElement())
                --print(nested_m.weight:size())
                nested_m.gradWeight = nil
            end
            if nested_m.bias and nested_m.gradBias then
                --print('sanitizing gradWeight in of size ' .. nested_m.gradBias:nElement())
                --print(nested_m.bias:size())
                nested_m.gradBias = nil
            end
        end 
    else  -- not nn.gModule
        if m.weight and m.gradWeight then
            --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
            --print(m.weight:size())
            m.gradWeight = nil
        end
        if m.bias and m.gradBias then
            --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
            --print(m.bias:size())
            m.gradBias = nil
        end
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if torch.type(m) == 'nn.gModule' then
        local nested_moduleList = net_utils.listModules(m) 
        for nested_k, nested_m in ipairs(nested_moduleList) do  
            if nested_m.weight and (not nested_m.gradWeight) then
                nested_m.gradWeight = nested_m.weight:clone():zero()
                --print('unsanitized gradWeight in of size ' .. nested_m.gradWeight:nElement())
                --print(nested_m.weight:size())
            end
            if nested_m.bias and (not nested_m.gradBias) then
                nested_m.gradBias = nested_m.bias:clone():zero()
                --print('unsanitized gradWeight in of size ' .. nested_m.gradBias:nElement())
                --print(nested_m.bias:size())
            end   
        end 
    else 
        if m.weight and (not m.gradWeight) then
            m.gradWeight = m.weight:clone():zero()
            --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
            --print(m.weight:size())
        end
        if m.bias and (not m.gradBias) then
            m.gradBias = m.bias:clone():zero()
            --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
            --print(m.bias:size())
        end
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

return net_utils
