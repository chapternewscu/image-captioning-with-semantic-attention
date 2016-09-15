require 'nn'
require 'nngraph'

require 'misc_saver.LinearTensorD3'
require 'misc_saver.BilinearD3_version2'
require 'misc_saver.CAddTableD2D3'
require 'misc_saver.CustomAlphaView' 

require 'misc_saver.probe' -- for debugger on nngraph module, put the layer to check gradient and outputs 
require 'misc_saver.utils_bg' -- also for debugger purpose 

local LSTM_Armour = {} 

function LSTM_Armour.makeWeightedSumUnit() 
    -- note each sample in the batch may has different alignments(or called weights)
    local alpha = nn.Identity()()              -- bz * L 
    local alphaMatrix = nn.CustomAlphaView()(alpha) -- bz * L * 1 
    
    local x = nn.Identity()()                      -- bz * L * xDim
    local g = nn.MM(true, false)({x, alphaMatrix}) -- bz * xDim * 1 

    g = nn.Select(3, 1)(g)                          -- bz * xDim 
    local inputs, outputs = {x, alpha}, {g}

    -- return a nn.Module 
    return nn.gModule(inputs, outputs)
end 

-- pass testing 
-- input_size 
function LSTM_Armour.lstm(input_size, rnn_size, n, dropout) 
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()())   --x, for test, bz * 8  
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L], bz * 8
    table.insert(inputs, nn.Identity()()) -- prev_h[L], bz * 8
  end

  local x, input_size_L

  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]  
      input_size_L = input_size 
    else  -- currently only 1 layer, this is not modified 
      x = outputs[(L-1)*2]  -- lower layer output: next_h
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='w2h_'..L} 
    -- to avoid double bias terms 
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate{name='h2h_'..L}

    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)

    -- 2 instead of 1 because it supports batch input
    -- split method is a node method which will return 4 new nodes
    -- because nn.SplitTable(2)() will return 4 output nodes 
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    -- next is 'current', which will be used as input at the next timestep
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- inputs = {x , prev_c, prev_h}
  -- outputs = {next_c, next_h}
  return nn.gModule(inputs, outputs) 
end



-- not pass testing ??????
function LSTM_Armour.lstm_with_output_attention(word_embed_size, input_size, rnn_size, outputSize, n, dropout) 
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()())   --x, glimpse, for test, bz * 8  
  table.insert(inputs, nn.Identity()())   -- As 

  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L], bz * 8
    table.insert(inputs, nn.Identity()()) -- prev_h[L], bz * 8
  end

  local x, input_size_L

  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+2]
    local prev_c = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1]  
      input_size_L = input_size 
    else  -- currently only 1 layer, this is not modified 
      x = outputs[(L-1)*2]  -- lower layer output: next_h
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='w2h_'..L} 
    -- to avoid double bias terms 
    local h2h = nn.Linear(rnn_size, 4 * rnn_size, false)(prev_h):annotate{name='h2h_'..L}

    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)

    -- 2 instead of 1 because it supports batch input
    -- split method is a node method which will return 4 new nodes
    -- because nn.SplitTable(2)() will return 4 output nodes 
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    -- next is 'current', which will be used as input at the next timestep
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  -- inputs = {x , prev_c, prev_h}
  -- outputs = {next_c, next_h, logsoft}
  -- set up output 
  local top_h = outputs[#outputs]

  local logsoft = LSTM_Armour.Make_Output_Attention_Bilinear_Unit(rnn_size, word_embed_size, outputSize)({top_h, inputs[2]}) 
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs) 
end 


function LSTM_Armour.Make_Input_Attention_Bilinear_Unit(word_embed_size, word_embed_size, m) 
	
    local prev_word_embed = nn.Identity()()                     -- prev_word_embed: bz * 300 
	local As = nn.Identity()()                          --  bz * 16 * 300

    -- the number of attributes, we may change it to be 10
    local attention_output = nn.BilinearD3(word_embed_size, word_embed_size, 10, false)({prev_word_embed, As})  -- no bias
    -- attention_output = nn.Probe()(attention_output)
	local alpha = nn.SoftMax()(attention_output) -- bz * L 

    local g_in = LSTM_Armour.makeWeightedSumUnit()({As, alpha})  -- g_in: bz * 300   
   
    -- local temp = nn.CAddTable()({nn.CMul(word_embed_size)(g_in), prev_word_embed}) 
    local temp = nn.CAddTable()({g_in, prev_word_embed}) 

    local x_t = nn.Linear(word_embed_size, m)(temp) -- m is 512 for coco, xt: bz * 512 

	local inputs, outputs = {prev_word_embed, As}, {x_t}

	return nn.gModule(inputs, outputs)
end


function LSTM_Armour.Make_Output_Attention_Bilinear_Unit(hDim, word_embed_size, outputSize, dropout) 
	dropout = dropout or 0 
    local h_t = nn.Identity()()                     -- current h_t: bz * 512  
	local As = nn.Identity()()                          --  bz * 10 * 300
    -- the number of attributes, we may change it to be 10
    local attention_output = nn.BilinearD3(hDim, word_embed_size, 10, false)({h_t, nn.Tanh()(As)})  -- no bias
    -- attention_output = nn.Probe()(attention_output)
	local beta = nn.SoftMax()(attention_output) -- bz * L  

    local g_out = LSTM_Armour.makeWeightedSumUnit()({nn.Tanh()(As), beta})  -- g_out: bz * 300(d)  
    
    local temp = nn.CAddTable()({nn.Linear(word_embed_size, hDim)(nn.CMul(word_embed_size)(g_out)), h_t}) 
    
    if dropout > 0 then temp = nn.Dropout(dropout)(temp) end  

    local proj = nn.Linear(hDim, outputSize)(temp) -- proj: bz * outputSize 
    local logsoft = nn.LogSoftMax()(proj)

	local inputs, outputs = {h_t, As}, {logsoft}

	return nn.gModule(inputs, outputs)
end

return LSTM_Armour
