---------------------------------------------------------------------------------
------------------------- adding attention unit ---------------------------------
--------------------------------------------------------------------------------
require 'nn'
require 'json'

local utils = require 'misc_saver.utils'
local net_utils = require 'misc_saver.net_utils'
local LSTM_Armour = require 'misc_saver.LSTM_Armour'
require 'misc_saver.LookupTableMaskZero'

local peek = require 'misc_saver.peek'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)
  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.word_encoding_size = utils.getopt(opt, 'word_encoding_size')
  self.image_encoding_size = utils.getopt(opt, 'image_encoding_size')

  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.att_size = utils.getopt(opt, 'attention_size', 128)
  self.num_layers = utils.getopt(opt, 'num_layers', 1)

  -- use glove 
  self.use_glove = utils.getopt(opt, 'use_glove', false)
  self.ix_to_word = utils.getopt(opt, 'ix_to_word', '') -- required
  self.glove_path = utils.getopt(opt, 'glove_path', '') -- required 
  self.glove_dim = utils.getopt(opt, 'glove_dim', '')


  local dropout = utils.getopt(opt, 'dropout', 0)
  
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')

  -- for very first time step, just a linear layer 

  self.linear_model = nn.Linear(self.image_encoding_size, self.rnn_size)

  -- input attention model 
  -- inputs = {prev_word_embed, As} 
  -- outputs: {x_t} 
  self.input_attention_model = LSTM_Armour.Make_Input_Attention_Bilinear_Unit(self.word_encoding_size, self.word_encoding_size, self.rnn_size) 
  -- lstm network.
  -- inputs: {x, As, prev_c, prev_h} 
  -- outputs: {next_c, next_h, logsoft} 
  self.lstm_att= LSTM_Armour.lstm_with_output_attention(self.word_encoding_size, self.rnn_size, self.rnn_size,self.vocab_size+1, self.num_layers, dropout)

  -- self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.word_encoding_size)
  self.lookup_table = nn.LookupTableMaskZero(self.vocab_size + 1, self.word_encoding_size)
  
  -- initialize self.lookup_table with glove embeddings 
  if self.use_glove then
    print('loading glove word vectors ... ')
    local word_to_ix = utils.invert_key_value(self.ix_to_word)
    local glove_table = net_utils.load_glove(self.glove_path, self.glove_dim)
    for ix, word in pairs(self.ix_to_word) do 
        if word == 'UNK' then -- 'UNK' in our case corresponds to '<unk>' in our glove table, that is one difference i found in my case 
            -- index add 1 because it is nn.LookuptableMaskZero, not nn.LookupTable
            self.lookup_table.weight[tonumber(ix)+1] = glove_table['unk']
        else
            if glove_table[word] == nil then 
                print(word .. ' not exists ' .. 'in glove files')
                self.lookup_table.weight[tonumber(ix)+1] = torch.Tensor(300):uniform(-1, 1)
            else 
                self.lookup_table.weight[tonumber(ix)+1] = glove_table[word]
            end 
        end 
    end 
    print('initialization lookuptable done')
  end 

  -- use pretrained glove vectors to initialize the self.lookup_table
  self:_createInitState(1) -- will be lazily resized later during forward passes
end


function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

-- should i add attention model hereby??? or more specifically, should i clone the attend unit 
function layer:createClones() 
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  -- note that just 1 element in self.linear_models to be used at the very first step 
  self.linear_models = {self.linear_model}
  self.input_attention_model_clones = {self.input_attention_model} 
  self.lstm_clones = {self.lstm_att}
  self.lookup_tables = {self.lookup_table}

  for t=2,self.seq_length+1 do  -- we will input the START Token and image at the same time
    self.input_attention_model_clones[t] = self.input_attention_model:clone('weight', 'bias', 'gradWeight', 'gradBias') 
    self.lstm_clones[t] = self.lstm_att:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end

  -- add an additional lookup_tables for attributes vector 
  self.lookup_tables[#self.lookup_tables + 1] = self.lookup_table:clone('weight', 'gradWeight')
end

function layer:getModulesList()
  return {self.linear_model, self.lookup_table, self.input_attention_model, self.lstm_att}
end

function layer:parameters()
  -- we only have two internal modules, return their params
 
  local p1,g1 = self.linear_model:parameters()
  local p2,g2 = self.lookup_table:parameters()
  local p3,g3 = self.input_attention_model:parameters()  
  local p4,g4 = self.lstm_att:parameters()
 
  local params = {}
  -- params of input_attention_model, output_attention_model, lstm and lookup table
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  for k,v in pairs(p3) do table.insert(params, v) end
  for k,v in pairs(p4) do table.insert(params, v) end

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end
  for k,v in pairs(g4) do table.insert(grad_params, v) end
   
  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)
  return params, grad_params
end

function layer:training()
  if self.lstm_clones==nil then self:createClones() end -- create these lazily if needed, just use self.lstm_clones hereby. 
                                                        -- because we aways clone all the stuff needed at the same time 
  for k,v in pairs(self.linear_models) do v:training() end 
  for k,v in pairs(self.input_attention_model_clones) do v:training() end
  for k,v in pairs(self.lstm_clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end

end

function layer:evaluate()
  if self.lstm_clones == nil then self:createClones() end -- create these lazily if needed
 
  for k,v in pairs(self.linear_models) do v:evaluate() end 
  for k,v in pairs(self.input_attention_model_clones) do v:evaluate() end
  for k,v in pairs(self.lstm_clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end

end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
-- may convert imgs to glimpse in the future
-- sample, of course no need to clone  
function layer:sample(input, opt)
  local imgs = input[1] 
  local attrs = input[2]
  
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  local batch_size = imgs:size(1)
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step

  -- in the future, we will use an exclusive lookup_tables 
  -- to embedding the attribute words 
  self.As = self.lookup_table:forward(attrs):clone() 

  for t=1,self.seq_length+1 do

    local x1, xt, it, sampleLogprobs
    if t == 1 then  -- input image 
      x1 = self.linear_model:forward(imgs) 
    else  -- t > 1 
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end

      x_word = self.lookup_table:forward(it)
      xt = {x_word, self.As}
    end   -- prepare inputs 
    
    -- do recording stuff
    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    
    -- input 
    if t ==1 then 
        -- out: {next_c, next_h, logsoft}
        local inputs = {x1, self.As, unpack(state)}
        local out = self.lstm_att:forward(inputs)  
        -- forward to output_attetion_model 
        logprobs = out[self.num_state + 1]
        state = {} 
        for i = 1, self.num_state do table.insert(state, out[i]) end         
    else -- t > 1  
        -- forward to input_attention_model 
        local g = self.input_attention_model:forward(xt)
        -- forward to lstm
        local inputs = {g, self.As, unpack(state)} 
        local out = self.lstm_att:forward(inputs) 
        -- forward to output attention
        logprobs = out[self.num_state + 1]
        state = {} 
        for i = 1, self.num_state do table.insert(state, out[i]) end

    end 

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end


--[[ add beam search later 
--
--]] 

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+1)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+1 because +1 for first dummy 
img and START/END tokens shift)
--]]
-- may convert imgs to glimpse in the futures
function layer:updateOutput(input)
  --print(input)
  local imgs = input[1]  -- 10 * 11, bz * img_enc 
  local seq = input[2]   -- 7 * 10, seq_len * bz 
  local attrs = input[3] -- 10 * 10, bz * top_attr
  
  --peek.peek_here(attrs) 
  if self.lstm_clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+1, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}

  self.lookup_tables_inputs = {}
   
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency

  -- add: self.As
  -- forward attrs to get attribute vectors {A_{i}}
  -- we just need to use only the first clone of lookup_tables
  -- output will be: bz * 16(or 10 attrs) * 256(word vector length)
  -- print(attrs)
  -- currently, we need to clone self.lookup_tables[1]'s output 
  -- to avoid later override's problem 
  -- in the future we may consider to use anoter clone of lookup_tables
  -- to process attritributed embeddings
  -- Now, we use the last extra clone of lookup_table, ie, self.lookup_tables[#self.lookup_tables]
  self.As = self.lookup_tables[#self.lookup_tables]:forward(attrs)  --bz * top_attrs * 11, ie, 10x10x11  
  
  self.input_attention_in = {} 
  self.lstm_inputs = {} 

  for t=1,self.seq_length+1 do
    local can_skip = false
    local x1, xt

    if t == 1 then  -- input image and START token 
      -- feed in the images 
      -- feed in the start tokens
      -- zero out self.As 
      -- local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      x1 = self.linear_models[t]:forward(imgs) -- bz * 8 
    else -- t > 1 
      -- feed in the rest of the sequence...
      local it = seq[t-1]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        local x_word = self.lookup_tables[t]:forward(it)
        xt = {x_word, self.As}
      end
    end
 
    if not can_skip then
      -- construct the inputs
      -- using unpack twice in a table will give error, what about flatten table
      if t == 1 then 
        self.lstm_inputs[t] = {x1, self.As:clone(), unpack(self.state[t-1])}  -- bz * 8 , v1  
        -- just input the image to self.lstm_clones[1] 
        -- out: {next_c, next_h, logsoft}
        local out = self.lstm_clones[t]:forward(self.lstm_inputs[t])  
        
        self.output[t] = out[self.num_state + 1]

        self.state[t] = {} 
        for i = 1, self.num_state do table.insert(self.state[t], out[i]) end 

      else -- t > 1
        self.input_attention_in[t] = xt
        -- input to the attention model 
        local g = self.input_attention_model_clones[t]:forward(xt)

        self.lstm_inputs[t] = {g, self.As,unpack(self.state[t-1])}
        -- out = {next_c, next_h, logsoft}  
        local out = self.lstm_clones[t]:forward(self.lstm_inputs[t])
        self.output[t] = out[self.num_state + 1]
        
        self.state[t] = {} 
        for i = 1, self.num_state do table.insert(self.state[t], out[i]) end 
      end

      self.tmax = t
    end
  end
  
  return self.output
end

--[[
gradOutput is an (D+1)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput) 
  local dimgs = nil  -- grad on input images
  local dAs = nil -- grad on the attrs 

  -- go backwards and lets compute gradients  
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros  

  for t=self.tmax,1,-1 do

    -- concat state gradients and output vector gradients at time step t
    local dout = {}

    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    
    local dinputs = self.lstm_clones[t]:backward(self.lstm_inputs[t], dout)

    -- split the gradient to dg_t and to state 
    local dg_t = dinputs[1] 
    if dAs == nil then  
        dAs = torch.Tensor():typeAs(dinputs[2]):resizeAs(dinputs[2]):copy(dinputs[2])
    else 
        dAs = dAs + dinputs[2]
    end

    dstate[t-1] = {} 
    for k = 3, self.num_state + 2 do table.insert(dstate[t-1], dinputs[k]) end 
 
    -- backprop to self.lstm_clones
    if t == 1 then 
        -- self.glimpses[1]  is output of self.linear_model  with input as imgs
        dimgs = self.linear_models[t]: backward(input[1], dg_t) 
    else
        -- backward to self.input_attention_models 
        local dattention_out = self.input_attention_model_clones[t]:backward(self.input_attention_in[t], dg_t)
        
        local dword_t, dAs_t2 = dattention_out[1], dattention_out[2]
        local it = self.lookup_tables_inputs[t]
        self.lookup_tables[t]:backward(it, dword_t) -- backprop into lookup table
        dAs = dAs + dAs_t2 
    end

  end

  self.lookup_tables[#self.lookup_tables]:backward(input[3],dAs)

  -- do we need to update gradients to self.lookup_tabls[1] using dAs? currently, let us say 'no' for simplicity
  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor(), torch.Tensor()}
  return self.gradInput

end


-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 2 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do 
      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t-1 is correct, since at t=1 image, we want to predict the first token (and 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end

  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n) -- has computed gradInput here
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
