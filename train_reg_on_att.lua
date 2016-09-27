require 'torch'
require 'nn'
require 'nngraph'

-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc_saver2_reg_atten_ws.utils'
require 'misc_saver2_reg_atten_ws.DataLoader'
require 'misc_saver2_reg_atten_ws.LanguageModel'
local net_utils = require 'misc_saver2_reg_atten_ws.net_utils'
require 'misc_saver2_reg_atten_ws.optim_updates'
require 'misc_saver2_reg_atten_ws.Attention_Weights_Criterion'
require 'misc_saver2_reg_atten_ws.L1Criterion'

-- nngraph.setDebug(true)

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-silent', false, 'print opt to the screen?')
-- data settings 
cmd:option('-input_h5', '', 'dirctory of dataset')
cmd:option('-input_json', '', 'dirctory of json file')

-- glove vector 
cmd:option('-use_glove', true, 'whether to use glove vector')
cmd:option('-glove_path','', 'specify glove vector data path')
cmd:option('-glove_dim',300, 'glove vetor dimension, by default, use 300 dimension')

-- VGG16 CNN model 
cmd:option('-cnn_proto','../cnn_model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','../cnn_model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-optim_state_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',128,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-word_encoding_size',128,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-image_encoding_size',128,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-attention_size', 128, 'attention size of the attention unit')
-- Optimization: General
cmd:option('-max_iters',-1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',8,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', 0, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Language Model
-- sgdmom: sgd with nesterov update 
-- sgdm: sgd with momentum, standard update
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',1e-5, 'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 12000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', 5000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?, 2500 by default')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-save_optim_state', true, 'save optim state and next id in data to restore training')

cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()


-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)

  
if not  opt.silent then 
	print(opt)
end 

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
--[[  
isDebug = false  
if isDebug then 
    -- do nothing, no loading the data 
else
--]] 
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json} 

--[[
end 
--]]

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------

local protos = {}

--[[
isDebug = false   
if isDebug then
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = 1000
  lmOpt.word_encoding_size = opt.word_encoding_size
  lmOpt.image_encoding_size = opt.image_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.attention_size = opt.attention_size 
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = 16
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled

  local load_cnn = true  
  if load_cnn then 
    local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
    protos.cnn = net_utils.build_cnn_v1(cnn_raw, {encoding_size = opt.image_encoding_size, backend = cnn_backend})
  else 
    protos.cnn = nn.Sequential() 
    protos.cnn:add(nn.Linear(224, opt.image_encoding_size)) 
  end 

  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()

else 
--]]


if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  
  local language_crit = nn.LanguageModelCriterion()
  local atten_weights_crit = nn.Attention_Weights_Criterion() 
  local atten_weights_crit2 = nn.Attention_Weights_Criterion() 
  -- local atten_weights_crit = nn.L1Criterion()

  -- set repeatTarget to be true to avoid slicing problem on target 
  protos.crit = nn.ParallelCriterion(true)  -- no in checkpoints, create manually 

  protos.crit:add(language_crit, 1) -- weights for the langauge criterion  
  
  -- will we add the two loss together
  -- currently not add the loss of the attention weights 
  protos.crit:add(atten_weights_crit, 0) -- weights for the attention weights criterion 
  protos.crit:add(atten_weights_crit2, 0) 

  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.word_encoding_size = opt.word_encoding_size
  lmOpt.image_encoding_size = opt.image_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.attention_size = opt.attention_size 
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
    
  -- glove parameters 
  lmOpt.use_glove = opt.use_glove
  lmOpt.ix_to_word = loader:getVocab()  
  lmOpt.glove_path = opt.glove_path 
  lmOpt.glove_dim = opt.glove_dim 

  protos.lm = nn.LanguageModel(lmOpt)

  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
  protos.cnn = net_utils.build_cnn_v1(cnn_raw, {encoding_size = opt.image_encoding_size, backend = cnn_backend})
 
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  
  -- criterion for the language model
  -- set repeatTarget to be true to avoid slicing problem 
  protos.crit = nn.ParallelCriterion(true)  -- no in checkpoints, create manually 

  local language_crit = nn.LanguageModelCriterion()
  local atten_weights_crit = nn.Attention_Weights_Criterion()  
  local atten_weights_crit2 = nn.Attention_Weights_Criterion()
  
  -- local atten_weights_crit = nn.L1Criterion()
  protos.crit:add(language_crit, 1) -- weights for the language criterion 

  -- will we add the two loss together??
  protos.crit:add(atten_weights_crit, 0) -- weights for the attention weights criterion 
  protos.crit:add(atten_weights_crit2, 0)
end

--[[
end
--]]

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

--[[ 
isDebug = false   
test_output = nil  
test_expanded = nil
if isDebug then
    local load_cnn = true 
    if load_cnn then 
        test_input = torch.CudaTensor(5, 3, 224, 224):uniform(-1, 1)
    else
        test_input = torch.CudaTensor(5, 224):uniform(-1, 1)
    end 

    -- test_output will be: 5 * 512 * 14 * 14 
    test_output = protos.cnn:forward(test_input)
    test_expended = protos.expander:forward(test_output)
end 
--]] 

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()

thin_lm.lstm_att:share(protos.lm.lstm_att, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
thin_lm.input_attention_model:share(protos.lm.input_attention_model, 'weight', 'bias')
thin_lm.linear_model:share(protos.lm.linear_model, 'weight', 'bias')

local thin_cnn = protos.cnn:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = 8, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- expand the data.semantic_words: batch_size * 16(attribute words per image) 
    local exp_attrs = protos.expander:forward(data.semantic_words):clone()  

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats):clone() 

    local logprobs, attrs, attrs_alpha = protos.lm:forward{expanded_feats, data.labels, exp_attrs}

    local loss = protos.crit:forward({logprobs, attrs, attrs_alpha}, data.labels)
    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    -- input: feats, data.semantic_words 
    local seq = protos.lm:sample({feats, data.semantic_words})

    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        print(string.format('image %s: %s', entry.image_id, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
   
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end
  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0

local function lossFun()
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()

  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img
  
  -- expand the data.semantic_words: batch_size * 16(attribute words per image) 
  local exp_attrs = protos.expander:forward(data.semantic_words):clone()  
 

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  
  -- we have to expand out image features, once for each sentence
  -- use protos.expander twice
  local expanded_feats = protos.expander:forward(feats):clone() 
  -- forward the language model
  -- input: expanded_feats, data.semantic_words 
 
  local logprobs, atts, atts_alpha = protos.lm:forward{expanded_feats, data.labels, exp_attrs}
  -- forward the language model criterion
  local loss = protos.crit:forward({logprobs, atts, atts_alpha}, data.labels)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local gradOutput = protos.crit:backward({logprobs, atts, atts_alpha}, data.labels)

  -- backprop language model
  local dexpanded_feats, ddummy, ddummy = unpack(protos.lm:backward({expanded_feats, data.labels, exp_attrs}, gradOutput))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- note that 
  -- weight decay comes from adding penalty term, here just lambda/2 * |w|^2 to the total loss functions 
  -- when compute deriative terms, just plus -lr * lambda * w to the gradParams.  
  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score

-- loading save optimization state to restore training from last training 
if string.len(opt.start_from) > 0  and string.len(opt.optim_state_from) > 0 then 
    optim_state_from = torch.load(opt.optim_state_from)
    optim_state = optim_state_from.optim_state 
    cnn_optim_state = optim_state_from.cnn_optim_state 
    -- restore next train data iterators 
    loader.iterators['train'] = optim_state_from.next_id  
end 

--[[
   isDebug = false      
   if isDebug then 
    -- do nothing 
    else 
--]] 

while true do  
  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then

    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end

    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
         
        -- storing optimization state(cnn_optim_state and optim_state) so that we can restore it 
        if opt.save_optim_state then 
            local optim_checkpoint_path = path.join(opt.checkpoint_path, 'optim_id' .. opt.id)
            local optim_checkpoint = {} 
            optim_checkpoint.cnn_optim_state = cnn_optim_state 
            optim_checkpoint.optim_state = optim_state  
            -- storing next index in training data 
            optim_checkpoint.next_id = loader.iterators['train']

            torch.save(optim_checkpoint_path .. '.t7', optim_checkpoint)
            print('wrote checkpoint to ' .. optim_checkpoint_path .. '.t7')
        end
      end
    end

    -- storing optimization state(cnn_optim_state and optim_state) so that we can restore it 
    if opt.save_optim_state then 
        local optim_checkpoint_path = path.join(opt.checkpoint_path, 'optim_id_latest' .. opt.id)
        local optim_checkpoint = {} 
        optim_checkpoint.cnn_optim_state = cnn_optim_state 
        optim_checkpoint.optim_state = optim_state  
        -- storing next index in training data 
        optim_checkpoint.next_id = loader.iterators['train']

        torch.save(optim_checkpoint_path .. '.t7', optim_checkpoint)
        print('wrote checkpoint to ' .. optim_checkpoint_path .. '.t7')
    end

  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  
  --[[
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.1, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end
  --]] 
  --[[
  -- currently only training lm model 
  if iter < 12000 then 
        learning_rate = 0.01
  elseif iter < 20000 then 
        learning_rate = 0.01 
  else 
        learning_rate = 0.001
  end 
  --]]

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end
 
--[[ 
end 
--]] 
