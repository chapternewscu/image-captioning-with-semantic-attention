require 'nn'
require 'misc.DataLoader' 

local utils = require 'misc.utils'
local net_utils  = require 'misc.net_utils'

loader = DataLoader{h5_file = '../coco_data/cocotalk.h5', json_file = '../coco_data/cocotalk.json'}


local use_glove = true 

test_glove_weights = nil
test_glove_table = nil 

if use_glove then 
    -- vocab is: ix_to_word 
    -- ex: ix_to_word[tostring(1)] = 'woods' 
    local ix_to_word = loader:getVocab()
    local word_to_ix = utils.invert_key_value(ix_to_word)

    local vocab_size = loader:getVocabSize() 
    

    -- here, we use glove vector of dimension 300
    local glove_weights = torch.Tensor(vocab_size+1, 300) 
    -- note here that lookuptable.weights is of size (vocab_size + 1, 300), +1 means the '<END>' token, which corresponds to index vocab_size+1 

    -- glove_table['word'] = vector
    local glove_table = net_utils.load_glove('..//glove_word2vec/glove.6B.300d.txt', 300)
    
    test_glove_table = glove_table 

    for ix, word in pairs(ix_to_word) do 
        if word == 'UNK' then -- 'UNK' in our case corresponds to '<unk>' in our glove table, that is one difference i found in my case 
            glove_weights[tonumber(ix)] = glove_table['unk']
        else
            if glove_table[word] == nil then 
                print(word .. ' not exists ' .. 'in glove files')
                glove_weights[tonumber(ix)] = torch.Tensor(300):uniform(-1, 1)
            else 
                glove_weights[tonumber(ix)] = glove_table[word]
            end 
        end 
    end 
    test_glove_weights = glove_weights 
end 

