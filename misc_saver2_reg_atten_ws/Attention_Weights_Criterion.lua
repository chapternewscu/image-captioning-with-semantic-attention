-- ref. paper: image caption with semantic attentions 
-- this file implements the g(a) criterion which is used to 
-- regularize the attention weights to enforce the completeness of 
-- attention as well as sparsity of attention at any particular timestep 
-- ie, minimize matrix norms of attention weights

local Attention_Weights_Criterion, parent = torch.class('nn.Attention_Weights_Criterion', 'nn.Criterion')

function  Attention_Weights_Criterion:__init(eps) 
    parent.__init(self)
    -- it is very import to avoid dividing zero problem 
    self.eps = eps or 1e-5 -- by default self.alpha is set to be 0.01 
end 


-- input is the attention weights matrix of size: bz x L x top_attrs
-- target is nil currently and no use 
function  Attention_Weights_Criterion:updateOutput(input, target) 
    assert(input:dim() == 3)

    local bz, L, top_attrs = unpack(input:size():totable()) 

    -- fist term 
    -- sum along L 
    local sum_along_L = torch.sum(input, input:dim()-1) -- bz x 1 x top_attrs 
    local pow_sum_L = sum_along_L:pow(2) 
    local sum_top_pow_sum_L = torch.sum(pow_sum_L, input:dim()) 
    local sqrt_sum_top_pow_sum_L = sum_top_pow_sum_L:sqrt() -- bz x 1 x 1 

    --second term 
    local sqrt_input = torch.sqrt(input + self.eps) 

    local sum_along_top = torch.sum(sqrt_input, input:dim())
    local pow_sum_along_top = sum_along_top:pow(2) 
    local sum_L_pow_sum_top = torch.sum(pow_sum_along_top, input:dim()-1) -- bz x 1 x 1  
    
    -- final results 
    self.output = torch.sum(sqrt_sum_top_pow_sum_L+sum_L_pow_sum_top) 

    return self.output 
end 

-- caculates gradient w.r.t input 
-- self.gradInput: bz x L x top_attrs
-- currently target is useless 
function Attention_Weights_Criterion:updateGradInput(input, target) 
    -- difficulities: how to compute the gradients w.r.t input 
    self.gradInput:resizeAs(input) -- bz x L x top_attrs

    for b = 1, self.gradInput:size(1) do  -- iterate over batch data of size bz 
        local sum_L = torch.sum(input[b], input[b]:dim()-1) 
        local pow_sum_L = sum_L:pow(2) 
        local sum_top_pow_sum_L =  torch.sum(pow_sum_L, pow_sum_L:dim())
       
        local multiplier11 = sum_top_pow_sum_L:pow(-0.5) -- 1 x 1  
        local multiplier12 = torch.sum(input[b], input[b]:dim()-1) -- 1 x top_attrs 
       
        local sqrt_input = torch.sqrt(input[b] + self.eps) 
        local multiplier21 = torch.sum(sqrt_input, sqrt_input:dim()) -- L x 1  
        local multiplier22 = sqrt_input:cinv()  -- L * top_attrs  -- inplace 

        local first_term = torch.cmul(multiplier11:expandAs(input[b]), multiplier12:expandAs(input[b]))
        -- error code, remains here to avoid my self that the following doesnot work
        -- local second_term = torch.cmul(multiplier21:expandAs(input[b]):cmul(multiplier22)
        local second_term = torch.cmul(multiplier21:expandAs(input[b]), multiplier22)
       
        --[[
        local first_term = torch.pow(torch.sum(torch.pow(torch.sum(input[b], 1), 2), 2), -0.5) * torch.sum(input[b], 1)
        first_term = first_term:expandAs(input[b])
        local second_term = torch.cmul(torch.sum(torch.pow(input[b], 0.5), 2):expandAs(input[b]), torch.pow(input[b], -0.5))
        --]]
        self.gradInput[b]:copy(first_term):add(second_term)
    end
    
    -- not worked!!!, may check it later
    --[[
    -- 1st step:
    local first_term_1 = torch.pow(torch.sum(torch.pow(torch.sum(input, input:dim()-1), 2), input:dim()), -0.5)-- bz x 1 x 1  
    -- 2nd step: 
    local first_term_2 = torch.sum(input, input:dim()-1) -- bz x 1 x top_attrs 
    -- 3rd step:
    local second_term_1 = torch.sum(torch.pow(input, 0.5), input:dim()) -- bz x L x 1 
    -- 4th step:
    local second_term_2 = torch.pow(input, -0.5) -- bz x L x top_attrs
    
    self.gradIntput = torch.cmul(first_term_1:expandAs(input), first_term_2:expandAs(input)) + torch.cmul(second_term_1:expandAs(input), second_term_2)
    --]]
    return self.gradInput 
end 
