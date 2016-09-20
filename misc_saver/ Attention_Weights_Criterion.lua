-- ref. paper: image caption with semantic attentions 
-- this file implements the g(a) criterion which is used to 
-- regularize the attention weights to enforce the completeness of 
-- attention as well as sparsity of attention at any particular timestep 
-- ie, minimize matrix norms of attention weights

local Attention_Weights_Criterion, parent = torch.class('nn.Attention_Weights_Criterion', 'nn.Criterion')

function  Attention_Weights_Criterion:__init() 
    parent.__init(self)     
end 


-- input is the attention weights matrix of size: bz x L x top_attrs
function  Attention_Weights_Criterion:updateOutput(input)
    assert(input:dim() == 3)
    local bz, L, top_attrs = unpack(input:size():totable()) 
    local sum_power_along_L = torch.pow(torch.sum(input, input:dim()-1), 2) -- bz x 1 x top_attrs 
    local sum_sqrt_along_top = torch.pow(torch.sum(sum_power_along_L, input:dim()), 0.5) -- bz x 1 x 1  
    -- squeeze method will remove all singleton dimenstion 
    sum_sqrt_along_top = sum_sqrt_along_top:squeeze() -- bz  

    local sqrt_sum_top = torch.sum(torch.pow(input, 0.5), input:dim()) 
    local power_sum_along_L = torch.sum(torch.pow(sqrt_sum_top, 2), input:dim() -1) -- bz x 1 x 1 
    -- squeeze singleton dimention  
    power_sum_along_L = power_sum_along_L:squeeze() -- bz  

    -- self.output is a value 
    self.output = torch.sum(sum_sqrt_along_top + power_sum_along_L) 
    return self.output 
end 

-- caculates gradient w.r.t input 
-- self.gradInput: bz x L x top_attrs
function Attention_Weights_Criterion:updateGradInput(input) 
    -- difficulities: how to compute the gradients w.r.t input 
    self.gradInput:resizeAs(input) 
    
    -- 1st step:
    local first_term_1 = torch.pow(torch.sum(torch.pow(torch.sum(input, input:dim()-1), 2), input:dim()), -0.5)-- bz x 1 x 1  
    -- 2nd step: 
    local first_term_2 = torch.sum(input, input:dim()-1) -- bz x 1 x top_attrs 
    -- 3rd step:
    local second_term_1 = torch.sum(torch.pow(input, 0.5), input:dim()) -- bz x L x 1 
    -- 4th step:
    local second_term_2 = torch.pow(input, -0.5) -- bz x L x top_attrs
    
    self.gradIntput = torch.cmul(first_term_1:expandAs(input), first_term_2:expandAs(input)) + torch.cmul(second_term_1:expandAs(input), second_term_2)
    return self.gradInput 
end 
