-- Custome Alpha View: [bz * L] --> [bz * L * 1]
local CustomAlphaView, parent = torch.class('nn.CustomAlphaView', 'nn.Module') 
function CustomAlphaView:__init() 
    parent.__init(self)
end 

function CustomAlphaView:updateOutput(input) 
    assert(input:dim() == 2) 
    self.output = input:view(input:size(1), input:size(2), 1)
    return self.output 
end 

function CustomAlphaView:updateGradInput(input, gradOutput)
   
    self.gradInput = gradOutput:view(gradOutput:size(1), gradOutput:size(2))
    
    return self.gradInput 
end 