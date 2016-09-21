-- class CAddTableD2D3
local CAddTableD2D3, parent = torch.class('nn.CAddTableD2D3', 'nn.Module')
function CAddTableD2D3:__init() 
    parent.__init(self) 
    self.gradInput = {} 
end 

function CAddTableD2D3:updateOutput(input) 
    -- input[1]: bz * Dh 
    -- input[2]: bz * L * Dh
    assert(type(input) == 'table' and #input == 2) 
    local hProj, xProj = unpack(input)
    assert(hProj:dim()==2 and xProj:dim()==3) 

    self.output:resizeAs(xProj)
    self.output:copy(xProj)
    
    local L = xProj:size(2)  -- xProj1, xProj2, ..., where index i means the i-th time step  

    for k = 1, L do 
        self.output:narrow(2, k, 1):add(hProj) 
    end 

    return self.output 
end 

function CAddTableD2D3:updateGradInput(input, gradOutput) 
    -- gradOutput: [bz * L * Dh] 
    assert(gradOutput:dim() == 3)
    local hProj, xProj = unpack(input) 

    for i = 1, #input do 
        self.gradInput[i] = self.gradInput[i] or input[i].new() 
        self.gradInput[i]:resizeAs(input[i])
    end 
    
    -- same reference 
    grad_h, grad_x = unpack(self.gradInput) 
    grad_h:fill(0)

    local L = xProj:size(2) 
   
    -- accumulated gradients w.r.t the hiddent state at previsous timestep: h_{i-1}
    for t = 1, L do 
        grad_h:add(gradOutput:narrow(2, t, 1))    
    end 
    
    -- just copy the gradients w.r.t input x_{1, 2, 3, ..., L} for attention based model at each time step 
    grad_x:copy(gradOutput)

    return self.gradInput 
end 
