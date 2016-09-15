-- LinearTensorD3: [bz * L * xDim] --> [bz * L * oDim]
local LinearTensorD3, parent = torch.class('nn.LinearTensorD3', 'nn.Linear')

function LinearTensorD3:__init(inputSize, outputSize) 
    parent.__init(self, inputSize, outputSize) 
end 

function LinearTensorD3:updateOutput(input)
    --input: bz * L * xDim
    --ouput: bz * L * oDim
    assert(input:dim() == 3)
    local bz, L, xDim = unpack(input:size():totable())
    self.output:resize(bz*L, self.bias:size(1))
    local inputView = input:view(bz*L, xDim)
    self.output:copy(parent.updateOutput(self, inputView))
    self.output:resize(bz, L, self.bias:size(1))

    return self.output
end 

function LinearTensorD3:updateGradInput(input, gradOutput) 
    assert(gradOutput:dim() == 3) 
    local bz, L, xDim = unpack(input:size():totable())
    local inputView = input:view(bz*L, xDim)
    local gradOutputView = gradOutput:view(bz*L, self.bias:size(1)) 

    self.gradInput:resize(bz*L, xDim) 
    self.gradInput:copy(parent.updateGradInput(self, inputView, gradOutputView))
    self.gradInput:resize(bz, L, xDim)
    return self.gradInput 
end 

function LinearTensorD3:accGradParameters(input, gradOutput, scale)
    scale = scale or 1 
    local bz, L, xDim = unpack(input:size():totable()) 

    local inputView = input:view(bz*L, xDim)
    local gradOutputView = gradOutput:view(bz*L, self.bias:size(1))
    parent.accGradParameters(self, inputView, gradOutputView, scale)
end 
