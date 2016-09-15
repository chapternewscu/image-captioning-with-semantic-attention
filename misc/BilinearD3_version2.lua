local BilinearD3, parent = torch.class('nn.BilinearD3', 'nn.Module') 

function BilinearD3:__init(inputSize1, inputSize2, outputSize, bias) 
    -- outputSize 16 
    parent.__init(self) 
    local bias = ((bias == nil) and true) or bias
    self.weight = torch.Tensor(inputSize1, inputSize2) 
    self.gradWeight = torch.Tensor(inputSize1, inputSize2) 
    if bias then 
        self.bias = torch.Tensor(1) 
        self.gradBias = torch.Tensor(1) 
    end 
    self.gradInput = {torch.Tensor(), torch.Tensor()} 
    self:reset() 
end 

function BilinearD3:reset(stdv) 
    assert(self) 
    if stdv then 
        stdv = stdv * math.sqrt(3) 
    else 
        stdv = 1 / math.sqrt(self.weight:size(2)) -- size(1) or size(2)??
    end
    self.weight:uniform(-stdv, stdv) 
    if self.bias then self.bias:uniform(-stdv, stdv) end 
    return self
end 

function BilinearD3:updateOutput(input)
    -- input[1]: bz * xDim 
    -- input[2]: bz * L * yDim
    assert(self) 
    -- compute output scores 
    self.output:resize(input[2]:size(1), input[2]:size(2)) -- bz * L
    
     -- print(input[2]:size())

    for k = 1, input[2]:size(2) do 
        local temp = torch.mm(self.weight, input[2]:select(2, k):t()) --  xDim * yDim, yDim * bz 
        temp:cmul(input[1]:t())  -- input[1]: bz * xDim, temp: xDim * bz 
        temp = temp:sum(1)  -- xDim
        if self.bias then
            temp = temp:add(self.bias:expand(input[1]:size(1)))
        end 
        self.output[{{}, k}] = temp:view(-1)
    end 
    return self.output
end 

function BilinearD3:updateGradInput(input, gradOutput) 
    assert(self)
    assert(gradOutput:dim() == 2) -- bz * L
    
    if self.gradInput then 
        -- compute d ouput /d input
        self.gradInput[1]:resizeAs(input[1]):fill(0) -- bz * xDim 
        self.gradInput[2]:resizeAs(input[2]):fill(0) -- bz * L * yDim 
       
        -- print(gradOutput) -- 80 * 16 
        -- print(self.gradInput) -- 80 * 300, 80 * 16 * 300

        for k = 1, gradOutput:size(2) do  
            local temp = torch.mm(input[2]:select(2, k), self.weight:t()) 
            temp:cmul(gradOutput:narrow(2, k, 1):expand(self.gradInput[1]:size(1), self.gradInput[1]:size(2)))  
            self.gradInput[1]:add(temp) 
            self.gradInput[2][{{},k, {}}]:addmm(1, input[1], self.weight) 
            self.gradInput[2][{{}, k, {}}]:cmul(gradOutput:narrow(2, k, 1):expand(self.gradInput[2]:size(1), self.gradInput[2]:size(3))) 
        end
    end 
    return self.gradInput 
end 

function BilinearD3:accGradParameters(input, gradOutput, scale)
    -- input[1]: bz * xDim 
    -- input[2]: bz * L * yDim 
    -- gradOutput: bz * L
    local scale = scale or 1 
    self.buff1 = self.buff1 or input[1].new() 
    self.buff1:resizeAs(input[1])

    -- accumulate parameter gradients 
    for k = 1, gradOutput:size(2) do -- L = gradOutput:size(2)
        torch.cmul(self.buff1, input[1], gradOutput:narrow(2, k, 1):expand(input[1]:size(1), input[1]:size(2)))
        -- it is okay not initialize self.gradWeight to zeros, because when training, we usually zero it before every iteration of training 
        -- to update it 
        self.gradWeight:addmm(self.buff1:t(), input[2][{{}, k, {}}]) -- self.buff1:bz * xDim, input[2][{{}, k, {}}]: bz * yDim 
    end 

    if self.bias then self.gradBias:add(scale, torch.Tensor{gradOutput:sum()}) end 
end 

-- we donot need to accumulate parameters when sharing 
BilinearD3.sharedAccUpdateGradParameters = BilinearD3.accUpdateGradParameters 


