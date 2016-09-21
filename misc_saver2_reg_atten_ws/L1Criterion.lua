require('nn')

local L1Criterion, parent = torch.class('nn.L1Criterion', 'nn.Criterion')

function L1Criterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha
end

-- calculates the L1 norm of input and returns it
function L1Criterion:updateOutput(input)
    self.output = self.alpha * torch.norm(input, 1)
    return self.output
end

-- calculates the gradient of the L1 norm of input w.r.t input and returns it
function L1Criterion:updateGradInput(input)
    self.gradInput = torch.mul(torch.sign(input), self.alpha)
    return self.gradInput
end