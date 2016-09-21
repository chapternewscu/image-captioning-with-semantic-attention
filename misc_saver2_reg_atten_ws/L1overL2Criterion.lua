
local L1overL2Criterion, parent = torch.class('nn.L1overL2Criterion', 'nn.Criterion')

function L1overL2Criterion:__init(alpha, beta)
    parent.__init(self)
    self.alpha = alpha
    self.beta = beta or 0.1
end

-- calculates the L1 norm over the L2 norm of input and returns it
function L1overL2Criterion:updateOutput(input)
    self.output = self.alpha * (torch.norm(input, 1) + self.beta)/ math.sqrt(torch.norm(input)^2 + self.beta^2)
    return self.output
end

-- calculates the gradient of the L1 norm over the L2 norm of input w.r.t input and returns it
function L1overL2Criterion:updateGradInput(input)
    local l1 = torch.norm(input, 1) + self.beta
    local l2 = math.sqrt(torch.norm(input)^2 + self.beta^2)
    self.gradInput = torch.add(torch.mul(torch.sign(input), self.alpha / l2), torch.mul(input, -self.alpha * l1 / (l2^3)))
    return self.gradInput
end
