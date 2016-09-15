local Probe, parent = torch.class('nn.Probe', 'nn.Module')

function Probe:__init()
    
end

function Probe:updateOutput(input)
    if type(input) == 'table' then
        print(input)
    else
        print(tensorInfo(input, 'input'))
    end
    -- debugger.enter()
    self.output = input
    return self.output
end

function Probe:updateGradInput(input, gradOutput) 
    print('backward')
    if type(input) == 'table' then
        print(gradOutput)
    else
        print(tensorInfo(gradOutput, 'gradOutput'))
    end
    self.gradInput = gradOutput
    -- debugger.enter()
    
    return self.gradInput
end
