-- this file exists only for ease of converting 
-- semmantic attributes index which are 0 can be zero vector 

local LookupTableMaskZero, parent = torch.class('nn.LookupTableMaskZero', 'nn.LookupTable')

function LookupTableMaskZero:__init(nIndex, nOutput)
  parent.__init(self, nIndex + 1, nOutput)
end

function LookupTableMaskZero:updateOutput(input)
	self.weight[1]:zero() -- very time we index it, it is an zero vector
	return parent.updateOutput(self, torch.add(input, 1))
end

-- No need to override accGradParameters because input is cached
-- by nn.LookupTable implementation and gradOuput is already as expected
--[[function LookupTable:accGradParameters(input, gradOutput, scale)
	parent.accGradParameters(self, torch.add(input, 1), gradOutput, scale)
end--]]
