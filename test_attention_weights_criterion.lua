require 'torch'
require 'nn'
require 'misc_saver2_reg_atten_ws.Attention_Weights_Criterion' 
--[[
require 'misc_saver2_reg_atten_ws.L1Criterion' 
require 'misc_saver2_reg_atten_ws.L1overL2Criterion' 
--]]

local mytester = torch.Tester() 
local jac 

local precision = 1e-5 
local expprecision = 1e-4 

local criterion_test = torch.TestSuite() 

local function criterionJacobianTest(cri, input)
   local eps = 1e-6
   local _ = cri:forward(input)
   local dfdx = cri:backward(input)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

--[[ passed 
function criterion_test.L1Criterion() 
    local bz = 4
    local L = 6
    -- local top_attrs = 2

    -- local input = torch.rand(bz, L, top_attrs) 
     local input = torch.rand(bz, L)

    -- local cri = nn.Attention_Weights_Criterion()
    local cri = nn.L1Criterion(0.1)

    print('Attention_Weights_Criterion test')

    criterionJacobianTest(cri, input)
end 
--]] 

-- passed
--[[
function criterion_test.L1overL2Criterion() 
    local bz = 4
    local L = 6

    -- local input = torch.rand(bz, L, top_attrs) 
     local input = torch.rand(bz, L)

    -- local cri = nn.Attention_Weights_Criterion()
    local cri = nn.L1overL2Criterion(0.1)

    print('Attention_Weights_Criterion test')

    criterionJacobianTest(cri, input)
end 
--]]

-- passed 
function criterion_test.Attention_Criterion() 
    local bz = 4
    local L = 6
    local top_attrs = 2

    local input = torch.rand(bz, L, top_attrs) 

    local cri = nn.Attention_Weights_Criterion()

    print('Attention_Weights_Criterion test')

    criterionJacobianTest(cri, input)
end 


math.randomseed(os.clock()) 

mytester:add(criterion_test)
jac = nn.Jacobian

mytester:run() 
