require 'nn'
require 'misc_saver.BilinearD3_version2'

local mytester = torch.Tester() 

local jac 
local sjac 

local precision = 1e-5 
local expprecison = 1e-4 

local nntest = torch.TestSuite() 

function nntest.BilinearD3() 
    -- set up data 
    local N = 10 
    local D1 = 5 
    local D2 = 4

    -- output size is 3 
    local K = 16

    local input = {torch.randn(N, D1), torch.randn(N, K, D2)} 

    -- test forward 
    local module = nn.BilinearD3(D1, D2, K)
    local expected = torch.zeros(N, K) 
    
    for k = 1, K do
        -- module.bi_core: 1 * D1 * D2 
        local temp = torch.mm(module.weight, input[2]:select(2, k):t()) 
        temp:cmul(input[1]:t()) 
        temp = temp:sum(1)  -- sum along first dimension
        temp:add(module.bias:expand(10))
        expected[{{}, k}] = temp:view(-1) 
    end 
     
    local output = module:forward(input) 

    mytester:assertTensorEq(expected, output, 0.000001, 'BilinearD3 forward error')

    --for testing grads 
    local input2 = torch.randn(2, N, D1) -- 2 * 10 * 5 
    
    local module2 = nn.Sequential() 
    module2:add(nn.SplitTable(1))  -- {N * D1, N * D1}

    module2:add(nn.ParallelTable():add(nn.Linear(D1, D1)):add(nn.Replicate(K, 2, 3)))
    
    module2:add(nn.BilinearD3(D1, D1, K)) 
    module2:add(nn.Linear(K, 1)) 

    local err = jac.testJacobian(module2, input2) 
    mytester:assertlt(err, precision, 'error on state ')

    local err = jac.testJacobianParameters(module2, input2, module2:get(3).weight, module2:get(3).gradWeight)
    mytester:assertlt(err, precision, 'error on weight ')

    local err = jac.testJacobianParameters(module2, input2, module2:get(3).bias, module2:get(3).gradBias)
    mytester:assertlt(err, precision, 'error on bias ')

    print('test BilinearD3 done')
end

mytester:add(nntest)
jac = nn.Jacobian
sjac = nn.SparseJacobian
mytester:run()
