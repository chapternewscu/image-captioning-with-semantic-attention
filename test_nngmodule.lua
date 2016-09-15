require 'totem'
require 'nngraph'
local LSTM_Armour = require 'misc_saver.LSTM_Armour'

local test = {}
local tester = totem.Tester()

local function checkGradients(...)
   totem.nn.checkGradients(tester, ...)
end

-- passed 
function test.test_example2()
      local x1=nn.Linear(20,20)()
      local x2=nn.Linear(10,10)()
      local m0=nn.Linear(20,1)(nn.Tanh()(x1))
      local m1=nn.Linear(10,1)(nn.Tanh()(x2))
      local madd=nn.CAddTable()({m0,m1})
      local m2=nn.Sigmoid()(madd)
      local m3=nn.Tanh()(madd)
      local gmod = nn.gModule({x1,x2},{m2,m3})

      local x = torch.rand(20)
      local y = torch.rand(10)
      checkGradients(gmod, {x, y})
end

-- passed 
function test.test_example3()
      local x1=nn.Identity()()
      local x2=nn.Identity()()
      
      local model = LSTM_Armour.makeWeightedSumUnit()({x1, x2}) 

      local gmod = nn.gModule({x1,x2},{model})

      local x = torch.rand(10, 8)
      local y = torch.rand(10, 8, 10)

      checkGradients(gmod, {y, x})
end

--passed 
function test.test_example4()
      local x =nn.Identity()()  -- glimpse 
      -- local x2=nn.Identity()()  -- As 
      local prev_c=nn.Identity()() 
      local prev_h=nn.Identity()() 

      local model = LSTM_Armour.lstm(8, 8, 1)({x, prev_c, prev_h}) 

      local gmod = nn.gModule({x, prev_c, prev_h},{model})

      local x = torch.rand(10, 8)
      -- local y = torch.rand(10, 2, 8) 
      local prev_c = torch.rand(10, 8) 
      local prev_h = torch.rand(10, 8)

      checkGradients(gmod, {x, prev_c, prev_h})
end 

-- passed 
-- testing Make Output Attention
function test.test_example5()
      local As =nn.Identity()()  -- glimpse 
      local h=nn.Identity()() 

      local model = LSTM_Armour.Make_Output_Attention_Bilinear_Unit(8, 8, 12)({h, As}) 

      local gmod = nn.gModule({h, As},{model})

      local h = torch.rand(10, 8)
      local As = torch.rand(10, 2, 8) 

      checkGradients(gmod, {h, As})
end

-- passed 
-- testing Make Input Attention
function test.test_example6() 

      local As =nn.Identity()()  -- glimpse 
      local prev_word_embed=nn.Identity()() 

      local model = LSTM_Armour.Make_Input_Attention_Bilinear_Unit(8, 8, 6)({prev_word_embed, As}) 

      local gmod = nn.gModule({prev_word_embed, As},{model})

      local prev_word_embed = torch.rand(10, 8)
      local As = torch.rand(10, 10, 8) 

      checkGradients(gmod, {prev_word_embed, As})
end



-- passed 
-- testing Make Input Attention
function test.test_example7() 

      local As =nn.Identity()()  
      local prev_word_embed=nn.Identity()() 

      local model = nn.BilinearD3(8, 8, 10)({prev_word_embed, As})
      local gmod = nn.gModule({prev_word_embed, As},{model})

      local prev_word_embed = torch.rand(10, 8)
      local As = torch.rand(10, 10, 8) 

      checkGradients(gmod, {prev_word_embed, As})
end

 tester:add(test):run()
