--just doing printing stuff 
peek = {} 

function peek.peek_here(...) 
    local args = {...} 
    -- command option:
    -- c: continue and just return and do nothing 
    -- p: print stuff 
    print('---------------------------------------\n')
    print('Peeking Starting .... \n') 
    print('peek content here\n')
    print('type command here to cotinue: c(continue), p(print content)\n')
    print('---------------------------------------\n')
    print('type a command:')

    while true do 
        local command = io.read() 
        if command == 'c' then 
            return 
        elseif command == 'p' then  
            for k, v in pairs(args) do 
                -- may log this(tensor) to a file
                print(k)
                print('\n')
                print(v) 
            end

            break 
        else -- command == nil  
            -- do nothing 
        end 
    end 
end 

return peek 

