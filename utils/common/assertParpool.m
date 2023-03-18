function [poolsize] = assertParpool(wait_time)
if(nargin==0)
    wait_time = 2;
end
[~,p_pool] = evalc('gcp(''nocreate'');');
if isempty(p_pool)
    user_choice = 'y';
    if(wait_time>0)
        fprintf('Parallel pool is missing. Hit ''ENTER'' within %d seconds to start a new parallel pool!',wait_time);
        user_choice = inputPrompter(wait_time);
    end
    if(user_choice == 'y')
        fprintf('Starting parallel pool...');
        [~,p_pool] = evalc('gcp;');
        poolsize = p_pool.NumWorkers;
        fprintf('Done. Connected to %d workers. \n',poolsize);
    else
        fprintf('Parallel pool not started.\n');
    end
else
    poolsize = p_pool.NumWorkers;
    fprintf('Parallel pool is running. Connected to %d workers. \n',poolsize);
end
end