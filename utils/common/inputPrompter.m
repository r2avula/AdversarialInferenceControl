function [user_choice] = inputPrompter(wait_time)
if(nargin<1 || isempty(wait_time))
    wait_time = inf;
end
input_string = '';
timeout_occured = false;
if(isinf(wait_time))
    input_string = input(' ','s');
elseif(wait_time>0)
    t = timer;
    t.ExecutionMode = 'singleShot';
    t.StartDelay = wait_time;
    t.TimerFcn = @terminatePrompter;
    start(t)
    input_string = input(' ','s');
    if (timeout_occured)
        input_string = 'n';
    end
    stop(t);
    delete(t);
end

if isempty(input_string)
    user_choice = 'y';
else
    input_string = lower(input_string);
    user_choice = input_string(1);
end

    function terminatePrompter(~,~)
        rob = java.awt.Robot;
        rob.keyPress(java.awt.event.KeyEvent.VK_ENTER)
        rob.keyRelease(java.awt.event.KeyEvent.VK_ENTER)
        timeout_occured = true;
    end
end

