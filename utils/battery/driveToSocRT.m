function [comsolParams,desired_state_reached,attempts,cell_reset] = driveToSocRT(comsolParams,des_SOC_low,des_SOC_high)
model = comsolParams.model;
cell_pow_set = comsolParams.cell_pow_set;
pow_num = length(cell_pow_set);

cur_soc = comsolParams.soc; 

des_SOC = (des_SOC_low + des_SOC_high)/2;
model.param.set('des_SOC',des_SOC);

cell_reset = 0;
charging_set = 0;
discharging_set = 0;
attempts = 0;
attempts_max = comsolParams.driveToSOC_attempts_max;
cycles = 0;
timePeriodScale = comsolParams.driveToSOC_timePeriodScaleFactor;
timePeriod = (timePeriodScale^cycles)*comsolParams.slotIntervalInSeconds*comsolParams.driveToSOC_timeAccelerationFactor;
model.param.set('period',timePeriod);
model.param.set('t_factor',comsolParams.driveToSOC_timeAccelerationFactor);

% fprintf(['Driving SOC from ',num2str(cur_soc,'%3.4f'),' to [',num2str(des_SOC_low,'%3.4f'),',',num2str(des_SOC_high,'%3.4f'),']; current SOC : ']);
% msg = char(sprintf('%3.4f', cur_soc));
% fprintf(msg);
% reverseStr = repmat(sprintf('\b'), 1, length(msg));
% time_stamp_start = tic;

desired_state_reached = 0;
while(~desired_state_reached)
%     msg = char(sprintf('%3.4f', cur_soc));
%     fprintf([reverseStr, msg]);
%     reverseStr = repmat(sprintf('\b'), 1, length(msg));

    if(cur_soc<des_SOC_low)
        ref_pow_idx = (randi(pow_num));
        while(cell_pow_set(ref_pow_idx)<=0)
            ref_pow_idx = (randi(pow_num));
        end
        if(~charging_set)
            model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', true, 2); % charging
            model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', false, 3); % discharging
            charging_set = 1;
        end
        if(discharging_set)
            discharging_set = 0;
            cycles = cycles + 1;
            timePeriod = (timePeriodScale^cycles)*comsolParams.slotIntervalInSeconds*comsolParams.driveToSOC_timeAccelerationFactor;
            model.param.set('period',timePeriod);
        end
    else
        ref_pow_idx = (randi(pow_num));
        while(cell_pow_set(ref_pow_idx)>=0)
            ref_pow_idx = (randi(pow_num));
        end
        if(~discharging_set)
            model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', false, 2); % charging
            model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', true, 3); % discharging
            discharging_set = 1;
        end        
        if(charging_set)
            charging_set = 0;
            cycles = cycles + 1;
            timePeriod = (timePeriodScale^cycles)*comsolParams.slotIntervalInSeconds*comsolParams.driveToSOC_timeAccelerationFactor;
            model.param.set('period',timePeriod);
        end
    end
    
    [out,cell_reset] = simulateBatteryCell(comsolParams,ref_pow_idx,0);
    if(cell_reset)
        attempts = 1;
    end
    comsolParams = out.comsolParams;
    cur_soc = comsolParams.soc; 
    
    if(cur_soc >= des_SOC_low && cur_soc <= des_SOC_high)
        desired_state_reached = 1;
        comsolParams.soc = cur_soc;
    elseif(cell_pow_set(ref_pow_idx)~=0)
        attempts = attempts + 1;
        if(attempts>attempts_max)
            comsolParams.soc = cur_soc;
            break;
        end
    end
end
% msg = char(sprintf('%3.4f', cur_soc));
% fprintf([reverseStr, msg]);
% elapsed_time = toc(time_stamp_start);
% if(elapsed_time>86400)
%     end_printString = sprintf('; Elapsed time = %3.1f days   ',elapsed_time/86400);
% elseif(elapsed_time>3600)
%     end_printString = sprintf(';\t Elapsed time = %3.1f hours ',elapsed_time/3600);
% elseif(elapsed_time>60)
%     end_printString = sprintf(';\t Elapsed time = %3.1f minutes',elapsed_time/60);
% else
%     end_printString = sprintf(';\t Elapsed time = %3.1f seconds',elapsed_time);
% end
% fprintf([end_printString,'\n']);


if(charging_set)
    model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', false, 2); % charging
end
if(discharging_set)
    model.sol('sol2').feature('t1').feature('st1').setIndex('stopcondActive', false, 3); % discharging
end
model.param.set('des_SOC',1);
model.param.set('period',comsolParams.slotIntervalInSeconds);
model.param.set('t_factor',1);
if ~(cur_soc >= des_SOC_low && cur_soc <= des_SOC_high)
    warning(strcat('want: [',num2str(des_SOC_low),num2str(des_SOC_high),'] acheived:',num2str(cur_soc)));
end
end