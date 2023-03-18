function [comsolParams] = driveToRelCap(comsolParams,des_cellRelativeCapacity)
cell_pow_set = comsolParams.cell_pow_set;
pow_num = length(cell_pow_set);

model = comsolParams.model;
cur_cellRelativeCapacity = comsolParams.relativeCapacity;

sim_start = false;
if(cur_cellRelativeCapacity <= des_cellRelativeCapacity)
    desired_state_reached = 1;
else
    sim_start = true;
    desired_state_reached = 0;
    model.param.set('period',comsolParams.slotIntervalInSeconds*comsolParams.driveToSOH_timeAccelerationFactor);
    model.param.set('t_factor',comsolParams.driveToSOH_timeAccelerationFactor);
    
    soc_k = comsolParams.soc;
    soc_grid_boundaries = comsolParams.soc_grid_boundaries;
    soc_num = length(soc_grid_boundaries)-1;
    soc_k_bin_idx = find(soc_grid_boundaries(2:end-1)>soc_k,1);
    if(isempty(soc_k_bin_idx))
        soc_k_bin_idx = soc_num;
    end
end

% if(sim_start)
%     fprintf(['Driving SOH from ',num2str(cur_cellRelativeCapacity),' to ',num2str(des_cellRelativeCapacity),'; current SOH : ']);
%     msg = char(sprintf('%3.2f', cur_cellRelativeCapacity));
%     fprintf(msg);
%     reverseStr = repmat(sprintf('\b'), 1, length(msg));
%     time_stamp_start = tic;
% end
while(~desired_state_reached)
%     msg = char(sprintf('%3.2f', cur_cellRelativeCapacity));
%     fprintf([reverseStr, msg]);
%     reverseStr = repmat(sprintf('\b'), 1, length(msg));

    if(soc_k_bin_idx==1)
        ref_pow_idx = (randi(pow_num));
        while(cell_pow_set(ref_pow_idx)<=0)
            ref_pow_idx = (randi(pow_num));
        end
    elseif(soc_k_bin_idx==soc_num)
        ref_pow_idx = (randi(pow_num));
        while(cell_pow_set(ref_pow_idx)>=0)
            ref_pow_idx = (randi(pow_num));
        end
    else
        ref_pow_idx = (randi(pow_num));
    end

    out = simulateBatteryCell(comsolParams,ref_pow_idx,1);
    comsolParams = out.comsolParams;
    cur_cellRelativeCapacity = comsolParams.relativeCapacity;
    
    if(cur_cellRelativeCapacity <= des_cellRelativeCapacity)
        desired_state_reached = 1;
    else
        desired_state_reached = 0;
        soc_k = comsolParams.soc;
        soc_grid_boundaries = comsolParams.soc_grid_boundaries;
        soc_num = length(soc_grid_boundaries)-1;
        soc_k_bin_idx = find(soc_grid_boundaries(2:end-1)>soc_k,1);
        if(isempty(soc_k_bin_idx))
            soc_k_bin_idx = soc_num;
        end
    end
end
if(sim_start)
    %     elapsed_time = toc(time_stamp_start);
    %     if(elapsed_time>86400)
    %         end_printString = sprintf('; Elapsed time = %3.1f days   ',elapsed_time/86400);
    %     elseif(elapsed_time>3600)
    %         end_printString = sprintf(';\t Elapsed time = %3.1f hours ',elapsed_time/3600);
    %     elseif(elapsed_time>60)
    %         end_printString = sprintf(';\t Elapsed time = %3.1f minutes',elapsed_time/60);
    %     else
    %         end_printString = sprintf(';\t Elapsed time = %3.1f seconds',elapsed_time);
    %     end
    %     fprintf([end_printString,'\n']);

    model.param.set('period',comsolParams.slotIntervalInSeconds);
    model.param.set('t_factor',1);
end

cell_pow_set = comsolParams.cell_pow_set;
zero_pow_idx = find(cell_pow_set == 0,1);
out = simulateBatteryCell(comsolParams,zero_pow_idx,1); % keep idle for one time slot
comsolParams = out.comsolParams;
end