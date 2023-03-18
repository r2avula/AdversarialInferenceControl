function [sim_out,cell_reset] = simulateBatteryCell(comsolParams,ref_pow_idx,drivingToDesiredCapacity)
cell_reset = 0;
if(comsolParams.initialRelCap-comsolParams.relativeCapacity > comsolParams.allowedRelativeCapacityChange)
    [comsolParams] = initializeNewCell(comsolParams);
    cell_reset = 1;
end


cur_voltage = comsolParams.terminal_voltage;
slotIntervalInSeconds = comsolParams.slotIntervalInSeconds;
model = comsolParams.model;
cell_pow_set = comsolParams.cell_pow_set;
ref_pow = cell_pow_set(ref_pow_idx);

soc_limit_buffer = 0.05;
volt_limit_buffer = 0.2;

model.param.set('P_ref',ref_pow);

cur_soc = comsolParams.soc;
run_sim = true;
if(ref_pow>0 && cur_soc >= comsolParams.SOC_high) || (ref_pow<0 && cur_soc <= comsolParams.SOC_low)
    run_sim = false;
end

SOC_low_changed = false;
E_min_changed = false;
SOC_high_changed = false;
E_max_changed = false;
something_changed = false;
sim_executed = false;

capacity_loss_factor = 0;
power_loss_param = 0;
simTimeInSeconds = 0;

% error_message = 'Simulation failed to start!';
if(run_sim)  
%     fprintf('Running COMSOL simulation ...');
    try
        model.study('std2').run;% Apply requested power
        sim_executed = true;
    catch ME
        error_message = 'Stop condition fulfilled for initial values';
        if(contains(ME.message,error_message))
            if(ref_pow>0)
                if(cur_soc <= comsolParams.SOC_low+soc_limit_buffer)
                    model.param.set('SOC_low',max(cur_soc-soc_limit_buffer,0));
                    SOC_low_changed = true;
                    something_changed = true;
                end
                if(cur_voltage <= comsolParams.cell_voltage_low+volt_limit_buffer)
                    model.param.set('E_min',cur_voltage-volt_limit_buffer);
                    E_min_changed = true;
                    something_changed = true;
                end
            end
            if(ref_pow<=0)
                if(cur_soc >= comsolParams.SOC_high-soc_limit_buffer)
                    model.param.set('SOC_high',min(cur_soc+soc_limit_buffer,1));
                    SOC_high_changed = true;
                    something_changed = true;
                end
                if(cur_voltage >= comsolParams.cell_voltage_high-volt_limit_buffer)
                    model.param.set('E_max',cur_voltage+volt_limit_buffer);
                    E_max_changed = true;
                    something_changed = true;
                end
            end
            if(something_changed)
                try
                    model.study('std2').run;% Apply requested power
                    sim_executed = true;
                catch ME2
                    if(~contains(ME2.message,error_message))
                        error_message = strcat(error_message,' + ',ME2.message);
                        ME2.message = error_message;
                        rethrow(ME2);
                    end
                end
            end            
        else
            rethrow(ME);
        end
    end
end

if(sim_executed)
%     fprintf('SUCCESS');
    simTimeInSeconds = floor(mphglobal(model,'t','dataset','dset2','solnum','end'));
    simTimeRatio = (simTimeInSeconds)/slotIntervalInSeconds;

    terminal_voltage = mphglobal(model,'Ecell','dataset','dset2','solnum','end');
    internal_resistance = mphglobal(model,'R_cell_int','dataset','dset2','solnum','end');

    if(drivingToDesiredCapacity)
        comsolParams.relativeCapacity = mphglobal(model,'Rel_cap','dataset','dset2','solnum','end')*100;
    else
        old_rel_cap = comsolParams.relativeCapacity;
        comsolParams.relativeCapacity = mphglobal(model,'Rel_cap','dataset','dset2','solnum','end')*100;
        capacity_loss_factor = (old_rel_cap-comsolParams.relativeCapacity)/100/simTimeRatio; 

        do_discrete_volt_avg = false;
        do_discrete_res_avg = false;
        if(simTimeInSeconds>0)
            try
                mean_terminal_voltage = mphglobal(model,strcat('timeint(0,',num2str(simTimeInSeconds),',Ecell)'),'dataset','dset2','solnum','end')/simTimeInSeconds; %in V
            catch ME
                if(contains(ME.message,'timeint'))
                    do_discrete_volt_avg = true;
                else
                    rethrow(ME);
                end
            end

            try
                mean_internal_resistance = mphglobal(model,strcat('timeint(0,',num2str(simTimeInSeconds),',R_cell_int)'),'dataset','dset2','solnum','end')/simTimeInSeconds; %in ohm
                if(isinf(mean_internal_resistance))
                    do_discrete_res_avg = true;
                end
            catch ME
                if(contains(ME.message,'timeint'))
                    do_discrete_res_avg = true;
                else
                    rethrow(ME);
                end
            end
        else
            do_discrete_volt_avg = true;
            do_discrete_res_avg = true;
        end

        if(do_discrete_volt_avg)
            prev_terminal_voltage = comsolParams.terminal_voltage;
            mean_terminal_voltage = (prev_terminal_voltage + terminal_voltage)/2;
        end

        if(do_discrete_res_avg)
            prev_internal_resistance = comsolParams.internal_resistance;
            if(isinf(internal_resistance))
                mean_internal_resistance = prev_internal_resistance;
                internal_resistance = prev_internal_resistance;
            elseif(isinf(prev_internal_resistance))
                mean_internal_resistance = internal_resistance;
            else
                mean_internal_resistance = (prev_internal_resistance + internal_resistance)/2;
            end
        end

        power_loss_param = mean_terminal_voltage^2/mean_internal_resistance;
    end

    comsolParams.soc = mphglobal(model,'SOCcell','dataset','dset2','solnum','end');
    comsolParams.terminal_voltage = terminal_voltage;
    comsolParams.internal_resistance = internal_resistance;
    comsolParams.model = model;
else
    %     fprintf(strcat('FAIL: ',error_message));
end

% fprintf('\n');
if(SOC_low_changed)
    model.param.set('SOC_low',comsolParams.SOC_low);
end
if(E_min_changed)
    model.param.set('E_min',comsolParams.cell_voltage_low);
end
if(SOC_high_changed)
    model.param.set('SOC_high',comsolParams.SOC_high);
end
if(E_max_changed)
    model.param.set('E_max',comsolParams.cell_voltage_high);
end

sim_out = struct;
sim_out.comsolParams = comsolParams;
sim_out.capacity_loss_factor = capacity_loss_factor;
sim_out.power_loss_param = power_loss_param;
sim_out.sim_executed = sim_executed;
sim_out.simTimeInSeconds = simTimeInSeconds;
end