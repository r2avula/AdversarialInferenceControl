function [cellSimData] = runComsolSimulation(comsolParams,comsol_model,progressDataQueue,incPercent)
import com.comsol.model.util.*;

soc_grid_boundaries = comsolParams.soc_grid_boundaries;
cell_pow_set = comsolParams.cell_pow_set;

sample_num = comsolParams.sample_num;

soc_num = length(soc_grid_boundaries)-1;
pow_num = length(cell_pow_set);

slotIntervalInSeconds = comsolParams.slotIntervalInSeconds;
slotIntervalInHours = slotIntervalInSeconds/3600;

batterySelfDischargeRatePerMonth = 0; % factor in [0,1]
if(batterySelfDischargeRatePerMonth >0)
    tau = 30*24/-log(1-batterySelfDischargeRatePerMonth); %h
    gamma = 1 - exp(-slotIntervalInHours/tau);
    gamma_tau = gamma*tau; % in h
else
    gamma = 0;
    gamma_tau = slotIntervalInHours; % in h
end

comsolParams.gamma = gamma;
comsolParams.gamma_tau = gamma_tau;

power_loss_param_samples = nan([soc_num,pow_num,sample_num]);
capacity_loss_factor_samples = nan([soc_num,pow_num,sample_num]);
cur_sample_idxs = ones([soc_num,pow_num]);

total_sim_steps = soc_num*pow_num*sample_num;
validDataCount = 0;

comsolParams.cellsIterated = 0;
comsolParams.model = comsol_model;

[comsolParams] = initializeNewCell(comsolParams);

des_soc_bin_idx = findUnfilledSoC(power_loss_param_samples);
desired_state_reached = 0;
attempts_to_driveToSOC = 0;
while(~desired_state_reached)
    [comsolParams,desired_state_reached,attempts,cell_reset] = driveToSoc(comsolParams,des_soc_bin_idx);
    if(~desired_state_reached)
        des_soc_bin_idx = findUnfilledSoC(power_loss_param_samples);        
    end
    if(cell_reset)
        attempts_to_driveToSOC = attempts;
    else
        attempts_to_driveToSOC = attempts_to_driveToSOC + attempts;
    end
end
soc_k_bin_idx = des_soc_bin_idx;

total_simsteps_done = 0;
incPercent_t = incPercent/total_sim_steps;
isSimDone = false;
while(~isSimDone)
    ref_pow_idx = findUnfilledPow(power_loss_param_samples,soc_k_bin_idx);
    if(isempty(ref_pow_idx))
        des_soc_bin_idx = findUnfilledSoC(power_loss_param_samples);
        isSimDone = isempty(des_soc_bin_idx);   
        if(~isSimDone)
            desired_state_reached = 0;
            attempts_to_driveToSOC = 0;
            while(~desired_state_reached)
                [comsolParams,desired_state_reached,attempts,cell_reset] = driveToSoc(comsolParams,des_soc_bin_idx);
                if(~desired_state_reached)
                    des_soc_bin_idx = findUnfilledSoC(power_loss_param_samples);
                end
                if(cell_reset)
                    attempts_to_driveToSOC = attempts;
                else
                    attempts_to_driveToSOC = attempts_to_driveToSOC + attempts;
                end
            end
            soc_k_bin_idx = des_soc_bin_idx;
            ref_pow_idx = findUnfilledPow(power_loss_param_samples,soc_k_bin_idx);
        end
    end

    if(~isSimDone)
        sample_idx = cur_sample_idxs(soc_k_bin_idx,ref_pow_idx);
        sim_out = simulateBatteryCell(comsolParams,ref_pow_idx,0);
        comsolParams = sim_out.comsolParams;

        if(sim_out.sim_executed)
            capacity_loss_factor_samples(soc_k_bin_idx,ref_pow_idx,sample_idx) = sim_out.capacity_loss_factor; %Ah
            power_loss_param_samples(soc_k_bin_idx,ref_pow_idx,sample_idx) = sim_out.power_loss_param; %W

            soc_kp1_bin_idx = find(soc_grid_boundaries(2:end-1)>comsolParams.soc,1);
            if(isempty(soc_kp1_bin_idx))
                soc_kp1_bin_idx = soc_num;
            end            

            validDataCount = validDataCount + 1;
        else
            capacity_loss_factor_samples(soc_k_bin_idx,ref_pow_idx,sample_idx) = nan;
            power_loss_param_samples(soc_k_bin_idx,ref_pow_idx,sample_idx) = 0;
            soc_kp1_bin_idx = soc_k_bin_idx;
        end
        
        cur_sample_idxs(soc_k_bin_idx,ref_pow_idx) = sample_idx + 1;
        soc_k_bin_idx = soc_kp1_bin_idx;
        send(progressDataQueue, incPercent_t);
        total_simsteps_done = total_simsteps_done + 1;
    end
end
send(progressDataQueue, incPercent_t*(total_sim_steps - total_simsteps_done));

cellSimData = struct;
cellSimData.capacity_loss_factor_samples = capacity_loss_factor_samples;
cellSimData.power_loss_param_samples = power_loss_param_samples;
cellSimData.cellsIterated = comsolParams.cellsIterated;
cellSimData.validDataCount = validDataCount;
end