function [cellSimData] = addCalenderAgingAndAaggregateSamples(cellSimData,cellSimParams,soc_grid_bin_mean)
[soc_num,pow_num,sample_num] = size(cellSimData.capacity_loss_factor_samples);

slotIntervalInSeconds = cellSimParams.slotIntervalInSeconds;
slotIntervalInHrs = slotIntervalInSeconds/3600;
experimental_socs = [0;0.5;1];
experimental_rel_cap_at_900_calenderDays =[0.965;0.908;0.865];
experimental_calender_capacity_loss_factor_per_slot =  (1-experimental_rel_cap_at_900_calenderDays)/(900*24/slotIntervalInHrs);
interpolated_calender_capacity_loss_factor_per_slot = interp1(experimental_socs,experimental_calender_capacity_loss_factor_per_slot,soc_grid_bin_mean);


capacity_loss_factor_samples = cellSimData.capacity_loss_factor_samples;

capacity_loss_factor_incl_calender_samples = Inf([soc_num,pow_num,sample_num]);
for soc_idx = 1:soc_num
    capacity_loss_factor_incl_calender_samples(soc_idx,:,:) = capacity_loss_factor_samples(soc_idx,:,:);
    for pow_idx = 1:pow_num
        for sample_idx = 1: sample_num
            if(~isnan(capacity_loss_factor_incl_calender_samples(soc_idx,pow_idx,sample_idx)))
                capacity_loss_factor_incl_calender_samples(soc_idx,pow_idx,sample_idx) = max(capacity_loss_factor_incl_calender_samples(soc_idx,pow_idx,sample_idx) + ...
                    interpolated_calender_capacity_loss_factor_per_slot(soc_idx),interpolated_calender_capacity_loss_factor_per_slot(soc_idx));
            end
        end
    end    
end


cellSimData.capacity_loss_factor_samples = capacity_loss_factor_incl_calender_samples;
end

