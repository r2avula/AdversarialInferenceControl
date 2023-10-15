function [params, essParams, comsolParams] = getEssParams(config, params, get_comsol_params)
if nargin == 2
    get_comsol_params = false;
end
z_num = config.z_num;
d_num = config.d_num;
l_num_paramapprox = config.l_num_paramapprox;
soc_grid_boundaries = config.soc_grid_boundaries;
z_k_idxs_given_l_k_idx_comsol= config.z_k_idxs_given_l_k_idx_comsol;
gamma = config.gamma;
gamma_tau = config.gamma_tau;
legsInParallel = config.legsInParallel;
cellsInSeries = config.cellsInSeries;
cell_1C_capacityInAh = config.cell_1C_capacityInAh;
cell_nominalVoltage = config.cell_nominalVoltage;
l_grid_boundaries_comsol = config.l_grid_boundaries_comsol;
cell_pow_set = config.cell_pow_set;
slotIntervalInSeconds = config.slotIntervalInSeconds;
l_grid_bin_mean_comsol = config.l_grid_bin_mean_comsol;
cell_SOC_low = config.cell_SOC_low;
cell_SOC_high = config.cell_SOC_high;
slotIntervalInHours = config.slotIntervalInHours;
soc_grid_bin_mean = config.soc_grid_bin_mean;
bat_pow_set = config.bat_pow_set;
z_cap = config.z_cap;
paramsPrecision = config.paramsPrecision;
batteryNominalVoltage = config.batteryNominalVoltage;

cache_folder_path = config.cache_folder_path;
essProcessedDataFileNamePrefix = strcat(cache_folder_path,'essProcessedData_');
if (isfield(config,'emulator') && config.emulator == "comsol")
    path_to_battery_emulator_data_folder = strcat("resources",filesep,"battery_emulator_data", filesep);
    batteryRatedCapacityInAh = config.batteryRatedCapacityInAh;

    deglifePartitions_num =  config.deglifePartitions_num;
    deglifePartitions = linspace(1,0.8,deglifePartitions_num+1);
    allowedRelativeCapacityChange = (deglifePartitions(1)-deglifePartitions(2))*100;

    cellSimData_all = cell(deglifePartitions_num,1);
    do_simulations = false(deglifePartitions_num,1);

    P_Zp1gZD_map_all = zeros(z_num,z_num,d_num,deglifePartitions_num);

    mean_capacityLossInAh_map_all = nan(z_num,d_num,deglifePartitions_num);
    cell_power_loss_param_map_all = nan(z_num,d_num,deglifePartitions_num);

    alpha_est_map_all = nan(z_num,d_num,deglifePartitions_num);
    beta_est_map_all = nan(z_num,d_num,deglifePartitions_num);

    cellSimData_fileNamePrefix = strcat(path_to_battery_emulator_data_folder, "cellSimData_", num2str(batteryRatedCapacityInAh), "_");
    
    comsolParams = struct;
    comsolParams.cell_1C_capacityInAh = cell_1C_capacityInAh;
    comsolParams.cell_nominalVoltage = cell_nominalVoltage;
    comsolParams.soc_grid_boundaries = l_grid_boundaries_comsol;
    comsolParams.cell_pow_set = cell_pow_set;
    comsolParams.slotIntervalInSeconds = slotIntervalInSeconds;
    comsolParams.sample_num = config.sample_num;
    comsolParams.SOC_low = cell_SOC_low;
    comsolParams.SOC_high = cell_SOC_high;
    comsolParams.allowedRelativeCapacityChange = allowedRelativeCapacityChange;
    comsolParams.SOC_init = config.cell_SOC_init;
    comsolParams.cell_voltage_high = config.cell_voltage_high;
    comsolParams.cell_voltage_low = config.cell_voltage_low;
    comsolParams.driveToSOH_timeAccelerationFactor = config.driveToSOH_timeAccelerationFactor;
    comsolParams.driveToSOC_timeAccelerationFactor = config.driveToSOC_timeAccelerationFactor;
    comsolParams.driveToSOC_timePeriodScaleFactor = config.driveToSOC_timePeriodScaleFactor;
    comsolParams.driveToSOC_attempts_max = config.driveToSOC_attempts_max;
    for partition_idx = 1:deglifePartitions_num
        comsolParams.initialRelCap = deglifePartitions(partition_idx)*100;
        [filename,fileExists] = findFileName(comsolParams,cellSimData_fileNamePrefix,'comsolParams');
        if(fileExists)
            cached_data = load(filename,'cellSimData');
            cellSimData = cached_data.cellSimData;
            cellSimData_all{partition_idx} = addCalenderAgingAndAaggregateSamples(cellSimData,comsolParams,l_grid_bin_mean_comsol);
        else
            do_simulations(partition_idx) = true;
        end
    end

    if(any(do_simulations))
        error('battery simulation required!')
        path_to_comsol_app = config.path_to_comsol_app;  
        addpath(genpath(strcat(path_to_comsol_app,"mli")));
        checkComsolConnection(path_to_comsol_app);

        comsol_model_filename = strcat(pwd, filesep, "utils", filesep, "battery", filesep, "li_degradation_core.mph");
        [comsol_model] = getComsolModel(comsol_model_filename);

        [progressData, progressDataQueue] = ProgressData('\t\t\tRunning battery simulations : ');
        incPercent = 100/sum(do_simulations);
        parsave_cellSimData_fn = @(x,y,z)parsave_cellSimData(x,y,z);
        for partition_idx = find(do_simulations)'
            comsolParams.initialRelCap = deglifePartitions(partition_idx)*100;
            [filename,~] = findFileName(comsolParams,cellSimData_fileNamePrefix,'cellSimParams');
            cellSimData = runComsolSimulation(comsolParams,comsol_model,progressDataQueue, incPercent);
            parsave_cellSimData_fn(filename,cellSimData,comsolParams)
            cellSimData_all{partition_idx} = addCalenderAgingAndAaggregateSamples(cellSimData,comsolParams,l_grid_bin_mean_comsol);
        end
        terminate(progressData);
    end

    for partition_idx = 1:deglifePartitions_num
        essProcessedDataParams = comsolParams;
        essProcessedDataParams.initialRelCap = deglifePartitions(partition_idx)*100;
        essProcessedDataParams.soc_grid_boundaries = soc_grid_boundaries;
        essProcessedDataParams.gamma = gamma;
        essProcessedDataParams.gamma_tau = gamma_tau;

        [filename,fileExists] = findFileName(essProcessedDataParams,essProcessedDataFileNamePrefix,'essProcessedDataParams');
        if(fileExists)
            load(filename,'cell_power_loss_param_map','alpha_est_map','beta_est_map','P_Zp1gZD_map','mean_capacityLossInAh_map');
        else
            cellSimData = cellSimData_all{partition_idx};
            capacity_loss_factor_samples = cellSimData.capacity_loss_factor_samples;
            power_loss_param_samples = cellSimData.power_loss_param_samples;

            alpha_est_map = nan(z_num,d_num);
            beta_est_map = nan(z_num,d_num);
            cell_power_loss_param_map = nan(z_num,d_num);


            num_unif_samples = config.sample_num;
            scaled_soc_grid_boundaries = rescale(soc_grid_boundaries,0,1,'InputMin',cell_SOC_low,'InputMax',cell_SOC_high);

            [progressData, progressDataQueue] = ProgressData('\t\t\tEstimating ESS params :');
            incPercent = (1/d_num/l_num_paramapprox)*100;
            P_Zp1gZD_map = zeros(z_num,z_num,d_num);
            mean_capacityLossInAh_map = nan(z_num,d_num);
            soc_kp1_estimate_3c_fn_ref = @(soc_k,power_loss_param,gamma,gamma_tau,sigma_d,en_cap)soc_kp1_estimate_3c_fn(soc_k,power_loss_param,gamma,gamma_tau,sigma_d,en_cap);
            for d_k_idx = 1:d_num
                soc_grid_boundaries_t = soc_grid_boundaries;
                scaled_soc_grid_boundaries_t = scaled_soc_grid_boundaries;
                soc_kp1_estimate_3c_fn_ref_t = soc_kp1_estimate_3c_fn_ref;

                alpha_est_t = nan(z_num,1);
                beta_est_t = nan(z_num,1);
                cell_power_loss_param_map_t = nan(z_num,1);

                P_Zp1gZD_map_t = zeros(z_num,z_num,1);
                for l_k_idx_comsol = 1:l_num_paramapprox
                    capacity_loss_factor_vec = reshape(capacity_loss_factor_samples(l_k_idx_comsol,d_k_idx,:),1,[]);
                    power_loss_param_samples_vec = reshape(power_loss_param_samples(l_k_idx_comsol,d_k_idx,:),1,[])*legsInParallel*cellsInSeries;

                    validSampleIdxs = (~isnan(capacity_loss_factor_vec) & ~isinf(capacity_loss_factor_vec));

                    capacity_loss_factor_vec = capacity_loss_factor_vec(validSampleIdxs);
                    power_loss_param_samples_vec = power_loss_param_samples_vec(validSampleIdxs);
                    numValidSamples = length(capacity_loss_factor_vec);
                    if(numValidSamples>0)
                        z_k_idxs = z_k_idxs_given_l_k_idx_comsol{l_k_idx_comsol};
                        for z_k_idx = z_k_idxs
                            cell_power_loss_param_map_t(z_k_idx) = mean(power_loss_param_samples_vec)/(legsInParallel*cellsInSeries);
                            soc_a = soc_grid_boundaries_t(z_k_idx);
                            soc_b = soc_grid_boundaries_t(z_k_idx+1);

                            soc_k_samples = rescale(rand(num_unif_samples,1),soc_a,soc_b,'InputMin',0,'InputMax',1);
                            soc_kp1_estimates_3c = nan(numValidSamples*num_unif_samples,1);
                            samples_count = 0;
                            for k_sample_idx = 1:num_unif_samples
                                soc_k = soc_k_samples(k_sample_idx);
                                for data_sample_idx = 1:numValidSamples
                                    [soc_kp1_estimate_3c,action_performed] = soc_kp1_estimate_3c_fn_ref_t(soc_k,power_loss_param_samples_vec(data_sample_idx),gamma,gamma_tau,bat_pow_set(d_k_idx),z_cap);
                                    if(action_performed)
                                        samples_count = samples_count + 1;
                                        soc_kp1_estimates_3c(samples_count) = soc_kp1_estimate_3c;
                                    end
                                end
                            end
                            soc_kp1_estimates_3c(samples_count+1:end) = [];

                            if(~isempty(soc_kp1_estimates_3c))
                                P_Zp1 = zeros(z_num,1);
                                if(all(soc_kp1_estimates_3c==cell_SOC_low))
                                    alpha_est = 0;
                                    beta_est = 1;
                                    P_Zp1(1) = 1;
                                elseif(all(soc_kp1_estimates_3c==cell_SOC_high))
                                    alpha_est = 1;
                                    beta_est = 0;
                                    P_Zp1(z_num) = 1;
                                else
                                    scaled_soc_kp1_estimates_3c = rescale(soc_kp1_estimates_3c,0,1,'InputMin',cell_SOC_low,'InputMax',cell_SOC_high);
                                    phat = betafit(scaled_soc_kp1_estimates_3c);
                                    alpha_est = phat(1);
                                    beta_est = phat(2);
                                    for z_kp1_idx = 1:z_num
                                        P_Zp1(z_kp1_idx) = betacdf(scaled_soc_grid_boundaries_t(z_kp1_idx+1),alpha_est,beta_est) - betacdf(scaled_soc_grid_boundaries_t(z_kp1_idx),alpha_est,beta_est);
                                    end
                                end

                                P_Zp1 = P_Zp1/sum(P_Zp1);
                                P_Zp1 = roundOffInSimplex(P_Zp1,paramsPrecision);
                                P_Zp1gZD_map_t(:,z_k_idx) = P_Zp1;
                                alpha_est_t(z_k_idx) = alpha_est;
                                beta_est_t(z_k_idx) = beta_est;
                            end

                            mean_capacityLossInAh_map(z_k_idx,d_k_idx) = mean(capacity_loss_factor_vec)*cell_1C_capacityInAh*legsInParallel;
                        end
                    end

                    send(progressDataQueue, incPercent);
                end

                alpha_est_map(:,d_k_idx) = alpha_est_t;
                beta_est_map(:,d_k_idx) = beta_est_t;
                P_Zp1gZD_map(:,:,d_k_idx) = P_Zp1gZD_map_t;
                cell_power_loss_param_map(:,d_k_idx) = cell_power_loss_param_map_t;

                if d_k_idx == -params.d_offset
                    P_Zp1gZD_map_t = zeros(z_num,z_num,1);
                    for z_k_idx = 1:z_num
                        P_Zp1gZD_map_t(z_k_idx, z_k_idx, 1) = 1;
                    end
                    P_Zp1gZD_map(:,:,d_k_idx) = P_Zp1gZD_map_t;
                end
            end
            progressData.terminate();
            save(filename,'cell_power_loss_param_map','alpha_est_map','beta_est_map','P_Zp1gZD_map','mean_capacityLossInAh_map','essProcessedDataParams');
        end

        P_Zp1gZD_map_all(:,:,:,partition_idx) = P_Zp1gZD_map;
        alpha_est_map_all(:,:,partition_idx) = alpha_est_map;
        beta_est_map_all(:,:,partition_idx) = beta_est_map;
        cell_power_loss_param_map_all(:,:,partition_idx) = cell_power_loss_param_map;
        mean_capacityLossInAh_map_all(:,:,partition_idx) = mean_capacityLossInAh_map;
    end

    P_Zp1gZD_map = P_Zp1gZD_map_all(:,:,:,1);

    cell_mean_power_loss_param = cell_power_loss_param_map_all(:);
    cell_mean_power_loss_param = cell_mean_power_loss_param(~isnan(cell_mean_power_loss_param));
    cell_mean_power_loss_param = roundOff(mean(cell_mean_power_loss_param),paramsPrecision);

    essParams = rmfield(comsolParams,{'soc_grid_boundaries','sample_num','SOC_init','driveToSOH_timeAccelerationFactor',...
        'driveToSOC_timeAccelerationFactor','driveToSOC_timePeriodScaleFactor','driveToSOC_attempts_max','initialRelCap'});

    essDegradationCost_mean_map = 5*(config.capacityCostPerkWh)*batteryNominalVoltage/1000*mean_capacityLossInAh_map_all(:,:,1);
    essDegradationCost_mean_map = roundOff(max(essDegradationCost_mean_map,0),paramsPrecision);

    essParams.essDegradationCost_mean_map = essDegradationCost_mean_map;
    essParams.cell_mean_power_loss_param = cell_mean_power_loss_param;
    essParams.gamma = gamma;
    essParams.gamma_tau = gamma_tau;
else
    cell_mean_power_loss_param = config.cell_mean_power_loss_param;
    beta_max_var = config.beta_max_var; %% should be in (0,0.25)
    min_alpha_plus_beta = config.min_alpha_plus_beta;

    essProcessedDataParams = struct;
    essProcessedDataParams.soc_grid_boundaries = soc_grid_boundaries;
    essProcessedDataParams.gamma = gamma;
    essProcessedDataParams.gamma_tau = gamma_tau;
    essProcessedDataParams.cell_mean_power_loss_param = cell_mean_power_loss_param;
    essProcessedDataParams.beta_max_var = beta_max_var;
    essProcessedDataParams.min_alpha_plus_beta = min_alpha_plus_beta;
    essProcessedDataParams.cell_1C_capacityInAh = cell_1C_capacityInAh;
    essProcessedDataParams.cell_nominalVoltage = cell_nominalVoltage;
    essProcessedDataParams.cell_pow_set = cell_pow_set;
    essProcessedDataParams.slotIntervalInSeconds = slotIntervalInSeconds;
    essProcessedDataParams.SOC_low = cell_SOC_low;
    essProcessedDataParams.SOC_high = cell_SOC_high;
    essProcessedDataParams.ess_extreme_prob_range_num = config.ess_extreme_prob_range_num;
    ess_extreme_prob_fact = config.ess_extreme_prob_fact;
    if essProcessedDataParams.ess_extreme_prob_range_num==1
        ess_extreme_prob_fact = 0;
    end
    essProcessedDataParams.ess_extreme_prob_fact = ess_extreme_prob_fact;

    [filename,fileExists] = findFileName(essProcessedDataParams,essProcessedDataFileNamePrefix,'essProcessedDataParams');
    if(fileExists)
        load(filename,'alpha_est_map','beta_est_map','P_Zp1gZD_map'); %#ok<NASGU> 
    else
        scaled_soc_grid_boundaries = rescale(soc_grid_boundaries,0,1,'InputMin',cell_SOC_low,'InputMax',cell_SOC_high);
        extreme_prob_fact = essProcessedDataParams.ess_extreme_prob_fact;
        extreme_prob_range_num = essProcessedDataParams.ess_extreme_prob_range_num;
        [progressData, progressDataQueue] = ProgressData('\t\t\tEstimating ESS params : ');
        incPercent = (1/d_num/z_num)*100;
        alpha_est_map = nan(z_num,d_num);
        beta_est_map = nan(z_num,d_num);
        P_Zp1gZD_map = zeros(z_num,z_num,d_num);
        soc_kp1_estimate_3c_fn_ref = @(soc_k,power_loss_param,gamma,gamma_tau,sigma_d,en_cap)soc_kp1_estimate_3c_fn(soc_k,power_loss_param,gamma,gamma_tau,sigma_d,en_cap);
        for d_k_idx = 1:d_num
            if d_k_idx == -params.d_offset
                P_Zp1gZD_map_t = zeros(z_num,z_num,1);
                for z_k_idx = 1:z_num
                    P_Zp1gZD_map_t(z_k_idx, z_k_idx, 1) = 1;
                end
                P_Zp1gZD_map(:,:,d_k_idx) = P_Zp1gZD_map_t;
            else
                soc_grid_bin_mean_t = soc_grid_bin_mean;
                scaled_soc_grid_boundaries_t = scaled_soc_grid_boundaries;
                soc_kp1_estimate_3c_fn_ref_t = soc_kp1_estimate_3c_fn_ref;

                alpha_est_t = nan(z_num,1);
                beta_est_t = nan(z_num,1);
                P_Zp1gZD_map_t = zeros(z_num,z_num,1);
                for z_k_idx = 1:z_num
                    power_loss_param = cell_mean_power_loss_param*legsInParallel*cellsInSeries;
                    soc_k = soc_grid_bin_mean_t(z_k_idx);
                    [soc_kp1_estimate_3c,action_performed] = soc_kp1_estimate_3c_fn_ref_t(soc_k,power_loss_param,gamma,gamma_tau,bat_pow_set(d_k_idx),z_cap);
                    if(action_performed)
                        beta_mean = rescale(soc_kp1_estimate_3c,0,1,'InputMin',cell_SOC_low,'InputMax',cell_SOC_high);
                        beta_var = min(beta_max_var,beta_mean*(1-beta_mean)/(min_alpha_plus_beta+1));

                        alpha_est = (((1-beta_mean)/beta_var) - (1/beta_mean))*(beta_mean^2);
                        beta_est = alpha_est*((1/beta_mean) - 1);
                        P_Zp1 = zeros(z_num,1);
                        if(isnan(alpha_est) || isnan(beta_est) || alpha_est<0 || beta_est<0)
                            prim_prob = 1 - (extreme_prob_fact);
                            sec_prob = (extreme_prob_fact/extreme_prob_range_num);
                            if(soc_kp1_estimate_3c==cell_SOC_low)
                                alpha_est = 0;
                                beta_est = 1;
                                P_Zp1(1) = prim_prob;
                                P_Zp1(2:extreme_prob_range_num) = sec_prob;
                            elseif(soc_kp1_estimate_3c==cell_SOC_high)
                                alpha_est = 1;
                                beta_est = 0;
                                P_Zp1(z_num) = prim_prob;
                                P_Zp1(z_num-extreme_prob_range_num+1:z_num-1) = sec_prob;
                            else
                                error('here beta dist not possible!')
                            end
                        else
                            for z_kp1_idx = 1:z_num
                                P_Zp1(z_kp1_idx) = betacdf(scaled_soc_grid_boundaries_t(z_kp1_idx+1),alpha_est,beta_est) - betacdf(scaled_soc_grid_boundaries_t(z_kp1_idx),alpha_est,beta_est);
                            end
                        end

                        P_Zp1 = P_Zp1/sum(P_Zp1);
                        P_Zp1 = roundOffInSimplex(P_Zp1,paramsPrecision);
                        P_Zp1gZD_map_t(:,z_k_idx) = P_Zp1;
                        alpha_est_t(z_k_idx) = alpha_est;
                        beta_est_t(z_k_idx) = beta_est;
                    end
                    send(progressDataQueue, incPercent);
                end

                alpha_est_map(:,d_k_idx) = alpha_est_t;
                beta_est_map(:,d_k_idx) = beta_est_t;
                P_Zp1gZD_map(:,:,d_k_idx) = P_Zp1gZD_map_t;
            end
        end
        progressData.terminate();
        save(filename,'alpha_est_map','beta_est_map','P_Zp1gZD_map','essProcessedDataParams');
    end

    essParams = rmfield(essProcessedDataParams,{'soc_grid_boundaries'});
end

if(~all(sum(P_Zp1gZD_map,1)>1-1e-12 | sum(P_Zp1gZD_map,1)==0,'all'))
    error('~all(sum(P_Zp1gZD,1)==1 | sum(P_Zp1gZD,1)==0,''all'')');
end

mean_soc_kp1_map = nan(z_num,d_num);
for d_k_idx = 1:d_num
    for z_k_idx = 1:z_num
        if(sum(P_Zp1gZD_map(:,z_k_idx,d_k_idx))>0)
            mean_soc_kp1_map(z_k_idx,d_k_idx) = soc_grid_bin_mean'*P_Zp1gZD_map(:,z_k_idx,d_k_idx);
        end
    end
end

essParams.P_Zp1gZD = P_Zp1gZD_map;
essParams.bat_pow_set = bat_pow_set;
essParams.z_cap = z_cap;
essParams.legsInParallel = legsInParallel;
essParams.cellsInSeries = cellsInSeries;
essParams.batteryNominalVoltage = batteryNominalVoltage;
essParams.capacityCostPerkWh = config.capacityCostPerkWh;

params.P_Zp1gZD = essParams.P_Zp1gZD;
if isfield(essParams, 'essDegradationCost_mean_map')
    params.essDegradationCost_mean_map = essParams.essDegradationCost_mean_map;
end

%% comsol params
if get_comsol_params
    deglifePartitions_num =  config.deglifePartitions_num;
    deglifePartitions = linspace(1,0.8,deglifePartitions_num+1);
    allowedRelativeCapacityChange = (deglifePartitions(1)-deglifePartitions(2))*100;

    comsolParams = struct;
    comsolParams.cell_1C_capacityInAh = config.cell_1C_capacityInAh;
    comsolParams.cell_nominalVoltage = config.cell_nominalVoltage;
    comsolParams.soc_grid_boundaries = config.l_grid_boundaries;
    comsolParams.cell_pow_set = config.cell_pow_set;
    comsolParams.slotIntervalInSeconds = config.slotIntervalInHours*3600;
    comsolParams.allowedRelativeCapacityChange = allowedRelativeCapacityChange;
    comsolParams.sample_num = config.sample_num;
    comsolParams.SOC_low = config.cell_SOC_low;
    comsolParams.SOC_high = config.cell_SOC_high;
    comsolParams.SOC_init = config.cell_SOC_init;
    comsolParams.cell_voltage_high = config.cell_voltage_high;
    comsolParams.cell_voltage_low = config.cell_voltage_low;
    comsolParams.driveToSOH_timeAccelerationFactor = config.driveToSOH_timeAccelerationFactor;
    comsolParams.driveToSOC_timeAccelerationFactor = config.driveToSOC_timeAccelerationFactor;
    comsolParams.driveToSOC_timePeriodScaleFactor = config.driveToSOC_timePeriodScaleFactor;
    comsolParams.driveToSOC_attempts_max = config.driveToSOC_attempts_max;
    comsolParams.gamma = config.gamma;
    comsolParams.gamma_tau = config.gamma_tau;
    comsolParams.bat_pow_set = config.bat_pow_set;
    comsolParams.z_cap = config.z_cap;
    comsolParams.legsInParallel = config.legsInParallel;
    comsolParams.cellsInSeries = config.cellsInSeries;
    comsolParams.batteryNominalVoltage = config.batteryNominalVoltage;
    comsolParams.capacityCostPerkWh = config.capacityCostPerkWh;
    comsolParams.slotIntervalInHours = config.slotIntervalInHours;
else
    comsolParams = [];
end

%% Supporting functions
    function [soc_kp1,action_performed] = soc_kp1_estimate_3c_fn(soc_k,power_loss_param,gamma,gamma_tau,sigma_d,en_cap)
        value_inside_square_root = 1 + (4*sigma_d/power_loss_param);
        action_performed = false;
        soc_kp1 = soc_k;
        if(value_inside_square_root>0)
            if(gamma==0)
                temp_calc = (power_loss_param/2/en_cap)*(sqrt(value_inside_square_root) - 1);
                soc_kp1 = soc_k + gamma_tau*temp_calc;
                soc_kp1 = min(max(soc_kp1,cell_SOC_low),cell_SOC_high);
                simTimeRatio = (soc_kp1 - soc_k)/temp_calc/slotIntervalInHours;
                if(simTimeRatio>0.5 || sigma_d==0)
                    action_performed = true;
                end
            else
                %                 soc_kp1 = (1-gamma)*soc_k + (gamma_tau*power_loss_param/2/en_cap)*(sqrt(value_inside_square_root) - 1);
                error('not implemented');
            end
        end
    end

    function [] = checkComsolConnection(path_to_comsol_app)
        import com.comsol.model.*
        import com.comsol.model.util.*
        try
            fprintf('Connecting to COMSOL server...')
            evalc('mphstart');
            mphopen -clear;
            fprintf('Done.\n')
        catch ME
            before_start_message = "Please check that a COMSOL server is started prior to calling";
            if(contains(ME.message,before_start_message))
                startComsolServer(path_to_comsol_app);
            end
            try
                mphopen -clear;
                fprintf('Done.\n')
            catch ME2
                error(ME2.message)
            end
        end
    end

    function [connection_success] = startComsolServer(path_to_comsol_app)
        import com.comsol.model.*
        import com.comsol.model.util.*
        connection_success = false;
        try
            if ispc
                system(strcat(path_to_comsol_app, "bin\win64\comsolmphserver.exe &"));
            else
                error("COMSOL start path not updated for unix")
            end
            pause(2)
            for attemp = 1:10
                try
                    evalc('mphstart');
                    connection_success = true;
                catch ME3
                    before_start_message = "Please check that a COMSOL server is started prior to calling";
                    if contains(ME3.message, before_start_message)
                        pause(2)
                    elseif contains(ME3.message, "license has expired")
                        error("COMSOL license has expired!");
                    else
                        warning(ME3.message);
                    end
                end
                if(connection_success)
                    break;
                end
            end
        catch ME3
            if contains(ME3.message, "license has expired")
                error("COMSOL license has expired!");
            else
                error(ME3.message);
            end
        end
    end

    function [comsol_model] = getComsolModel(comsol_model_filename)
        import com.comsol.model.*
        import com.comsol.model.util.*
        try
            comsol_model = mphopen(convertStringsToChars(comsol_model_filename),'sim_model','-nostore');
        catch ME1
            error(ME1.message)
        end
    end

    function parsave_cellSimData(filename,cellSimData,comsolParams)
        save(filename,'cellSimData','comsolParams')
    end
end