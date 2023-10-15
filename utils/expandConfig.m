function [config] = expandConfig(config)
h_num = config.hypothesisStatesNum;
hypothesisStatesPerAppliance = cell2mat(config.hypothesisStatesPerAppliance);
appliances_num = config.applianceGroupsNum;
h_vec_space = 1:hypothesisStatesPerAppliance(1);
for temp_t = 2:appliances_num
    h_vec_space = combvec(h_vec_space,1:hypothesisStatesPerAppliance(temp_t));
end

costPerApplianceDetection_emu = cell2mat(config.costPerApplianceDetection_emu);
C_HgHh_homogeneous = zeros(h_num);
for h_vec_idx = 1:h_num
    h_vec = h_vec_space(:,h_vec_idx);
    for hh_vec_idx = 1:h_num
        hh_vec = h_vec_space(:,hh_vec_idx);    
        ON_detection_app_flag = hh_vec>1;
        ON_detection_app_flag(h_vec~=hh_vec) = false;
        OFF_detection_app_flag = hh_vec==1;
        OFF_detection_app_flag(h_vec~=hh_vec) = false;
        C_HgHh_homogeneous(h_vec_idx,hh_vec_idx) = sum(1*costPerApplianceDetection_emu(ON_detection_app_flag)) +...
            sum(1*costPerApplianceDetection_emu(OFF_detection_app_flag));
    end
end

x_p_pu = config.emu_measurement_powerQuantPU; % in W
y_p_pu = config.smartmeter_powerQuantPU; % in W
d_p_pu = config.emu_measurement_powerQuantPU; % in W
y_control_p_pu = config.emu_control_powerQuantPU; % in W
minPowerDemandInW = config.minPowerDemandInW; % in W
maxPowerDemandInW_compensated = config.maxPowerDemandInW - minPowerDemandInW; % in W
x_max_pu = floor(maxPowerDemandInW_compensated/x_p_pu);
x_min_pu = 0;
x_num = x_max_pu-x_min_pu+1;
x_offset = x_min_pu - 1;

slotIntervalInSeconds = config.slotIntervalInSeconds;
slotIntervalInHours = slotIntervalInSeconds/3600; %in h

controlStartHourIndex = config.controlStartTime+1;
controlEndHourIndex = config.controlEndTime;

k_num = (controlEndHourIndex-controlStartHourIndex+1)/slotIntervalInHours; %Measurement slots
if(k_num~=floor(k_num))
    error('Wrong slotIntervalInSeconds setting!');
end
emu_measurement_energyQuantPU = x_p_pu*slotIntervalInHours; % in Wh

batteryNominalVoltage = config.batteryNominalVoltage;
cell_nominalVoltage = config.cell_nominalVoltage; %in V
cellsInSeries = ceil(batteryNominalVoltage/cell_nominalVoltage);
if(cellsInSeries~=floor(cellsInSeries))
    warning('Battery voltage is modified!');
end
batteryNominalVoltage = cellsInSeries*cell_nominalVoltage;% in V
converterEfficiency = (config.converterEfficiency)/100;
batteryRatedCapacityInAh = config.batteryRatedCapacityInAh; %in Ah
cell_SOC_high = config.cell_SOC_high;
cell_SOC_low = config.cell_SOC_low;
z_cap = (batteryRatedCapacityInAh*batteryNominalVoltage); % in Wh

bat_power_scaling_factor = config.bat_power_scaling_factor;

cell_1C_capacityInAh = config.cell_1C_capacityInAh; %in Ah
cell_1C_power = (cell_1C_capacityInAh*cell_nominalVoltage); %in W
legsInParallel = round(batteryRatedCapacityInAh/cell_1C_capacityInAh);
d_max_ch_ess = bat_power_scaling_factor*cell_1C_power*legsInParallel*cellsInSeries/converterEfficiency;
d_max_disch_ess = -bat_power_scaling_factor*cell_1C_power*legsInParallel*cellsInSeries*converterEfficiency;

d_rated = config.batteryRatedPower;
d_max_ch = min(d_rated,d_max_ch_ess);
d_max_disch = max(-d_rated,d_max_disch_ess);

if config.limit_y_range_to_x_range
    y_max_pu = round(x_max_pu*x_p_pu/y_p_pu);
    y_min_pu = round(x_min_pu*x_p_pu/y_p_pu);
    y_num = y_max_pu-y_min_pu+1;
    y_offset = y_min_pu - 1;

    y_control_max_pu = round(y_max_pu*y_p_pu/y_control_p_pu);
    y_control_min_pu = round(y_min_pu*y_p_pu/y_control_p_pu);
    y_control_num = y_control_max_pu-y_control_min_pu+1;
    y_control_offset = y_control_min_pu - 1;

    d_max_ch_req = (y_control_max_pu*y_control_p_pu - x_min_pu*x_p_pu);
    d_max_disch_req = (y_control_min_pu*y_control_p_pu - x_max_pu*x_p_pu);
    d_max_ch = min(d_max_ch,d_max_ch_req);
    d_max_disch = max(d_max_disch,d_max_disch_req);

    d_max_ch_pu = floor(d_max_ch/d_p_pu);
    d_max_disch_pu = ceil(d_max_disch/d_p_pu);
    out_pow_set = (d_max_disch_pu:d_max_ch_pu)*d_p_pu;
    d_num = length(out_pow_set);
    d_offset = d_max_disch_pu-1;
else
    d_max_ch_pu = floor(d_max_ch/d_p_pu);
    d_max_disch_pu = ceil(d_max_disch/d_p_pu);
    out_pow_set = (d_max_disch_pu:d_max_ch_pu)*d_p_pu;
    d_num = length(out_pow_set);
    d_offset = d_max_disch_pu-1;

    y_max_pu = x_max_pu + d_max_ch_pu;
    y_min_pu = x_min_pu + d_max_disch_pu;
    y_num = y_max_pu-y_min_pu+1;
    y_offset = y_min_pu - 1;

    y_control_max_pu = round(y_max_pu*y_p_pu/y_control_p_pu);
    y_control_min_pu = round(y_min_pu*y_p_pu/y_control_p_pu);
    y_control_num = y_control_max_pu-y_control_min_pu+1;
    y_control_offset = y_control_min_pu - 1;
end

bat_pow_set = zeros(1,d_num);
for pow_idx = 1:d_num
    if(out_pow_set(pow_idx)<0)
        bat_pow_set(pow_idx) = out_pow_set(pow_idx)/converterEfficiency;
    else
        bat_pow_set(pow_idx) = out_pow_set(pow_idx)*converterEfficiency;
    end
end

z_min_pu = floor(cell_SOC_low*z_cap/emu_measurement_energyQuantPU);
z_max_pu = floor(cell_SOC_high*z_cap/emu_measurement_energyQuantPU);
z_num = z_max_pu-z_min_pu+1;

l_num_paramapprox = config.batteryLevelsNum_paramapprox;
z_num_per_level_comsol = floor(z_num/l_num_paramapprox);
z_num = z_num_per_level_comsol*l_num_paramapprox;
z_offset = z_min_pu - 1;
% cell_SOC_high = roundOff((z_num+z_offset)*emu_measurement_energyQuantPU/z_cap,paramsPrecision);
l_grid_boundaries_comsol = linspace(cell_SOC_low,cell_SOC_high,l_num_paramapprox+1);

l_grid_bin_mean_comsol = zeros(l_num_paramapprox,1);
for bin_idx = 1:l_num_paramapprox
    l_grid_bin_mean_comsol(bin_idx) = (l_grid_boundaries_comsol(bin_idx) + l_grid_boundaries_comsol(bin_idx+1))/2;
end

cell_pow_set = round(bat_pow_set/(cellsInSeries*legsInParallel),12);
soc_grid_boundaries = linspace(cell_SOC_low,cell_SOC_high,z_num+1);
soc_grid_bin_mean = zeros(z_num,1);
for bin_idx = 1:z_num
    soc_grid_bin_mean(bin_idx) = (soc_grid_boundaries(bin_idx) + soc_grid_boundaries(bin_idx+1))/2;
end

batterySelfDischargeRatePerMonth = 0; % factor in [0,1]
if(batterySelfDischargeRatePerMonth >0)
    tau = 30*24/-log(1-batterySelfDischargeRatePerMonth); %h
    gamma = 1 - exp(-slotIntervalInHours/tau);
    gamma_tau = gamma*tau; % in h
else
    gamma = 0;
    gamma_tau = slotIntervalInHours; % in h
end
z_k_idxs_given_l_k_idx_comsol = cell(1, l_num_paramapprox);
for z_k_idx=1:z_num
    z_grid_bin_mean_ = soc_grid_bin_mean(z_k_idx);
    l_k_idx_comsol = find(z_grid_bin_mean_<=l_grid_boundaries_comsol,1)-1;
    z_k_idxs_given_l_k_idx_comsol{l_k_idx_comsol} = [z_k_idxs_given_l_k_idx_comsol{l_k_idx_comsol},z_k_idx];
end

% Expand Config
config.h_num = h_num;
config.x_num = x_num;
config.z_num = z_num;
config.d_num = d_num;
config.l_num_paramapprox = l_num_paramapprox;
config.z_k_idxs_given_l_k_idx_comsol = z_k_idxs_given_l_k_idx_comsol;
config.batteryRatedCapacityInAh = batteryRatedCapacityInAh;
config.cell_1C_capacityInAh = cell_1C_capacityInAh;
config.cell_nominalVoltage = cell_nominalVoltage;
config.l_grid_boundaries_comsol = l_grid_boundaries_comsol;
config.cell_pow_set = cell_pow_set;
config.cell_SOC_low = cell_SOC_low;
config.cell_SOC_high = cell_SOC_high;
config.l_grid_bin_mean_comsol = l_grid_bin_mean_comsol;
config.soc_grid_boundaries = soc_grid_boundaries;
config.gamma = gamma;
config.gamma_tau = gamma_tau;
config.legsInParallel = legsInParallel;
config.cellsInSeries = cellsInSeries;
config.bat_pow_set = bat_pow_set;
config.z_cap = z_cap;
config.slotIntervalInHours = slotIntervalInHours;
config.soc_grid_bin_mean = soc_grid_bin_mean;
config.C_HgHh_homogeneous = C_HgHh_homogeneous;
config.k_num = k_num;
config.x_offset = x_offset;
config.x_p_pu = x_p_pu;
config.y_p_pu = y_p_pu;
config.y_control_p_pu = y_control_p_pu;
config.d_p_pu = d_p_pu;
config.y_num = y_num;
config.y_control_num = y_control_num;
config.d_offset = d_offset;
config.y_offset = y_offset;
config.y_control_offset = y_control_offset;
config.h_vec_space = h_vec_space;
config.hypothesisStatesPerAppliance = hypothesisStatesPerAppliance;
config.z_offset = z_offset;
end
