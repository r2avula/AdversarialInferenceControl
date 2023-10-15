clear;
rng_id_sim = 1;
simStartup(0,rng_id_sim);
dbstop if error

storeAuxData = true;
storeBeliefs = false;
showGUI = true;
useparpool = true;
UseGPU = true;

%% configs with ESS data available
config_filename = 'stove_oven_h2_x5_yc5_z35_k48.yaml';

%% params initialization
[fileDir,~,~] = fileparts(pwd);
cache_folder_path = [fileDir filesep 'AdversarialInferenceControl_Cache' filesep 'RealDataSimulations', filesep];
path_to_raw_sm_data = [fileDir filesep 'KTH_LIL_Data', filesep];
if ispc
    path_to_comsol_app = "C:\Program Files\COMSOL\COMSOL61\Multiphysics\";
else
    path_to_comsol_app = "";
end

config = yaml.loadFile(config_filename);
config.cache_folder_path = cache_folder_path;
config.path_to_raw_sm_data = path_to_raw_sm_data;
config.path_to_comsol_app = path_to_comsol_app;

NC= config.NC;
subopt_DBS_FDC_UA = config.subopt_DBS_FDC_UA;
best_effort_moderation = config.best_effort_moderation;
differential_privacy = config.differential_privacy;
AMDPG = config.AMDPG;

if(isequal(config.dataset,'kth_lil'))
    [smData,applianceData,gtData,dateStrings] = fetch_KTH_LIL_Data(config);
else
    error('Not implemented!');
end
data_training_validation_ratio = config.data_training_validation_ratio;

totalDays = size(smData,2);
trainingDays = round(totalDays*data_training_validation_ratio);
additional_data.training_data.training_applianceData = applianceData(:,1:trainingDays,:);
additional_data.training_data.training_noiseData = max((smData - sum(applianceData,3)),0);
additional_data.training_data.training_gtData = gtData(:,1:trainingDays,:);

[params_FDC, config] = initParams(config,additional_data,false);

validation_days = totalDays-trainingDays;
sm_data_test = smData(:,(trainingDays + 1):(trainingDays + validation_days));
gt_data_test = gtData(:,(trainingDays + 1):(trainingDays + validation_days),:);
dateStrings_test = dateStrings((trainingDays + 1):(trainingDays + validation_days));

hypothesisStatesPerAppliance = params_FDC.hypothesisStatesPerAppliance;
h_vec_space = params_FDC.h_vec_space;
k_num = params_FDC.k_num;

gt_data_vecIdx_test = zeros(k_num,validation_days);
for dayIdx = 1:validation_days
    [~,gt_data_vecIdx_test(:,dayIdx)] = ismember(reshape(gt_data_test(:,dayIdx,:),k_num,[]),h_vec_space','rows');
end
appliances_num = size(h_vec_space,1);
h_0_idxs_test = gt_data_vecIdx_test(k_num,:);
gt_data_vecIdx_test_vec = reshape(gt_data_vecIdx_test,[],1);

numHorizons = config.numEvalHorizons;
evalDayIdxs = randi([1 validation_days],1,numHorizons);

sm_data_test = sm_data_test(:,evalDayIdxs);
gt_data_test = gt_data_test(:,evalDayIdxs,:);
h_0_idxs_test = h_0_idxs_test(evalDayIdxs);
in_debug_mode = false;

%% Optimal Bayesian detection without controller
if(NC)
    fprintf('Optimal Bayesian detection without controller --- \n');
    evalParams = struct;
    evalParams.x_num = params_FDC.x_num;
    evalParams.y_num = params_FDC.y_num;
    evalParams.h_num = params_FDC.h_num;
    evalParams.k_num = k_num;
    evalParams.C_HgHh_design = params_FDC.C_HgHh_design;
    evalParams.C_HgHh_homogeneous = params_FDC.C_HgHh_homogeneous;
    evalParams.x_p_pu = params_FDC.x_p_pu;
    evalParams.y_p_pu = params_FDC.y_p_pu;
    evalParams.x_offset = params_FDC.x_offset;
    evalParams.y_offset = params_FDC.y_offset;
    evalParams.P_XHgHn1 = params_FDC.P_XHgHn1;
    evalParams.beliefSpacePrecision_adv = params_FDC.beliefSpacePrecision_adv;
    evalParams.paramsPrecision = params_FDC.paramsPrecision;
    evalParams.minLikelihoodFilter = params_FDC.minLikelihoodFilter;
    evalParams.minPowerDemandInW = params_FDC.minPowerDemandInW;
    evalParams.P_H0 = params_FDC.P_H0;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.numHorizons = numHorizons;
    fileNamePrefix = sprintf('%sevalData_NC_',cache_folder_path);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'detectionData','fscores', 'tp', 'tn','precision');
    else
        rng(rng_id_sim,'twister');
        detectionData = runSequentialBayesDetection_NC(evalParams, sm_data_test, gt_data_test, false);
        [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, detectionData.estimatedhypStateData);
        save(fileFullPath,'detectionData','fscores', 'tp', 'tn','evalParams','precision')
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_NC = detectionData.bayesian_reward;
    precision_NC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_NC, num2str(precision_NC'));
end

[params_FDC, essParams, comsolParams] = getEssParams(config, params_FDC);
params_FDC_bkup = params_FDC;

%% SubOptimal controller with discrete belief space designed against unaware adversary
if(subopt_DBS_FDC_UA)
    fprintf('Sub-optimal infinite horizon control of UA adv using discrete FD belief space ---\n');
    params_FDC = params_FDC_bkup;
    params_FDC.discountFactor= config.discountFactor_MDP;
    params_FDC.beliefSpacePrecision_EMU_subopt_DBS = config.beliefSpacePrecision_EMU_subopt_DBS;
    value_iter_conv_threshold = config.value_iter_conv_threshold;
    max_valueFnIterations = config.max_valueFnIterations;

    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','k_num','x_p_pu','x_offset','y_offset','y_control_p_pu','y_control_num','y_control_offset',...
        'minPowerDemandInW','beliefSpacePrecision_adv','P_ZgA','P_H0','minLikelihoodFilter',...
        'discountFactor','beliefSpacePrecision_EMU_subopt_DBS','d_num','d_p_pu','y_p_pu','u_num','C_HgHh_homogeneous','s_num'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    clear policy;
    pp_data_fileNamePrefix = sprintf('%sppdata_FD_',cache_folder_path);
    [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);
    PP_data_filenamePrefix = sprintf('%spp_data_UA_subopt_DBS_FDC_',cache_folder_path);
    PP_data_fileFullPath = get_PP_data_UA_subopt_DBS_FDC_filename(used_params,PP_data_filenamePrefix,in_debug_mode, pp_data, useparpool);

    policy_fileNamePrefix = sprintf('%spolicy_UA_subopt_DBS_FDC_',cache_folder_path);
    policy = get_policy_UA_subopt_DBS_FDC(used_params,max_valueFnIterations,value_iter_conv_threshold,...
        PP_data_fileFullPath,policy_fileNamePrefix, pp_data);

    if(policy.isTerminated)
        error('\tPolicy terminated in %d iterations. max_valueFunction_diff: %.2e! [Threshold: %.2e]\n',policy.iter_idx,policy.max_val_inc,value_iter_conv_threshold);
    end
    if(policy.isConverged)
        fprintf('\tPolicy converged in %d iterations. max_valueFunction_diff: %.2e! [Threshold: %.2e]\n',policy.iter_idx,policy.max_val_inc,value_iter_conv_threshold);
    end

    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.policy_iter_idx = policy.iter_idx;
    clear optimalControlData_UA;
    fileNamePrefix = sprintf('%sevalData_UA_subopt_DBS_FDC_',cache_folder_path);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData_UA','fscores', 'tp', 'tn','detectionData_AA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA");
    else
        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        rng(rng_id_sim,'twister');
        [optimalControlData_UA, detectionData_AA] = simulate_UA_subopt_DBS_FDC(evalParams_t,policy,PP_data_fileFullPath, pp_data, false);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, detectionData_AA.estimatedhypStateData);
        [fscores_UA, tp_UA, tn_UA,precision_UA] = computeFScores(params_FDC, gt_data_test, optimalControlData_UA.estimatedhypStateData);
        save(fileFullPath,'optimalControlData_UA','evalParams','fscores', 'tp', 'tn','detectionData_AA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_UA_subopt_DBS_FDC = mean([detectionData_AA.bayesian_reward]);
    precision_UA_subopt_DBS_FDC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_UA_subopt_DBS_FDC, num2str(precision_UA_subopt_DBS_FDC'));

    fprintf('\t---When tested with UA adv--- \n');
    reward_UA_subopt_DBS_FDC_UA = mean([optimalControlData_UA.bayesian_reward]);
    precision_UA_subopt_DBS_FDC_UA = precision_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_UA_subopt_DBS_FDC_UA, num2str(precision_UA_subopt_DBS_FDC_UA'));
end

%% Passive controller with best effort moderation strategy
if(best_effort_moderation)
    fprintf('Optimal Bayesian detection with BEM controller --- \n');
    params_FDC = params_FDC_bkup;
    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_num','x_p_pu','y_p_pu','d_p_pu','x_offset','y_offset','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','k_num','minPowerDemandInW','beliefSpacePrecision_adv','y_control_p_pu','y_control_num','y_control_offset',...
        'P_ZgA','P_H0','minLikelihoodFilter','u_num','s_num','C_HgHh_homogeneous','P_HgHn1','P_XgH'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end
    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.storeYkn1Idxs = false;
    fileNamePrefix = sprintf('%sevalData_BEM_',cache_folder_path);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'bemControlData','tp', 'tn','fscores','precision');
    else
        pp_data_fileNamePrefix = sprintf('%sppdata_FD_',cache_folder_path);
        [pp_data_FD, ~] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

        pp_data_fileNamePrefix = sprintf('%sppdata_BEM_FD_',cache_folder_path);
        [pp_data_BEM] = get_ppdata_BEM_FD(used_params,pp_data_fileNamePrefix,pp_data_FD);

        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        evalParams_t.in_debug_mode = in_debug_mode;

        rng(rng_id_sim,'twister');
        [bemControlData] = simulate_BEM_FDC(evalParams_t, pp_data_BEM);
        [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, bemControlData.estimatedhypStateData);

        save(fileFullPath,'bemControlData','evalParams','fscores', 'tp', 'tn','precision')
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end

    reward_BEM = bemControlData.bayesian_reward;
    precision_BEM = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_BEM, num2str(precision_BEM'));
end

%% Passive controller with differential privacy
if(differential_privacy)
    differential_privacy_epsilon_ranges = 10.^(-9:9);
    epsilon_sweep_results = zeros(length(differential_privacy_epsilon_ranges),3);
    fprintf('Optimal Bayesian detection with differential_privacy controller--- \n');
    for idx = 1:length(epsilon_sweep_results)
        differential_privacy_epsilon = differential_privacy_epsilon_ranges(idx);
        params_FDC = params_FDC_bkup;
        params_FDC.differential_privacy_epsilon = differential_privacy_epsilon;
        used_params = struct;
        used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_num','x_p_pu','y_p_pu','d_p_pu','x_offset','y_offset','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
            'P_HgA','C_HgHh_design','k_num','minPowerDemandInW','beliefSpacePrecision_adv','y_control_p_pu','y_control_num','y_control_offset',...
            'P_ZgA','P_H0','minLikelihoodFilter','u_num','s_num','C_HgHh_homogeneous','P_HgHn1','P_XgH','differential_privacy_epsilon'};
        for fn = used_fieldnames
            used_params.(fn{1}) = params_FDC.(fn{1});
        end
        evalParams = struct;
        evalParams.params = used_params;
        evalParams.numHorizons = numHorizons;
        evalParams.storeAuxData = storeAuxData;
        evalParams.storeBeliefs = storeBeliefs;
        evalParams.storeYkn1Idxs = false;
        fileNamePrefix = sprintf('%sevalData_differential_privacy_',cache_folder_path);
        [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
        [~,filename] = fileparts(fileFullPath);
        if(fileExists)
            load(fileFullPath,'differential_privacy_ControlData','tp', 'tn','fscores','precision');
        else
            pp_data_fileNamePrefix = sprintf('%sppdata_FD_',cache_folder_path);
            [pp_data_FD, ~] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

            pp_data_fileNamePrefix = sprintf('%sppdata_differential_privacy_FD_',cache_folder_path);
            [pp_data_differential_privacy] = get_ppdata_differential_privacy_FD(used_params,pp_data_fileNamePrefix,pp_data_FD);

            evalParams_t = evalParams;
            evalParams_t.sm_data = sm_data_test;
            evalParams_t.gt_data = gt_data_test;
            evalParams_t.h_0_idxs = h_0_idxs_test;
            evalParams_t.in_debug_mode = in_debug_mode;

            rng(rng_id_sim,'twister');
            [differential_privacy_ControlData] = simulate_differential_privacy_FDC(evalParams_t, pp_data_differential_privacy);
            [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, differential_privacy_ControlData.estimatedhypStateData);

            save(fileFullPath,'differential_privacy_ControlData','evalParams','fscores', 'tp', 'tn','precision')
        end

        reward_differential_privacy = differential_privacy_ControlData.bayesian_reward;
        precision_differential_privacy = precision;
        epsilon_sweep_results(idx,1) = differential_privacy_epsilon;
        epsilon_sweep_results(idx,2) = reward_differential_privacy;
        epsilon_sweep_results(idx,3) = precision_differential_privacy;
    end
    [~,min_reward_idx] = min(min(epsilon_sweep_results(:,2)));
    [~,max_reward_idx] = max(min(epsilon_sweep_results(:,2)));
    fprintf('\tMin Bayesian reward: %f; Max Bayesian reward: %f\n', min(epsilon_sweep_results(min_reward_idx,2)), max(epsilon_sweep_results(max_reward_idx,2)));
    fprintf('\tMin precision: %f; Max precision: %f\n', min(epsilon_sweep_results(min_reward_idx,3)), max(epsilon_sweep_results(max_reward_idx,3)));
end

%% AMDPG Reinforcement learning based controller
if(AMDPG)
    disp('SubOptimal infinite horizon control using AMDPG Reinforcement learning based controller--- ');
    params_FDC = params_FDC_bkup;
    params_FDC.numTrainHorizons= config.numTrainHorizons;
    params_FDC.discountFactor = config.discountFactor_RL;
    params_FDC.learning_rate_Ac = config.learning_rate_Ac;
    params_FDC.learning_rate_C = config.learning_rate_C;
    params_FDC.TargetSmoothFactor_C = config.TargetSmoothFactor_C;
    params_FDC.MiniBatchSize = config.MiniBatchSize;
    params_FDC.ReplayBufferLength = config.ReplayBufferLength;
    params_FDC.TargetSmoothFactor_Ac = config.TargetSmoothFactor_Ac;
    params_FDC.actor_net_hidden_layers = config.actor_net_hidden_layers;
    params_FDC.actor_net_hidden_layer_neurons_ratio_obs = config.actor_net_hidden_layer_neurons_ratio_obs;
    params_FDC.actor_net_hidden_layer_neurons_ratio_act = config.actor_net_hidden_layer_neurons_ratio_act;
    params_FDC.critic_net_hidden_layers = config.critic_net_hidden_layers;
    params_FDC.critic_net_hidden_layer_neurons_ratio_obs = config.critic_net_hidden_layer_neurons_ratio_obs;
    params_FDC.critic_net_hidden_layer_neurons_ratio_act = config.critic_net_hidden_layer_neurons_ratio_act;
    params_FDC.y_num_for_exploration = config.y_num_for_exploration;
    params_FDC.num_rand_adv_strats_for_exploration = config.num_rand_adv_strats_for_exploration;
    params_FDC.logistic_param_limit = config.logistic_param_limit;
    params_FDC.GradientDecayFactor_Ac = config.GradientDecayFactor_Ac;
    params_FDC.SquaredGradientDecayFactor_Ac = config.SquaredGradientDecayFactor_Ac;
    params_FDC.Epsilon_adam_Ac = config.Epsilon_adam_Ac;
    params_FDC.GradientDecayFactor_C = config.GradientDecayFactor_C;
    params_FDC.SquaredGradientDecayFactor_C = config.SquaredGradientDecayFactor_C;
    params_FDC.Epsilon_adam_C = config.Epsilon_adam_C;
    params_FDC.noise_epsilon = config.noise_epsilon;
    params_FDC.with_bem_initialized_actor = config.with_bem_initialized_actor;
    params_FDC.noise_sd = config.noise_sd;
    params_FDC.with_controller_reward = config.with_controller_reward;
    params_FDC.with_mean_reward = config.with_mean_reward;
    params_FDC.with_a_nncells = config.with_a_nncells;
    params_FDC.exploration_epsilon = config.exploration_epsilon;

    used_params = struct;
    used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
        'y_offset','x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','C_HgHh_design','paramsPrecision','minLikelihoodFilter',...
        'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0','learning_rate_Ac','learning_rate_C',...
        'discountFactor','P_XHgHn1','numTrainHorizons','minPowerDemandInW','y_p_pu','u_num','with_mean_reward',...
        'actor_net_hidden_layers','actor_net_hidden_layer_neurons_ratio_obs','actor_net_hidden_layer_neurons_ratio_act','critic_net_hidden_layers','TargetSmoothFactor_Ac',...
        'critic_net_hidden_layer_neurons_ratio_obs','TargetSmoothFactor_C','y_num_for_exploration','with_a_nncells',...
        'exploration_epsilon','MiniBatchSize','ReplayBufferLength','with_bem_initialized_actor','noise_sd',...
        'C_HgHh_homogeneous','num_rand_adv_strats_for_exploration','s_num','with_controller_reward',...
        'logistic_param_limit','GradientDecayFactor_Ac','SquaredGradientDecayFactor_Ac','Epsilon_adam_Ac','GradientDecayFactor_C',...
        'SquaredGradientDecayFactor_C','Epsilon_adam_C','noise_epsilon','critic_net_hidden_layer_neurons_ratio_act'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    policy_fileNamePrefix = sprintf('%spolicy_RL_DeterministicActorCriticAgent_',cache_folder_path);
    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.numTrainHorizons = used_params.numTrainHorizons;
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_RL_DeterministicActorCriticAgent_FDA_',cache_folder_path);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData', 'tp', 'tn','fscores','precision',"precision_UA","tn_UA","tp_UA","fscores_UA",'detectionData_UA');
    else
        pp_data_fileNamePrefix = sprintf('%sppdata_FD_',cache_folder_path);
        [pp_data_FD, ~] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

        pp_data_fileNamePrefix = sprintf('%sppdata_BEM_FD_',cache_folder_path);
        [pp_data_BEM] = get_ppdata_BEM_FD(used_params,pp_data_fileNamePrefix,pp_data_FD);
        BEM_initialized_actor_fileNamePrefix = sprintf('%sBEM_initialized_actor_',cache_folder_path);

        clear rl_agent;
        [rl_agent, policy_fileFullPath] = get_DeterministicActorCriticAgent(used_params,policy_fileNamePrefix, showGUI, UseGPU, pp_data_BEM, BEM_initialized_actor_fileNamePrefix, true);
        plot_RL_training_data(policy_fileFullPath, [], []);

        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        evalParams_t.rl_agent = rl_agent;
        rng(rng_id_sim,'twister');
        optimalControlData = simulate_ActorCriticAgent(evalParams_t, pp_data_FD, useparpool);
        [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, optimalControlData.estimatedhypStateData);

        evalParams_t = struct;
        evalParams_t.x_num = params_FDC.x_num;
        evalParams_t.y_num = params_FDC.y_num;
        evalParams_t.h_num = params_FDC.h_num;
        evalParams_t.k_num = k_num;
        evalParams_t.C_HgHh_design = params_FDC.C_HgHh_design;
        evalParams_t.C_HgHh_homogeneous = params_FDC.C_HgHh_homogeneous;
        evalParams_t.x_p_pu = params_FDC.x_p_pu;
        evalParams_t.y_p_pu = params_FDC.y_p_pu;
        evalParams_t.x_offset = params_FDC.x_offset;
        evalParams_t.y_offset = params_FDC.y_offset;
        evalParams_t.P_XHgHn1 = params_FDC.P_XHgHn1;
        evalParams_t.beliefSpacePrecision_adv = params_FDC.beliefSpacePrecision_adv;
        evalParams_t.paramsPrecision = params_FDC.paramsPrecision;
        evalParams_t.minLikelihoodFilter = params_FDC.minLikelihoodFilter;
        evalParams_t.minPowerDemandInW = params_FDC.minPowerDemandInW;
        evalParams_t.P_H0 = params_FDC.P_H0;
        evalParams_t.storeBeliefs = storeBeliefs;

        rng(rng_id_sim,'twister');
        detectionData_UA = runSequentialBayesDetection_NC(evalParams_t, optimalControlData.modifiedSMdata, gt_data_test,false);
        [fscores_UA, tp_UA, tn_UA,precision_UA] = computeFScores(params_FDC, gt_data_test, detectionData_UA.estimatedhypStateData);

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','precision',"precision_UA","tn_UA","tp_UA","fscores_UA",'detectionData_UA')
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_AMDPG = optimalControlData.bayesian_reward;
    precision_AMDPG = precision;
    fprintf('\tBayesian reward: %f; fscore: %s\n', reward_AMDPG, num2str(precision_AMDPG'));

    fprintf('\t---When tested with UA adv--- \n');
    reward_AMDPG_UA = mean([detectionData_UA.bayesian_reward]);
    precision_AMDPG_UA = precision_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_AMDPG_UA, num2str(precision_AMDPG_UA'));
end