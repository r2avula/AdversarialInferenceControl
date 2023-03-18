clear;
rng_id_sim = 1;
simStartup(0,rng_id_sim);
dbstop if error

storeAuxData = true;
storeBeliefs = false;

%% configs with ESS data available
config_filename = 'stove_oven_h2_x12_yc12_z70_l5_k48.yaml';

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
RL_DeterministicActorCriticAgent_RD = config.RL_DeterministicActorCriticAgent_RD;
subopt_DBS_FDC_UA = config.subopt_DBS_FDC_UA;

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

[params_FDC, params_RDC, config] = initParams(config,additional_data,false);

validation_days = totalDays-trainingDays;
sm_data_test = smData(:,(trainingDays + 1):(trainingDays + validation_days));
gt_data_test = gtData(:,(trainingDays + 1):(trainingDays + validation_days),:);
dateStrings_test = dateStrings((trainingDays + 1):(trainingDays + validation_days));

hypothesisStatesPerAppliance = params_RDC.hypothesisStatesPerAppliance;
h_vec_space = params_RDC.h_vec_space;
k_num = params_RDC.k_num;

gt_data_vecIdx_test = zeros(k_num,validation_days);
for dayIdx = 1:validation_days
    [~,gt_data_vecIdx_test(:,dayIdx)] = ismember(reshape(gt_data_test(:,dayIdx,:),k_num,[]),h_vec_space','rows');
end
appliances_num = size(h_vec_space,1);
h_0_idxs_test = gt_data_vecIdx_test(k_num,:);
gt_data_vecIdx_test_vec = reshape(gt_data_vecIdx_test,[],1);

params_FDC_bkup = params_RDC;
numHorizons = config.numEvalHorizons;
evalDayIdxs = randi([1 validation_days],1,numHorizons);

sm_data_test = sm_data_test(:,evalDayIdxs);
gt_data_test = gt_data_test(:,evalDayIdxs);
h_0_idxs_test = h_0_idxs_test(evalDayIdxs);
in_debug_mode = false;

%% Optimal Bayesian detection without controller
if(NC)
    fprintf('Optimal Bayesian detection without controller --- \n');
    evalParams = struct;
    evalParams.x_num = params_RDC.x_num;
    evalParams.y_num = params_RDC.y_num;
    evalParams.h_num = params_RDC.h_num;
    evalParams.k_num = k_num;
    evalParams.C_HgHh_design = params_FDC.C_HgHh_design;
    evalParams.C_HgHh_homogeneous = params_FDC.C_HgHh_homogeneous;
    evalParams.x_p_pu = params_RDC.x_p_pu;
    evalParams.y_p_pu = params_RDC.y_p_pu;
    evalParams.x_offset = params_RDC.x_offset;
    evalParams.y_offset = params_RDC.y_offset;
    evalParams.P_XHgHn1 = params_RDC.P_XHgHn1;
    evalParams.beliefSpacePrecision_adv = params_RDC.beliefSpacePrecision_adv;
    evalParams.paramsPrecision = params_RDC.paramsPrecision;
    evalParams.minLikelihoodFilter = params_RDC.minLikelihoodFilter;
    evalParams.minPowerDemandInW = params_RDC.minPowerDemandInW;
    evalParams.P_H0 = params_RDC.P_H0;
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
    risk_NC = detectionData.mean_correct_detection;
    reward_NC = detectionData.bayesian_reward;
    fscores_NC = fscores;
    precision_NC = precision;
    recall_NC = tp;
    tn_NC = tn;
    mean_PYkgY12kn1_NC = detectionData.mean_PYkgY12kn1;
    fprintf('\tBayesian risk: %f; Bayesian reward: %f; precision: %s; recall: %s; fscore: %s; mean_PYkgY12kn1: %s\n',...
        risk_NC, reward_NC, num2str(precision_NC'), num2str(recall_NC'), num2str(fscores_NC'), num2str(mean_PYkgY12kn1_NC'));
end

[params_RDC, essParams, comsolParams] = getEssParams(config, params_RDC);
params_RDC_bkup = params_RDC;

showGUI = true; 
useparpool = true;
UseGPU = true;

%% Reinforcement learning based controller
if(RL_DeterministicActorCriticAgent_RD)
    disp('SubOptimal infinite horizon control using TD Actor-critic RL Deterministic policy approach with FD belief space--- ');
    params_RDC = params_RDC_bkup;
    params_RDC.numTrainHorizons= config.numTrainHorizons;
    params_RDC.discountFactor = config.discountFactor;
    params_RDC.learning_rate_Ac = config.learning_rate_Ac;
    params_RDC.learning_rate_C = config.learning_rate_C;
    params_RDC.exploration_epsilon = config.exploration_epsilon;
    params_RDC.TargetSmoothFactor_C = config.TargetSmoothFactor_C;
    params_RDC.MiniBatchSize = config.MiniBatchSize;
    params_RDC.InMemoryUpdateInterval = config.InMemoryUpdateInterval;
    params_RDC.ReplayBufferLength = config.ReplayBufferLength;
    params_RDC.TargetSmoothFactor_Ac = config.TargetSmoothFactor_Ac;
    params_RDC.actor_net_hidden_layers = config.actor_net_hidden_layers;
    params_RDC.actor_net_hidden_layer_neurons_ratio = config.actor_net_hidden_layer_neurons_ratio;
    params_RDC.critic_net_hidden_layers = config.critic_net_hidden_layers;
    params_RDC.critic_net_hidden_layer_neurons_ratio_obs = config.critic_net_hidden_layer_neurons_ratio_obs;
    params_RDC.critic_net_hidden_layer_neurons_ratio_act = config.critic_net_hidden_layer_neurons_ratio_act;    
    params_RDC.penalty_factor = config.penalty_factor;
    params_RDC.y_num_for_exploration = config.y_num_for_exploration;
    params_RDC.num_rand_adv_strats_for_exploration = config.num_rand_adv_strats_for_exploration;
    params_RDC.logistic_param_limit = config.logistic_param_limit;
    params_RDC.u_num = params_FDC.u_num;

    params_RDC.GradientDecayFactor_Ac = config.GradientDecayFactor_Ac;
    params_RDC.SquaredGradientDecayFactor_Ac = config.SquaredGradientDecayFactor_Ac;
    params_RDC.Epsilon_adam_Ac = config.Epsilon_adam_Ac;
    params_RDC.GradientDecayFactor_C = config.GradientDecayFactor_C;
    params_RDC.SquaredGradientDecayFactor_C = config.SquaredGradientDecayFactor_C;
    params_RDC.Epsilon_adam_C = config.Epsilon_adam_C;
    params_RDC.noise_epsilon = config.noise_epsilon;

    used_params = struct;
    used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
        'y_offset','x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','C_HgHh_design','paramsPrecision','minLikelihoodFilter',...
        'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0','learning_rate_Ac','learning_rate_C',...
        'discountFactor','P_XHgHn1','numTrainHorizons','minPowerDemandInW','y_p_pu','u_num'...
        'actor_net_hidden_layers','actor_net_hidden_layer_neurons_ratio','critic_net_hidden_layers','TargetSmoothFactor_Ac',...
        'critic_net_hidden_layer_neurons_ratio_obs','TargetSmoothFactor_C','penalty_factor','y_num_for_exploration',...
        'exploration_epsilon','MiniBatchSize','InMemoryUpdateInterval','ReplayBufferLength',...
        'l_num','w_num','LIdxgZIdx','ZIdxsgLIdx','z_num_per_level','b_num','t_num','P_HgB','P_BgA','C_HgHh_homogeneous','num_rand_adv_strats_for_exploration',...
       'logistic_param_limit','GradientDecayFactor_Ac','SquaredGradientDecayFactor_Ac','Epsilon_adam_Ac','GradientDecayFactor_C',...
       'SquaredGradientDecayFactor_C','Epsilon_adam_C','noise_epsilon','critic_net_hidden_layer_neurons_ratio_act'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_RDC.(fn{1});
    end

    policy_fileNamePrefix = sprintf('%spolicy_RL_DeterministicActorCriticAgent_RD_',cache_folder_path);
    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.numTrainHorizons = used_params.numTrainHorizons;
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_RL_DeterministicActorCriticAgent_RD_FDA_',cache_folder_path);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData', 'tp', 'tn','fscores','precision',"precision_UA","tn_UA","tp_UA","fscores_UA",'detectionData_UA');
    else
        pp_data_fileNamePrefix = sprintf('%sppdata_RD_',cache_folder_path);
        [pp_data_RD, ~] = get_ppdata_RD(used_params,pp_data_fileNamePrefix);

        clear rl_agent;
        [rl_agent, policy_fileFullPath] = get_DeterministicActorCriticAgent_RDC(used_params,policy_fileNamePrefix, showGUI, pp_data_RD, UseGPU);  
        plot_RL_training_data(policy_fileFullPath, [], []);

        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        evalParams_t.rl_agent = rl_agent;
        rng(rng_id_sim,'twister');
        optimalControlData = simulate_ActorCriticAgent_RDC(evalParams_t, pp_data_RD);
        [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, optimalControlData.estimatedhypStateData);

        evalParams_t = struct;
        evalParams_t.x_num = params_RDC.x_num;
        evalParams_t.y_num = params_RDC.y_num;
        evalParams_t.h_num = params_RDC.h_num;
        evalParams_t.k_num = k_num;
        evalParams_t.C_HgHh_design = params_FDC.C_HgHh_design;
        evalParams_t.C_HgHh_homogeneous = params_FDC.C_HgHh_homogeneous;
        evalParams_t.x_p_pu = params_RDC.x_p_pu;
        evalParams_t.y_p_pu = params_RDC.y_p_pu;
        evalParams_t.x_offset = params_RDC.x_offset;
        evalParams_t.y_offset = params_RDC.y_offset;
        evalParams_t.P_XHgHn1 = params_RDC.P_XHgHn1;
        evalParams_t.beliefSpacePrecision_adv = params_RDC.beliefSpacePrecision_adv;
        evalParams_t.paramsPrecision = params_RDC.paramsPrecision;
        evalParams_t.minLikelihoodFilter = params_RDC.minLikelihoodFilter;
        evalParams_t.minPowerDemandInW = params_RDC.minPowerDemandInW;
        evalParams_t.P_H0 = params_RDC.P_H0;
        evalParams_t.storeBeliefs = storeBeliefs;

        rng(rng_id_sim,'twister');
        detectionData_UA = runSequentialBayesDetection_NC(evalParams_t, optimalControlData.modifiedSMdata, gt_data_test,false);
        [fscores_UA, tp_UA, tn_UA,precision_UA] = computeFScores(params_FDC, gt_data_test, detectionData_UA.estimatedhypStateData);

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','precision',"precision_UA","tn_UA","tp_UA","fscores_UA",'detectionData_UA')
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    risk_RLDeterministicActorCriticAgent_RD = optimalControlData.mean_correct_detection;
    reward_RLDeterministicActorCriticAgent_RD = optimalControlData.bayesian_reward;
    precision_RL_RD = precision;
    recall_RL_RD = tp;
    tn_RL_RD = tn;
    fscores_RL_RD = fscores;
    mean_PYkgY12kn1_RLDeterministicActorCriticAgent_RD = optimalControlData.mean_PYkgY12kn1;
    fprintf('\tBayesian risk: %f; Bayesian reward: %f; precision: %s; recall: %s; fscore: %s; mean_PYkgY12kn1: %s\n',...
        risk_RLDeterministicActorCriticAgent_RD, reward_RLDeterministicActorCriticAgent_RD, num2str(precision_RL_RD'), num2str(recall_RL_RD'), num2str(fscores_RL_RD'), num2str(mean_PYkgY12kn1_RLDeterministicActorCriticAgent_RD'));
    
    disp('SubOptimal infinite horizon control using TD Actor-critic RL Deterministic policy approach with FD belief space, tested with UA adv--- ');
    risk_RLDeterministicActorCriticAgent_RD_UA = mean([detectionData_UA.mean_correct_detection]);
    reward_RLDeterministicActorCriticAgent_RD_UA = mean([detectionData_UA.bayesian_reward]);
    precision_RL_RD_UA = precision_UA;
    recall_RL_RD_UA = tp_UA;
    tn_RL_RD_UA = tn_UA;
    fscores_RL_RD_UA = fscores_UA;
    fprintf('\tBayesian risk: %f; Bayesian reward: %f; precision: %s; recall: %s: fscores: %s\n',risk_RLDeterministicActorCriticAgent_RD_UA, reward_RLDeterministicActorCriticAgent_RD_UA, ...
        num2str(precision_RL_RD_UA'), num2str(recall_RL_RD_UA'),  num2str(fscores_RL_RD_UA'));
end

if(subopt_DBS_FDC_UA)
    [params_FDC, essParams, comsolParams] = getEssParams(config, params_FDC);
    fprintf('Sub-optimal infinite horizon control of UA adv using discrete FD belief space ---\n');
    params_FDC.discountFactor= config.discountFactor_DBS;
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
        [optimalControlData_UA, detectionData_AA] = simulate_UA_subopt_DBS_FDC(evalParams_t,policy,PP_data_fileFullPath, pp_data, true);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, detectionData_AA.estimatedhypStateData);
        [fscores_UA, tp_UA, tn_UA,precision_UA] = computeFScores(params_FDC, gt_data_test, optimalControlData_UA.estimatedhypStateData);
        save(fileFullPath,'optimalControlData_UA','evalParams','fscores', 'tp', 'tn','detectionData_AA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end

    avg_succ_det_UA_subopt_DBS_FDC = mean([detectionData_AA.mean_correct_detection]);
    reward_UA_subopt_DBS_FDC = mean([detectionData_AA.bayesian_reward]);
    fprintf('\tBayesian risk: %f; Bayesian reward: %f; precision: %s; recall: %s; fscore: %s\n',avg_succ_det_UA_subopt_DBS_FDC, reward_UA_subopt_DBS_FDC, num2str(precision'), num2str(tp'),num2str(fscores'));
    precision_UA_subopt_DBS_FDC = precision;
    tn_UA_subopt_DBS_FDC = tn;
    recall_UA_subopt_DBS_FDC = tp;
    fscores_UA_subopt_DBS_FDC = fscores;

    fprintf('Sub-optimal infinite horizon control using discrete FD belief space, tested with unaware adversary ---\n');
    avg_succ_det_UA_subopt_DBS_FDC_UA = mean([optimalControlData_UA.mean_correct_detection]);
    reward_UA_subopt_DBS_FDC_UA = mean([optimalControlData_UA.bayesian_reward]);
    fprintf('\tBayesian risk: %f; Bayesian reward: %f; precision: %s; recall: %s; fscore: %s\n',avg_succ_det_UA_subopt_DBS_FDC_UA, reward_UA_subopt_DBS_FDC_UA, num2str(precision_UA'), num2str(tp_UA'), num2str(fscores_UA'));
    recall_UA_subopt_DBS_FDC_UA = tp_UA;
    tn_UA_subopt_DBS_FDC_UA = tn_UA;
    precision_UA_subopt_DBS_FDC_UA = precision_UA;
    fscores_UA_subopt_DBS_FDC_UA = fscores_UA;
end