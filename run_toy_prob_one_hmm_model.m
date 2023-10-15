function [outputs] = run_toy_prob_one_hmm_model(inputs,sm_data_test,gt_data_test,h_0_idxs_test)
if(nargin==0)
        config_filename = 'toy_h2_x2_yc2_z2_k96.yaml';

    [fileDir,~,~] = fileparts(pwd);
    cache_folder_path = [fileDir filesep 'AdversarialInferenceControl_Cache' filesep 'SyntheticDataSimulations', filesep];

    numHorizons = 2000;
    P_HgHn1_p_idx =2;
    P_HgHn1_q_idx =2;
    [inputs,sm_data_test,gt_data_test,h_0_idxs_test] = get_default_inputs(config_filename, P_HgHn1_p_idx, P_HgHn1_q_idx,numHorizons);
end

NC = inputs.NC;
inst_opt_FDC = inputs.inst_opt_FDC;
opt_det_subpolicy_FDC = inputs.opt_det_subpolicy_FDC;
subopt_det_subpolicy_FDC = inputs.subopt_det_subpolicy_FDC;
subopt_DBS_FDC = inputs.subopt_DBS_FDC;
subopt_DBS_FDC_UA = inputs.subopt_DBS_FDC_UA;
best_effort_moderation = inputs.best_effort_moderation;
differential_privacy = inputs.differential_privacy;
AMDPG = inputs.AMDPG;

storeBeliefs = inputs.storeBeliefs;
storeAuxData = inputs.storeAuxData;
rng_id_sim = inputs.rng_id_sim;
in_debug_mode = inputs.in_debug_mode;
cache_folder_path = inputs.cache_folder_path;
config = inputs.config;

P_HgHn1_p_idx = inputs.P_HgHn1_p_idx;
P_HgHn1_q_idx = inputs.P_HgHn1_q_idx;
params_FDC = inputs.params_FDC;

[k_num, numHorizons] = size(sm_data_test);

showGUI = true;
useparpool = true;
UseGPU = false;

outputs = struct;
%% Optimal Bayesian detection without controller
if(NC)
    fprintf('Optimal Bayesian detection without controller --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
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
    fileNamePrefix = sprintf('%sevalData_NC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'detectionData', 'tp', 'tn','fscores','precision');
    else
        rng(rng_id_sim,'twister');
        detectionData = runSequentialBayesDetection_NC(evalParams, sm_data_test, gt_data_test, false);
        [fscores, tp, tn, precision] = computeFScores(params_FDC, gt_data_test, detectionData.estimatedhypStateData);
        save(fileFullPath,'detectionData','evalParams','fscores', 'tp', 'tn','precision')
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_NC = detectionData.bayesian_reward;
    precision_NC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_NC, num2str(precision_NC'));
    outputs.reward_NC = reward_NC;
    outputs.precision_NC = precision_NC;
end

[params_FDC, ~, ~] = getEssParams(config, params_FDC);
params_FDC_bkup = params_FDC;

%% Optimal Bayesian control
if(inst_opt_FDC)
    params_FDC = params_FDC_bkup;
    fprintf('Sub-optimal control using one step FD greedy policy --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_num','x_p_pu','y_p_pu','d_p_pu','x_offset','y_offset','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','k_num','x_p_pu','x_offset','y_offset','minPowerDemandInW','beliefSpacePrecision_adv','y_control_p_pu','y_control_num','y_control_offset',...
        'P_ZgA','P_H0','minLikelihoodFilter','u_num','s_num','C_HgHh_homogeneous','P_HgHn1','P_XgH'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_OSG_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData', 'tp', 'tn','fscores','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA");
    else
        cache_fileNamePrefix = sprintf('%spolicy_OSG_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [cache_fileName,fileExists] = findFileName(used_params,cache_fileNamePrefix,'params');

        pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        evalParams_t.in_debug_mode = in_debug_mode;
        evalParams_t.max_cache_size = inputs.max_cache_size;

        rng(rng_id_sim,'twister');
        [optimalControlData] = simulate_OSG_FDC(evalParams_t,cache_fileName, pp_data, useparpool);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, optimalControlData.estimatedhypStateData);

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

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end

    reward_inst_opt_FDC = optimalControlData.bayesian_reward;
    precision_inst_opt_FDC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_inst_opt_FDC, num2str(precision_inst_opt_FDC'));
    outputs.reward_inst_opt_FDC = reward_inst_opt_FDC;
    outputs.precision_inst_opt_FDC = precision_inst_opt_FDC;

    fprintf('\t---When tested with UA adv--- \n');
    reward_inst_opt_FDC_UA = mean([detectionData_UA.bayesian_reward]);
    precision_inst_opt_FDC_UA = precision_UA;
    outputs.reward_inst_opt_FDC_UA = reward_inst_opt_FDC_UA;
    outputs.precision_inst_opt_FDC_UA = precision_inst_opt_FDC_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_inst_opt_FDC_UA, num2str(precision_inst_opt_FDC_UA'));
end

if(opt_det_subpolicy_FDC)
    params_FDC = params_FDC_bkup;
    fprintf('Optimal Bayesian control with gamma vectors in belief space partitions --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
    params_FDC.gamma_vec_conv_threshold = inputs.gamma_vec_conv_threshold;
    params_FDC.minDet = inputs.minDet;
    params_FDC.discountFactor= inputs.discountFactor_MDP;
    params_FDC.doPruning = inputs.doPruning;
    params_FDC.max_num_EMUsubpolicies = inputs.max_num_EMUsubpolicies_opt;
    max_num_gamma_vectors = inputs.max_num_gamma_vectors;
    max_valueFnIterations = inputs.max_valueFnIterations;

    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_num','x_p_pu','y_p_pu','d_p_pu','x_offset','y_offset','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','minDet','max_num_EMUsubpolicies','doPruning','y_control_p_pu','y_control_num','y_control_offset',...
        'gamma_vec_conv_threshold','discountFactor','u_num'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    clear policy;
    pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

    PP_data_filenamePrefix = sprintf('%spp_data_det_SP_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    PP_data_fileFullPath = get_PP_data_det_SP_FDC_fileFullPath(used_params,PP_data_filenamePrefix, pp_data, useparpool);

    policy_fileNamePrefix = sprintf('%spolicy_opt_det_SP_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    policy = get_policy_opt_det_SP_FDC(used_params,max_num_gamma_vectors,max_valueFnIterations,...
        PP_data_fileFullPath,policy_fileNamePrefix, useparpool);

    if(policy.isTerminated)
        fprintf('\tPolicy terminated in %d iterations. Gamma vectors count: %.2e! [Limit: %.2e]\n',policy.iter_idx,sum(policy.num_gamma_vectors,'all'),max_num_gamma_vectors);
    end
    if(policy.isConverged)
        error('\tPolicy converged in %d iterations. Evaluation not implemented!',policy.iter_idx);
    end
end

%% Sub-optimal Bayesian control
if(subopt_det_subpolicy_FDC)
    params_FDC = params_FDC_bkup;
    fprintf('Sub-optimal Bayesian control with gamma vectors in belief space partitions --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
    params_FDC.gamma_vec_conv_threshold = inputs.gamma_vec_conv_threshold;
    params_FDC.minDet = inputs.minDet;
    params_FDC.discountFactor= inputs.discountFactor_MDP;
    params_FDC.max_num_EMUsubpolicies = inputs.max_num_EMUsubpolicies_subopt;
    max_valueFnIterations = inputs.max_valueFnIterations;

    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num','d_num','d_p_pu',...
        'P_HgA','C_HgHh_design','minDet','max_num_EMUsubpolicies','k_num','x_p_pu','x_offset','y_offset','y_control_p_pu','y_control_num','y_control_offset',...
        'minPowerDemandInW','beliefSpacePrecision_adv','P_ZgA','P_H0','minLikelihoodFilter','y_p_pu',...
        'gamma_vec_conv_threshold','discountFactor','u_num','C_HgHh_homogeneous','s_num'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    clear policy;
    pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);
    PP_data_filenamePrefix = sprintf('%spp_data_det_SP_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    PP_data_fileFullPath = get_PP_data_det_SP_FDC_fileFullPath(used_params,PP_data_filenamePrefix,pp_data, useparpool);

    policy_fileNamePrefix = sprintf('%spolicy_subopt_det_SP_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    policy = get_policy_subopt_det_SP_FDC(used_params,max_valueFnIterations,PP_data_fileFullPath,policy_fileNamePrefix,useparpool);
    if(policy.isConverged)
        fprintf('\tPolicy converged in %d iterations. max_gamma_vectors_diff: %.2e! [Threshold: %.2e]\n',policy.iter_idx,policy.max_gamma_vectors_diff,used_params.gamma_vec_conv_threshold);
    else
        error('\tPolicy not converged in %d iterations. max_gamma_vectors_diff: %.2e! [Threshold: %.2e]\n',policy.iter_idx,policy.max_gamma_vectors_diff,used_params.gamma_vec_conv_threshold);
    end

    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.policy_iter_idx = policy.iter_idx;
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_subopt_det_SP_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData', 'fscores', 'tp', 'tn','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA");
    else
        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        rng(rng_id_sim,'twister');
        optimalControlData = simulate_subopt_det_SP_FDC(evalParams_t,policy,PP_data_fileFullPath, pp_data, useparpool);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, optimalControlData.estimatedhypStateData);

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

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")

        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_subopt_det_SP_FDC = mean([optimalControlData.bayesian_reward]);
    precision_subopt_det_SP_FDC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_subopt_det_SP_FDC, num2str(precision_subopt_det_SP_FDC'));
    outputs.reward_subopt_det_SP_FDC = reward_subopt_det_SP_FDC;
    outputs.precision_subopt_det_SP_FDC = precision_subopt_det_SP_FDC;

    fprintf('\t---When tested with UA adv--- \n');
    reward_subopt_det_SP_FDC_UA = mean([detectionData_UA.bayesian_reward]);
    precision_subopt_det_SP_FDC_UA = precision_UA;
    outputs.reward_subopt_det_SP_FDC_UA = reward_subopt_det_SP_FDC_UA;
    outputs.precision_subopt_det_SP_FDC_UA = precision_subopt_det_SP_FDC_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_subopt_det_SP_FDC_UA, num2str(precision_subopt_det_SP_FDC_UA'));
end

if(subopt_DBS_FDC)
    params_FDC = params_FDC_bkup;
    fprintf('Sub-optimal infinite horizon control using discrete FD belief space --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
    params_FDC.discountFactor= inputs.discountFactor_MDP;
    params_FDC.beliefSpacePrecision_EMU_subopt_DBS = inputs.beliefSpacePrecision_EMU_subopt_DBS;
    value_iter_conv_threshold = inputs.value_iter_conv_threshold;
    max_valueFnIterations = inputs.max_valueFnIterations;

    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','k_num','x_p_pu','x_offset','y_offset','y_control_p_pu','y_control_num','y_control_offset',...
        'minPowerDemandInW','beliefSpacePrecision_adv','P_ZgA','P_H0','minLikelihoodFilter',...
        'discountFactor','beliefSpacePrecision_EMU_subopt_DBS','d_num','d_p_pu','y_p_pu','u_num','C_HgHh_homogeneous','s_num'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    clear policy;
    pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);
    PP_data_filenamePrefix = sprintf('%spp_data_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    PP_data_fileFullPath = get_PP_data_subopt_DBS_FDC_filename(used_params,PP_data_filenamePrefix,in_debug_mode, pp_data, useparpool);

    policy_fileNamePrefix = sprintf('%spolicy_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    policy = get_policy_subopt_DBS_FDC(used_params,max_valueFnIterations,value_iter_conv_threshold,...
        PP_data_fileFullPath,policy_fileNamePrefix, pp_data, useparpool);

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
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData','fscores', 'tp', 'tn','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA");
    else
        evalParams_t = evalParams;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        rng(rng_id_sim,'twister');
        optimalControlData = simulate_subopt_DBS_FDC(evalParams_t,policy,PP_data_fileFullPath, pp_data, useparpool);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, optimalControlData.estimatedhypStateData);

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

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','detectionData_UA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_subopt_DBS_FDC = optimalControlData.bayesian_reward;
    precision_subopt_DBS_FDC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_subopt_DBS_FDC, num2str(precision_subopt_DBS_FDC'));
    outputs.reward_subopt_DBS_FDC = reward_subopt_DBS_FDC;
    outputs.precision_subopt_DBS_FDC = precision_subopt_DBS_FDC;

    fprintf('\t---When tested with UA adv--- \n');
    reward_subopt_DBS_FDC_UA = mean([detectionData_UA.bayesian_reward]);
    precision_subopt_DBS_FDC_UA = precision_UA;
    outputs.reward_subopt_DBS_FDC_UA = reward_subopt_DBS_FDC_UA;
    outputs.precision_subopt_DBS_FDC_UA = precision_subopt_DBS_FDC_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_subopt_DBS_FDC_UA, num2str(precision_subopt_DBS_FDC_UA'));
end

if(subopt_DBS_FDC_UA)
    params_FDC = params_FDC_bkup;
    fprintf('Sub-optimal infinite horizon control of UA adv using discrete FD belief space --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
    params_FDC.discountFactor= inputs.discountFactor_MDP;
    params_FDC.beliefSpacePrecision_EMU_subopt_DBS = inputs.beliefSpacePrecision_EMU_subopt_DBS;
    value_iter_conv_threshold = inputs.value_iter_conv_threshold;
    max_valueFnIterations = inputs.max_valueFnIterations;

    used_params = struct;
    used_fieldnames = {'paramsPrecision','y_num','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
        'P_HgA','C_HgHh_design','k_num','x_p_pu','x_offset','y_offset','y_control_p_pu','y_control_num','y_control_offset',...
        'minPowerDemandInW','beliefSpacePrecision_adv','P_ZgA','P_H0','minLikelihoodFilter',...
        'discountFactor','beliefSpacePrecision_EMU_subopt_DBS','d_num','d_p_pu','y_p_pu','u_num','C_HgHh_homogeneous','s_num'};
    for fn = used_fieldnames
        used_params.(fn{1}) = params_FDC.(fn{1});
    end

    clear policy;
    pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);
    PP_data_filenamePrefix = sprintf('%spp_data_UA_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    PP_data_fileFullPath = get_PP_data_UA_subopt_DBS_FDC_filename(used_params,PP_data_filenamePrefix,in_debug_mode, pp_data, useparpool);

    policy_fileNamePrefix = sprintf('%spolicy_UA_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
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
    fileNamePrefix = sprintf('%sevalData_UA_subopt_DBS_FDC_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
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
        [optimalControlData_UA, detectionData_AA] = simulate_UA_subopt_DBS_FDC(evalParams_t,policy,PP_data_fileFullPath, pp_data, useparpool);
        [fscores, tp, tn,precision] = computeFScores(params_FDC, gt_data_test, detectionData_AA.estimatedhypStateData);
        [fscores_UA, tp_UA, tn_UA,precision_UA] = computeFScores(params_FDC, gt_data_test, optimalControlData_UA.estimatedhypStateData);
        save(fileFullPath,'optimalControlData_UA','evalParams','fscores', 'tp', 'tn','detectionData_AA','precision_UA',"precision",'tn_UA',"tp_UA","fscores_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end
    reward_UA_subopt_DBS_FDC = mean([detectionData_AA.bayesian_reward]);
    precision_UA_subopt_DBS_FDC = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_UA_subopt_DBS_FDC, num2str(precision_UA_subopt_DBS_FDC'));
    outputs.reward_UA_subopt_DBS_FDC = reward_UA_subopt_DBS_FDC;
    outputs.precision_UA_subopt_DBS_FDC = precision_UA_subopt_DBS_FDC;

    fprintf('\t---When tested with UA adv--- \n');
    reward_UA_subopt_DBS_FDC_UA = mean([optimalControlData_UA.bayesian_reward]);
    precision_UA_subopt_DBS_FDC_UA = precision_UA;
    outputs.reward_UA_subopt_DBS_FDC_UA = reward_UA_subopt_DBS_FDC_UA;
    outputs.precision_UA_subopt_DBS_FDC_UA = precision_UA_subopt_DBS_FDC_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_UA_subopt_DBS_FDC_UA, num2str(precision_UA_subopt_DBS_FDC_UA'));
end

%% Passive controller with best effort moderation strategy
if(best_effort_moderation)
    params_FDC = params_FDC_bkup;
    fprintf('Optimal Bayesian detection with BEM controller --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
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
    fileNamePrefix = sprintf('%sevalData_BEM_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'bemControlData','tp', 'tn','fscores','precision');
    else
        pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

        pp_data_fileNamePrefix = sprintf('%sppdata_BEM_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [pp_data_BEM] = get_ppdata_BEM_FD(used_params,pp_data_fileNamePrefix, pp_data);

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
    precision_BEM = fscores;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_BEM, num2str(precision_BEM'));
    outputs.reward_BEM = reward_BEM;
    outputs.precision_BEM = precision_BEM;
end

%% Passive controller with differential privacy
if(differential_privacy)
    differential_privacy_epsilon_ranges = 10.^(-3:3);
    epsilon_sweep_results = zeros(length(differential_privacy_epsilon_ranges),3);
    fprintf('Optimal Bayesian detection with differential_privacy controller --- %d_%d\n',P_HgHn1_p_idx,P_HgHn1_q_idx);
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
        fileNamePrefix = sprintf('%sevalData_differential_privacy_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
        [~,filename] = fileparts(fileFullPath);
        if(fileExists)
            load(fileFullPath,'differential_privacy_ControlData','tp', 'tn','fscores','precision');
        else
            pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
            [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

            pp_data_fileNamePrefix = sprintf('%sppdata_differential_privacy_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
            [pp_data_differential_privacy] = get_ppdata_differential_privacy_FD(used_params,pp_data_fileNamePrefix, pp_data);

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
        precision_differential_privacy = fscores;
        epsilon_sweep_results(idx,1) = differential_privacy_epsilon;
        epsilon_sweep_results(idx,2) = reward_differential_privacy;
        epsilon_sweep_results(idx,3) = precision_differential_privacy;
    end
    fprintf('\tMin Bayesian reward: %f; Max Bayesian reward: %f\n', min(epsilon_sweep_results(:,2)), max(epsilon_sweep_results(:,2)));
    fprintf('\tMin precision: %f; Max precision: %f\n', min(epsilon_sweep_results(:,3)), max(epsilon_sweep_results(:,3)));
    outputs.min_reward_DP = min(epsilon_sweep_results(:,2));
    outputs.max_reward_DP = max(epsilon_sweep_results(:,2));
    outputs.min_precision_DP = min(epsilon_sweep_results(:,3));
    outputs.max_precision_DP = max(epsilon_sweep_results(:,3));
end

%% AMDPG Reinforcement learning based controller
if(AMDPG)
    disp('SubOptimal infinite horizon control using Actor-critic RL control of FD adversary--- ');
    params_FDC = params_FDC_bkup;
    params_FDC.numTrainHorizons= inputs.numTrainHorizons;
    params_FDC.discountFactor = inputs.discountFactor_RL;
    params_FDC.learning_rate_Ac = inputs.learning_rate_Ac;
    params_FDC.learning_rate_C = inputs.learning_rate_C;
    params_FDC.TargetSmoothFactor_C = inputs.TargetSmoothFactor_C;
    params_FDC.MiniBatchSize = inputs.MiniBatchSize;
    params_FDC.ReplayBufferLength = inputs.ReplayBufferLength;
    params_FDC.TargetSmoothFactor_Ac = inputs.TargetSmoothFactor_Ac;
    params_FDC.actor_net_hidden_layers = inputs.actor_net_hidden_layers;
    params_FDC.actor_net_hidden_layer_neurons_ratio_obs = inputs.actor_net_hidden_layer_neurons_ratio_obs;
    params_FDC.actor_net_hidden_layer_neurons_ratio_act = inputs.actor_net_hidden_layer_neurons_ratio_act;
    params_FDC.critic_net_hidden_layers = inputs.critic_net_hidden_layers;
    params_FDC.critic_net_hidden_layer_neurons_ratio_obs = inputs.critic_net_hidden_layer_neurons_ratio_obs;
    params_FDC.critic_net_hidden_layer_neurons_ratio_act = inputs.critic_net_hidden_layer_neurons_ratio_act;
    params_FDC.y_num_for_exploration = inputs.y_num_for_exploration;
    params_FDC.num_rand_adv_strats_for_exploration = inputs.num_rand_adv_strats_for_exploration;
    params_FDC.logistic_param_limit = inputs.logistic_param_limit;
    params_FDC.GradientDecayFactor_Ac = inputs.GradientDecayFactor_Ac;
    params_FDC.SquaredGradientDecayFactor_Ac = inputs.SquaredGradientDecayFactor_Ac;
    params_FDC.Epsilon_adam_Ac = inputs.Epsilon_adam_Ac;
    params_FDC.GradientDecayFactor_C = inputs.GradientDecayFactor_C;
    params_FDC.SquaredGradientDecayFactor_C = inputs.SquaredGradientDecayFactor_C;
    params_FDC.Epsilon_adam_C = inputs.Epsilon_adam_C;
    params_FDC.noise_epsilon = inputs.noise_epsilon;
    params_FDC.with_bem_initialized_actor = inputs.with_bem_initialized_actor;
    params_FDC.noise_sd = inputs.noise_sd;
    params_FDC.with_controller_reward = inputs.with_controller_reward;
    params_FDC.with_mean_reward = inputs.with_mean_reward;
    params_FDC.with_a_nncells = inputs.with_a_nncells;
    params_FDC.exploration_epsilon = inputs.exploration_epsilon;

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

    evalParams = struct;
    evalParams.params = used_params;
    evalParams.numHorizons = numHorizons;
    evalParams.storeAuxData = storeAuxData;
    evalParams.storeBeliefs = storeBeliefs;
    evalParams.numTrainHorizons = used_params.numTrainHorizons;
    clear optimalControlData;
    fileNamePrefix = sprintf('%sevalData_RLDeterministicActorCriticAgent_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(evalParams,fileNamePrefix,'evalParams');
    [~,filename] = fileparts(fileFullPath);
    if(fileExists)
        fprintf('\t\tEvaluation skipped. Data found in: %s\n',filename);
        load(fileFullPath,'optimalControlData','fscores', 'tp', 'tn','detectionData_UA',"tn_UA","tp_UA","fscores_UA",'precision',"precision_UA");
    else
        clear rl_agent;
        pp_data_fileNamePrefix = sprintf('%sppdata_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [pp_data] = get_ppdata_FD(used_params,pp_data_fileNamePrefix);

        pp_data_fileNamePrefix = sprintf('%sppdata_BEM_FD_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [pp_data_BEM] = get_ppdata_BEM_FD(used_params,pp_data_fileNamePrefix, pp_data);
        BEM_initialized_actor_fileNamePrefix = sprintf('%sBEM_initialized_actor_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        
        policy_fileNamePrefix = sprintf('%spolicy_RL_DeterministicActorCriticAgent_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [rl_agent, policy_fileFullPath] = get_DeterministicActorCriticAgent(used_params,policy_fileNamePrefix, showGUI, UseGPU, pp_data_BEM, BEM_initialized_actor_fileNamePrefix, false);  %#ok<*ASGLU>
        plot_RL_training_data(policy_fileFullPath, inputs.P_HgHn1_p, inputs.P_HgHn1_q);

        evalParams_t = evalParams;
        evalParams_t.params.s_num = params_FDC.s_num;
        evalParams_t.sm_data = sm_data_test;
        evalParams_t.gt_data = gt_data_test;
        evalParams_t.h_0_idxs = h_0_idxs_test;
        evalParams_t.rl_agent = rl_agent;
        rng(rng_id_sim,'twister');
        optimalControlData = simulate_ActorCriticAgent(evalParams_t, pp_data, useparpool);
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
        [fscores_UA, tp_UA, tn_UA, precision_UA] = computeFScores(params_FDC, gt_data_test, detectionData_UA.estimatedhypStateData);

        save(fileFullPath,'optimalControlData','evalParams','fscores', 'tp', 'tn','detectionData_UA',"tn_UA","tp_UA","fscores_UA",'precision',"precision_UA")
        fprintf('\t\tEvaluation complete. Data saved in: %s\n',filename);
    end

    reward_AMDPG = optimalControlData.bayesian_reward;
    precision_AMDPG = precision;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_AMDPG, num2str(precision_AMDPG'));

    fprintf('\t---When tested with UA adv--- \n');
    reward_AMDPG_UA = mean([detectionData_UA.bayesian_reward]);
    precision_AMDPG_UA = precision_UA;
    fprintf('\tBayesian reward: %f; precision: %s\n', reward_AMDPG_UA, num2str(precision_AMDPG_UA'));

    outputs.reward_AMDPG = reward_AMDPG;
    outputs.precision_AMDPG = precision_AMDPG;
    outputs.reward_AMDPG_UA = reward_AMDPG_UA;
    outputs.precision_AMDPG_UA = precision_AMDPG_UA;
end

%% Supporting functions
    function [eval_inputs,sm_data_test,gt_data_test,h_0_idxs] = get_default_inputs(config_filename, P_HgHn1_p_idx, P_HgHn1_q_idx, numHorizons)
        rng_id_sim = 1;
        simStartup(0,rng_id_sim);
        dbstop if error

        in_debug_mode = false;
        storeAuxData = false;
        storeBeliefs = false;

        P_HgHn1_elem_range = 0:0.2:1;

        eval_inputs = struct;
        config = yaml.loadFile(config_filename);
        config.cache_folder_path = cache_folder_path;

        h_num = config.hypothesisStatesNum;
        P_H0 = ones(h_num,1)/h_num;
        additional_data.P_H0 = P_H0;

        NC = 1;
        inst_opt_FDC = config.inst_opt_FDC;
        opt_det_subpolicy_FDC= config.opt_det_subpolicy_FDC;
        subopt_det_subpolicy_FDC = config.subopt_det_subpolicy_FDC;
        subopt_DBS_FDC = config.subopt_DBS_FDC;
        subopt_DBS_FDC_UA = config.subopt_DBS_FDC_UA;
        best_effort_moderation = config.best_effort_moderation;
        differential_privacy = config.differential_privacy;
        AMDPG = config.AMDPG;

        eval_inputs.P_HgHn1_p_idx = P_HgHn1_p_idx;
        eval_inputs.P_HgHn1_q_idx = P_HgHn1_q_idx;

        config.P_HgHn1_p = P_HgHn1_elem_range(P_HgHn1_p_idx);
        config.P_HgHn1_q = P_HgHn1_elem_range(P_HgHn1_q_idx);
        config.P_HgHn1_p_idx = P_HgHn1_p_idx;
        config.P_HgHn1_q_idx = P_HgHn1_q_idx;
        config.rng_id_sim = rng_id_sim;

        [params_FDC,config] = initParams(config,additional_data,true);

        eval_inputs.P_HgHn1_p = config.P_HgHn1_p;
        eval_inputs.P_HgHn1_q = config.P_HgHn1_q;
        eval_inputs.params_FDC = params_FDC;

        %% Evaluation params
        if(inst_opt_FDC)
            eval_inputs.max_cache_size = config.max_cache_size;
        end
        if(opt_det_subpolicy_FDC)
            eval_inputs.doPruning = config.doPruning;
            eval_inputs.max_num_gamma_vectors =  config.max_num_gamma_vectors;
            eval_inputs.max_num_EMUsubpolicies_opt = config.max_num_EMUsubpolicies_opt;
        end
        if(opt_det_subpolicy_FDC||subopt_det_subpolicy_FDC)
            eval_inputs.minDet = config.minDet;
            eval_inputs.gamma_vec_conv_threshold = config.gamma_vec_conv_threshold;
        end
        if(subopt_det_subpolicy_FDC)
            eval_inputs.max_num_EMUsubpolicies_subopt =  config.max_num_EMUsubpolicies_subopt;
        end
        if(opt_det_subpolicy_FDC||subopt_det_subpolicy_FDC||subopt_DBS_FDC || subopt_DBS_FDC_UA)
            eval_inputs.max_valueFnIterations = config.max_valueFnIterations;
        end
        if(opt_det_subpolicy_FDC||subopt_det_subpolicy_FDC||subopt_DBS_FDC || subopt_DBS_FDC_UA)
            eval_inputs.discountFactor_MDP = config.discountFactor_MDP;
        end
        if(subopt_DBS_FDC || subopt_DBS_FDC_UA)
            eval_inputs.beliefSpacePrecision_EMU_subopt_DBS = config.beliefSpacePrecision_EMU_subopt_DBS;
            eval_inputs.value_iter_conv_threshold = config.value_iter_conv_threshold;
        end
        if(AMDPG)
            eval_inputs.numTrainHorizons= config.numTrainHorizons;
            eval_inputs.discountFactor_RL = config.discountFactor_RL;
            eval_inputs.learning_rate_Ac = config.learning_rate_Ac;
            eval_inputs.learning_rate_C = config.learning_rate_C;
            eval_inputs.TargetSmoothFactor_C = config.TargetSmoothFactor_C;
            eval_inputs.MiniBatchSize = config.MiniBatchSize;
            eval_inputs.ReplayBufferLength = config.ReplayBufferLength;
            eval_inputs.TargetSmoothFactor_Ac = config.TargetSmoothFactor_Ac;
            eval_inputs.actor_net_hidden_layers = config.actor_net_hidden_layers;
            eval_inputs.actor_net_hidden_layer_neurons_ratio_obs = config.actor_net_hidden_layer_neurons_ratio_obs;
            eval_inputs.actor_net_hidden_layer_neurons_ratio_act = config.actor_net_hidden_layer_neurons_ratio_act;
            eval_inputs.critic_net_hidden_layers = config.critic_net_hidden_layers;
            eval_inputs.critic_net_hidden_layer_neurons_ratio_obs = config.critic_net_hidden_layer_neurons_ratio_obs;
            eval_inputs.critic_net_hidden_layer_neurons_ratio_act = config.critic_net_hidden_layer_neurons_ratio_act;
            eval_inputs.y_num_for_exploration = config.y_num_for_exploration;
            eval_inputs.num_rand_adv_strats_for_exploration = config.num_rand_adv_strats_for_exploration;
            eval_inputs.logistic_param_limit = config.logistic_param_limit;
            eval_inputs.GradientDecayFactor_Ac = config.GradientDecayFactor_Ac;
            eval_inputs.SquaredGradientDecayFactor_Ac = config.SquaredGradientDecayFactor_Ac;
            eval_inputs.Epsilon_adam_Ac = config.Epsilon_adam_Ac;
            eval_inputs.GradientDecayFactor_C = config.GradientDecayFactor_C;
            eval_inputs.SquaredGradientDecayFactor_C = config.SquaredGradientDecayFactor_C;
            eval_inputs.Epsilon_adam_C = config.Epsilon_adam_C;
            eval_inputs.noise_epsilon = config.noise_epsilon;
            eval_inputs.with_bem_initialized_actor = config.with_bem_initialized_actor;
            eval_inputs.noise_sd = config.noise_sd;
            eval_inputs.with_controller_reward = config.with_controller_reward;
            eval_inputs.with_mean_reward = config.with_mean_reward;
            eval_inputs.with_a_nncells = config.with_a_nncells;
            eval_inputs.exploration_epsilon = config.exploration_epsilon;
        end

        eval_inputs.NC = NC;
        eval_inputs.inst_opt_FDC = inst_opt_FDC;
        eval_inputs.opt_det_subpolicy_FDC = opt_det_subpolicy_FDC;
        eval_inputs.subopt_det_subpolicy_FDC = subopt_det_subpolicy_FDC;
        eval_inputs.subopt_DBS_FDC = subopt_DBS_FDC;
        eval_inputs.subopt_DBS_FDC_UA = subopt_DBS_FDC_UA;
        eval_inputs.best_effort_moderation = best_effort_moderation;
        eval_inputs.differential_privacy = differential_privacy;
        eval_inputs.AMDPG = AMDPG;

        eval_inputs.storeBeliefs = storeBeliefs;
        eval_inputs.storeAuxData = storeAuxData;
        eval_inputs.rng_id_sim = rng_id_sim;
        eval_inputs.in_debug_mode = in_debug_mode;
        eval_inputs.cache_folder_path = cache_folder_path;
        eval_inputs.config = config;

        %% Generate data
        genDataParams = struct;
        genDataParams.k_num = params_FDC.k_num;
        genDataParams.h_num = params_FDC.h_num;
        genDataParams.x_p_pu = params_FDC.x_p_pu;
        genDataParams.x_offset = params_FDC.x_offset;
        genDataParams.P_XgH = params_FDC.P_XgH;
        genDataParams.P_HgHn1 = params_FDC.P_HgHn1;
        genDataParams.numHorizons = numHorizons;
        genDataParams.P_H0 = params_FDC.P_H0;

        fileNamePrefix = sprintf('%ssyntheticData_%d_%d_',cache_folder_path,P_HgHn1_p_idx,P_HgHn1_q_idx);
        [fileFullPath,fileExists] = findFileName(genDataParams,fileNamePrefix,'genDataParams');
        if(fileExists)
            load(fileFullPath,'sm_data_test','gt_data_test','h_0_idxs');
        else
            rng(rng_id_sim,'twister');
            [sm_data_test,gt_data_test,~,h_0_idxs] = generateSyntheticData(genDataParams);
            save(fileFullPath,'sm_data_test','gt_data_test','h_0_idxs','genDataParams')
        end
    end
end