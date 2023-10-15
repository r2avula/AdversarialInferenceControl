clear;
rng_id_sim = 1;
[fileDir,~,~] = fileparts(pwd);
cache_folder_path = [fileDir filesep 'AdversarialInferenceControl_Cache' filesep 'SyntheticDataSimulations', filesep];
simStartup(0,rng_id_sim);
dbstop if error

print_fig = 1;
in_debug_mode = false;

storeAuxData = false;
storeBeliefs = false;

numHorizons = 2000;
P_HgHn1_elem_range = 0:0.2:1;
P_HgHn1_elem_num = length(P_HgHn1_elem_range);

config_filename = 'toy_h2_x2_yc2_z2_k96.yaml';

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

eval_inputs = struct;
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

valid_model_flag = false(P_HgHn1_elem_num,P_HgHn1_elem_num);

reward_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_BEM = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
min_reward_DP = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
max_reward_DP = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_AMDPG = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_AMDPG_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

precision_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_BEM = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
min_precision_DP = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
max_precision_DP = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_DDPG = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_DDPG_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_AMDPG = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_AMDPG_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

config_bkup = config;
for P_HgHn1_p_idx = 2:P_HgHn1_elem_num-1 % lambda_1
% for P_HgHn1_p_idx = [2,3,4,5] % lambda_1
%     for P_HgHn1_q_idx = 2:5 % lambda_2
    for P_HgHn1_q_idx = 2:P_HgHn1_elem_num-1 % lambda_2
        %% params Initialization
        eval_inputs.P_HgHn1_p_idx = P_HgHn1_p_idx;
        eval_inputs.P_HgHn1_q_idx = P_HgHn1_q_idx;

        config = config_bkup;
        config.P_HgHn1_p = P_HgHn1_elem_range(P_HgHn1_p_idx);
        config.P_HgHn1_q = P_HgHn1_elem_range(P_HgHn1_q_idx);
        config.P_HgHn1_p_idx = P_HgHn1_p_idx;
        config.P_HgHn1_q_idx = P_HgHn1_q_idx;
        config.rng_id_sim = rng_id_sim;

        [params_FDC,config] = initParams(config, additional_data, true);

        eval_inputs.P_HgHn1_p = config.P_HgHn1_p;
        eval_inputs.P_HgHn1_q = config.P_HgHn1_q;
        eval_inputs.params_FDC = params_FDC;
        eval_inputs.config = config;

        %% validate HMM model
        if(any(sum(params_FDC.P_HgHn1,2)==0))
            continue;
        end
        mc = dtmc(params_FDC.P_HgHn1);
        xFix = asymptotics(mc);
        if(size(xFix,1)>1)
            continue;
        end
        valid_model_flag(P_HgHn1_p_idx,P_HgHn1_q_idx) = true;

        %% generate data
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

        %% evaluate HMM with genrated data
        [eval_outputs] = run_toy_prob_one_hmm_model(eval_inputs,sm_data_test,gt_data_test,h_0_idxs);

        %% save results
        if(NC)
            reward_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_NC;
            precision_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_NC;
        end

        if(best_effort_moderation)
            reward_BEM(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_BEM;
            precision_BEM(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_BEM;
        end

        if(inst_opt_FDC)
            reward_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_inst_opt_FDC;
            precision_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_inst_opt_FDC;
            reward_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_inst_opt_FDC_UA;
            precision_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_inst_opt_FDC_UA;
        end

        if(subopt_det_subpolicy_FDC)
            reward_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_det_SP_FDC;
            precision_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_det_SP_FDC;
            reward_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_det_SP_FDC_UA;
            precision_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_det_SP_FDC_UA;
        end

        if(subopt_DBS_FDC)
            reward_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_DBS_FDC;
            precision_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_DBS_FDC;
            reward_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_DBS_FDC_UA;
            precision_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_DBS_FDC_UA;
        end

        if(subopt_DBS_FDC_UA)
            reward_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_UA_subopt_DBS_FDC;
            precision_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_UA_subopt_DBS_FDC;
            reward_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_UA_subopt_DBS_FDC_UA;
            precision_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_UA_subopt_DBS_FDC_UA;
        end

        if(best_effort_moderation)
            reward_BEM(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_BEM;
            precision_BEM(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_BEM;
        end

        if(differential_privacy)
            min_reward_DP(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.min_reward_DP;
            max_reward_DP(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.max_reward_DP;
            min_precision_DP(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.min_precision_DP;
            max_precision_DP(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.max_precision_DP;
        end

        if(AMDPG)
            reward_AMDPG(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_AMDPG;
            reward_AMDPG_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_AMDPG_UA;
            precision_AMDPG(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_AMDPG;
            precision_AMDPG_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_AMDPG_UA;
        end
    end
end

if(print_fig && usejava('desktop'))
    plot_inputs = struct;
    plot_inputs.NC = NC;
    plot_inputs.inst_opt_FDC = inst_opt_FDC;
    plot_inputs.opt_det_subpolicy_FDC = opt_det_subpolicy_FDC;
    plot_inputs.subopt_det_subpolicy_FDC = subopt_det_subpolicy_FDC;
    plot_inputs.subopt_DBS_FDC = subopt_DBS_FDC;
    plot_inputs.subopt_DBS_FDC_UA = subopt_DBS_FDC_UA;
    plot_inputs.best_effort_moderation = best_effort_moderation;
    plot_inputs.differential_privacy = differential_privacy;
    plot_inputs.AMDPG = AMDPG;
    
    plot_inputs.valid_model_flag = valid_model_flag;
    plot_inputs.P_HgHn1_elem_range = P_HgHn1_elem_range;

    plot_inputs.reward_NC = reward_NC;
    plot_inputs.reward_inst_opt_FDC = reward_inst_opt_FDC;
    plot_inputs.reward_inst_opt_FDC_UA = reward_inst_opt_FDC_UA;
    plot_inputs.reward_subopt_det_SP_FDC = reward_subopt_det_SP_FDC;
    plot_inputs.reward_subopt_det_SP_FDC_UA = reward_subopt_det_SP_FDC_UA;
    plot_inputs.reward_subopt_DBS_FDC = reward_subopt_DBS_FDC;
    plot_inputs.reward_subopt_DBS_FDC_UA = reward_subopt_DBS_FDC_UA;
    plot_inputs.reward_UA_subopt_DBS_FDC = reward_UA_subopt_DBS_FDC;
    plot_inputs.reward_UA_subopt_DBS_FDC_UA = reward_UA_subopt_DBS_FDC_UA;
    plot_inputs.reward_BEM = reward_BEM;
    plot_inputs.min_reward_DP = min_reward_DP;
    plot_inputs.max_reward_DP = max_reward_DP;
    plot_inputs.reward_AMDPG = reward_AMDPG;
    plot_inputs.reward_AMDPG_UA = reward_AMDPG_UA;

    plot_inputs.precision_NC = precision_NC;
    plot_inputs.precision_inst_opt_FDC = precision_inst_opt_FDC;
    plot_inputs.precision_inst_opt_FDC_UA = precision_inst_opt_FDC_UA;
    plot_inputs.precision_subopt_det_SP_FDC = precision_subopt_det_SP_FDC;
    plot_inputs.precision_subopt_det_SP_FDC_UA = precision_subopt_det_SP_FDC_UA;
    plot_inputs.precision_subopt_DBS_FDC = precision_subopt_DBS_FDC;
    plot_inputs.precision_subopt_DBS_FDC_UA = precision_subopt_DBS_FDC_UA;
    plot_inputs.precision_UA_subopt_DBS_FDC = precision_UA_subopt_DBS_FDC;
    plot_inputs.precision_UA_subopt_DBS_FDC_UA = precision_UA_subopt_DBS_FDC_UA;
    plot_inputs.precision_BEM = precision_BEM;
    plot_inputs.min_precision_DP = min_precision_DP;
    plot_inputs.max_precision_DP = max_precision_DP;
    plot_inputs.precision_AMDPG = precision_AMDPG;
    plot_inputs.precision_AMDPG_UA = precision_AMDPG_UA;

    plot_fig_sweep_hmm_models(plot_inputs);
    %     set(gcf,'SelectionHighlight','off')
    %     exportgraphics(gcf,'mean_correct_detection_comparision_binary_toy.pdf','ContentType','vector');
end