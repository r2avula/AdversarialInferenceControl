clear;
rng_id_sim = 1;
[fileDir,~,~] = fileparts(pwd);
cache_folder_path = [fileDir filesep 'AdversarialInferenceControl_Cache' filesep 'SyntheticDataSimulations', filesep];
simStartup(0,rng_id_sim);
dbstop if error

print_fig = 1;
in_debug_mode = true;

storeAuxData = false;
storeBeliefs = false;

numHorizons = 2000;
P_HgHn1_elem_range = 0:0.2:1;
P_HgHn1_elem_num = length(P_HgHn1_elem_range);

config_filename = 'h2_x2_yc2_z2_l1_k96.yaml';

config = yaml.loadFile(config_filename);
config.cache_folder_path = cache_folder_path;

h_num = config.hypothesisStatesNum;
P_H0 = ones(h_num,1)/h_num;
additional_data.P_H0 = P_H0;

NC= config.NC;
opt_det_subpolicy_FDC= config.opt_det_subpolicy_FDC;
inst_opt_FDC = config.inst_opt_FDC;
subopt_det_subpolicy_FDC = config.subopt_det_subpolicy_FDC;
subopt_DBS_FDC = config.subopt_DBS_FDC;
RL_DeterministicActorCriticAgent = config.RL_DeterministicActorCriticAgent;
if isfield(config, 'RL_DeterministicActorCriticAgent_RD')
    RL_DeterministicActorCriticAgent_RD = config.RL_DeterministicActorCriticAgent_RD;
else
    RL_DeterministicActorCriticAgent_RD = 1;
end
subopt_DBS_FDC_UA = config.subopt_DBS_FDC_UA;

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
    eval_inputs.discountFactor = config.discountFactor;
end
if(subopt_DBS_FDC || subopt_DBS_FDC_UA)
    eval_inputs.beliefSpacePrecision_EMU_subopt_DBS = config.beliefSpacePrecision_EMU_subopt_DBS;
    eval_inputs.value_iter_conv_threshold = config.value_iter_conv_threshold;
end
if(RL_DeterministicActorCriticAgent)
    eval_inputs.numTrainHorizons = config.numTrainHorizons;
    eval_inputs.discountFactor_rl = config.discountFactor_rl;
    eval_inputs.learning_rate_Ac = config.learning_rate_Ac;
    eval_inputs.learning_rate_C = config.learning_rate_C;
    eval_inputs.exploration_epsilon = config.exploration_epsilon;
    eval_inputs.TargetSmoothFactor_C = config.TargetSmoothFactor_C;
    eval_inputs.MiniBatchSize = config.MiniBatchSize;
    eval_inputs.InMemoryUpdateInterval = config.InMemoryUpdateInterval;
    eval_inputs.ReplayBufferLength = config.ReplayBufferLength;
    eval_inputs.TargetSmoothFactor_Ac = config.TargetSmoothFactor_Ac;
    eval_inputs.actor_net_hidden_layers = config.actor_net_hidden_layers;
    eval_inputs.actor_net_hidden_layer_neurons_ratio = config.actor_net_hidden_layer_neurons_ratio;
    eval_inputs.critic_net_hidden_layers = config.critic_net_hidden_layers;
    eval_inputs.critic_net_hidden_layer_neurons_ratio = config.critic_net_hidden_layer_neurons_ratio;
    eval_inputs.penalty_factor = config.penalty_factor;

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
end
if(RL_DeterministicActorCriticAgent_RD)
    eval_inputs.numTrainHorizons = config.numTrainHorizons;
    eval_inputs.discountFactor_rl = config.discountFactor_rl;
    eval_inputs.learning_rate_Ac = config.learning_rate_Ac;
    eval_inputs.learning_rate_C = config.learning_rate_C;
    eval_inputs.exploration_epsilon = config.exploration_epsilon;
    eval_inputs.TargetSmoothFactor_C = config.TargetSmoothFactor_C;
    eval_inputs.MiniBatchSize = config.MiniBatchSize;
    eval_inputs.InMemoryUpdateInterval = config.InMemoryUpdateInterval;
    eval_inputs.ReplayBufferLength = config.ReplayBufferLength;
    eval_inputs.TargetSmoothFactor_Ac = config.TargetSmoothFactor_Ac;
    eval_inputs.actor_net_hidden_layers = config.actor_net_hidden_layers;
    eval_inputs.actor_net_hidden_layer_neurons_ratio = config.actor_net_hidden_layer_neurons_ratio;
    eval_inputs.critic_net_hidden_layers = config.critic_net_hidden_layers;
    eval_inputs.penalty_factor = config.penalty_factor;

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

    eval_inputs.critic_net_hidden_layer_neurons_ratio_obs = config.critic_net_hidden_layer_neurons_ratio_obs;
    eval_inputs.critic_net_hidden_layer_neurons_ratio_act = config.critic_net_hidden_layer_neurons_ratio_act;
end

eval_inputs.NC = NC;
eval_inputs.inst_opt_FDC = inst_opt_FDC;
eval_inputs.opt_det_subpolicy_FDC = opt_det_subpolicy_FDC;
eval_inputs.subopt_det_subpolicy_FDC = subopt_det_subpolicy_FDC;
eval_inputs.subopt_DBS_FDC = subopt_DBS_FDC;
eval_inputs.RL_DeterministicActorCriticAgent = RL_DeterministicActorCriticAgent;
eval_inputs.RL_DeterministicActorCriticAgent_RD = RL_DeterministicActorCriticAgent_RD;
eval_inputs.subopt_DBS_FDC_UA = subopt_DBS_FDC_UA;

eval_inputs.storeBeliefs = storeBeliefs;
eval_inputs.storeAuxData = storeAuxData;
eval_inputs.rng_id_sim = rng_id_sim;
eval_inputs.in_debug_mode = in_debug_mode;
eval_inputs.cache_folder_path = cache_folder_path;

valid_model_flag = false(P_HgHn1_elem_num,P_HgHn1_elem_num);

mean_correct_detection_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_correct_detection_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

reward_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
reward_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

fscores_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
fscores_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

precision_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
precision_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

recall_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
recall_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

mean_PYkgY12kn1_NC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_inst_opt_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_inst_opt_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_subopt_det_SP_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_subopt_det_SP_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_UA_subopt_DBS_FDC = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_UA_subopt_DBS_FDC_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_RLDeterministicActorCriticAgent = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);
mean_PYkgY12kn1_RLDeterministicActorCriticAgent_UA = nan(P_HgHn1_elem_num,P_HgHn1_elem_num);

% for P_HgHn1_p_idx = 2:P_HgHn1_elem_num-1 % lambda_1
for P_HgHn1_p_idx = 2:5 % lambda_1
%     for P_HgHn1_q_idx = 2:5 % lambda_2
    for P_HgHn1_q_idx = 2:P_HgHn1_elem_num-1 % lambda_2
        %% params Initialization
        eval_inputs.P_HgHn1_p_idx = P_HgHn1_p_idx;
        eval_inputs.P_HgHn1_q_idx = P_HgHn1_q_idx;

        config.P_HgHn1_p = P_HgHn1_elem_range(P_HgHn1_p_idx);
        config.P_HgHn1_q = P_HgHn1_elem_range(P_HgHn1_q_idx);
        config.P_HgHn1_p_idx = P_HgHn1_p_idx;
        config.P_HgHn1_q_idx = P_HgHn1_q_idx;
        config.rng_id_sim = rng_id_sim;
        eval_inputs.P_HgHn1_p = config.P_HgHn1_p;
        eval_inputs.P_HgHn1_q = config.P_HgHn1_q;

        [params_FDC,params_RDC] = initParams(config, additional_data, true);

        if params_RDC.l_num == 1
            eval_inputs.RL_DeterministicActorCriticAgent_RD = 0;
            eval_inputs.DDPG_RD = 0;            
        end

        eval_inputs.params_FDC = params_FDC;
        eval_inputs.params_RDC = params_RDC;

        %% validate HMM model
        if(any(sum(params_FDC.P_HgHn1,2)==0))
            continue;
            %     error('here')
        end
        mc = dtmc(params_FDC.P_HgHn1);
        xFix = asymptotics(mc);
        if(size(xFix,1)>1)
            continue;
            %     error('here')
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
            mean_correct_detection_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_NC;
            reward_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_NC;
            precision_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_NC;
            recall_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_NC;
            fscores_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_NC;
            mean_PYkgY12kn1_NC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_PYkgY12kn1_NC;
        end

        if(inst_opt_FDC)
            mean_correct_detection_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_inst_opt_FDC;
            reward_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_inst_opt_FDC;
            precision_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_inst_opt_FDC;
            recall_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_inst_opt_FDC;
            fscores_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_inst_opt_FDC;
            mean_PYkgY12kn1_inst_opt_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_PYkgY12kn1_inst_opt_FDC;

            mean_correct_detection_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_inst_opt_FDC_UA;
            reward_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_inst_opt_FDC_UA;
            precision_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_inst_opt_FDC_UA;
            recall_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_inst_opt_FDC_UA;
            fscores_inst_opt_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_inst_opt_FDC_UA;
        end

        if(subopt_det_subpolicy_FDC)
            mean_correct_detection_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_subopt_det_SP_FDC;
            reward_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_det_SP_FDC;
            precision_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_det_SP_FDC;
            recall_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_subopt_det_SP_FDC;
            fscores_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_subopt_det_SP_FDC;
            mean_PYkgY12kn1_subopt_det_SP_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_PYkgY12kn1_subopt_det_SP_FDC;

            mean_correct_detection_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_subopt_det_SP_FDC_UA;
            reward_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_det_SP_FDC_UA;
            precision_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_det_SP_FDC_UA;
            recall_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_subopt_det_SP_FDC_UA;
            fscores_subopt_det_SP_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_subopt_det_SP_FDC_UA;
        end

        if(subopt_DBS_FDC)
            mean_correct_detection_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_subopt_DBS_FDC;
            reward_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_DBS_FDC;
            precision_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_DBS_FDC;
            recall_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_subopt_DBS_FDC;
            fscores_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_subopt_DBS_FDC;
            mean_PYkgY12kn1_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_PYkgY12kn1_subopt_DBS_FDC;

            mean_correct_detection_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_subopt_DBS_FDC_UA;
            reward_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_subopt_DBS_FDC_UA;
            precision_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_subopt_DBS_FDC_UA;
            recall_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_subopt_DBS_FDC_UA;
            fscores_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_subopt_DBS_FDC_UA;
        end

        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            mean_correct_detection_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_RLDeterministicActorCriticAgent;
            reward_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_RLDeterministicActorCriticAgent;
            precision_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_RLDeterministicActorCriticAgent;
            recall_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_RLDeterministicActorCriticAgent;
            fscores_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_RLDeterministicActorCriticAgent;
            mean_PYkgY12kn1_RLDeterministicActorCriticAgent(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_PYkgY12kn1_RLDeterministicActorCriticAgent;

            mean_correct_detection_RLDeterministicActorCriticAgent_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_RLDeterministicActorCriticAgent_UA;
            reward_RLDeterministicActorCriticAgent_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_RLDeterministicActorCriticAgent_UA;
            precision_RLDeterministicActorCriticAgent_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_RLDeterministicActorCriticAgent_UA;
            recall_RLDeterministicActorCriticAgent_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_RLDeterministicActorCriticAgent_UA;
            fscores_RLDeterministicActorCriticAgent_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_RLDeterministicActorCriticAgent_UA;
        end

        if(subopt_DBS_FDC_UA)
            mean_correct_detection_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_UA_subopt_DBS_FDC;
            reward_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_UA_subopt_DBS_FDC;
            precision_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_UA_subopt_DBS_FDC;
            recall_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_UA_subopt_DBS_FDC;
            fscores_UA_subopt_DBS_FDC(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_UA_subopt_DBS_FDC;

            mean_correct_detection_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.mean_correct_detection_UA_subopt_DBS_FDC_UA;
            reward_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.reward_UA_subopt_DBS_FDC_UA;
            precision_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.precision_UA_subopt_DBS_FDC_UA;
            recall_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.recall_UA_subopt_DBS_FDC_UA;
            fscores_UA_subopt_DBS_FDC_UA(P_HgHn1_p_idx, P_HgHn1_q_idx) = eval_outputs.fscores_UA_subopt_DBS_FDC_UA;
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
    plot_inputs.RL_DeterministicActorCriticAgent = RL_DeterministicActorCriticAgent;
    plot_inputs.RL_DeterministicActorCriticAgent_RD = RL_DeterministicActorCriticAgent_RD;
    plot_inputs.subopt_DBS_FDC_UA = subopt_DBS_FDC_UA;
    plot_inputs.P_HgHn1_elem_range = P_HgHn1_elem_range;
    
    plot_inputs.valid_model_flag = valid_model_flag;

    plot_inputs.mean_correct_detection_NC = mean_correct_detection_NC;
    plot_inputs.mean_correct_detection_inst_opt_FDC = mean_correct_detection_inst_opt_FDC;
    plot_inputs.mean_correct_detection_inst_opt_FDC_UA = mean_correct_detection_inst_opt_FDC_UA;
    plot_inputs.mean_correct_detection_subopt_det_SP_FDC = mean_correct_detection_subopt_det_SP_FDC;
    plot_inputs.mean_correct_detection_subopt_det_SP_FDC_UA = mean_correct_detection_subopt_det_SP_FDC_UA;
    plot_inputs.mean_correct_detection_subopt_DBS_FDC = mean_correct_detection_subopt_DBS_FDC;
    plot_inputs.mean_correct_detection_subopt_DBS_FDC_UA = mean_correct_detection_subopt_DBS_FDC_UA;
    plot_inputs.mean_correct_detection_UA_subopt_DBS_FDC = mean_correct_detection_UA_subopt_DBS_FDC;
    plot_inputs.mean_correct_detection_UA_subopt_DBS_FDC_UA = mean_correct_detection_UA_subopt_DBS_FDC_UA;
    plot_inputs.mean_correct_detection_RLDeterministicActorCriticAgent = mean_correct_detection_RLDeterministicActorCriticAgent;
    plot_inputs.mean_correct_detection_RLDeterministicActorCriticAgent_UA = mean_correct_detection_RLDeterministicActorCriticAgent_UA;

    plot_inputs.reward_NC = reward_NC;
    plot_inputs.reward_inst_opt_FDC = reward_inst_opt_FDC;
    plot_inputs.reward_inst_opt_FDC_UA = reward_inst_opt_FDC_UA;
    plot_inputs.reward_subopt_det_SP_FDC = reward_subopt_det_SP_FDC;
    plot_inputs.reward_subopt_det_SP_FDC_UA = reward_subopt_det_SP_FDC_UA;
    plot_inputs.reward_subopt_DBS_FDC = reward_subopt_DBS_FDC;
    plot_inputs.reward_subopt_DBS_FDC_UA = reward_subopt_DBS_FDC_UA;
    plot_inputs.reward_UA_subopt_DBS_FDC = reward_UA_subopt_DBS_FDC;
    plot_inputs.reward_UA_subopt_DBS_FDC_UA = reward_UA_subopt_DBS_FDC_UA;
    plot_inputs.reward_RLDeterministicActorCriticAgent = reward_RLDeterministicActorCriticAgent;
    plot_inputs.reward_RLDeterministicActorCriticAgent_UA = reward_RLDeterministicActorCriticAgent_UA;

    plot_inputs.fscores_NC = fscores_NC;
    plot_inputs.fscores_inst_opt_FDC = fscores_inst_opt_FDC;
    plot_inputs.fscores_inst_opt_FDC_UA = fscores_inst_opt_FDC_UA;
    plot_inputs.fscores_subopt_det_SP_FDC = fscores_subopt_det_SP_FDC;
    plot_inputs.fscores_subopt_det_SP_FDC_UA = fscores_subopt_det_SP_FDC_UA;
    plot_inputs.fscores_subopt_DBS_FDC = fscores_subopt_DBS_FDC;
    plot_inputs.fscores_subopt_DBS_FDC_UA = fscores_subopt_DBS_FDC_UA;
    plot_inputs.fscores_UA_subopt_DBS_FDC = fscores_UA_subopt_DBS_FDC;
    plot_inputs.fscores_UA_subopt_DBS_FDC_UA = fscores_UA_subopt_DBS_FDC_UA;
    plot_inputs.fscores_RLDeterministicActorCriticAgent = fscores_RLDeterministicActorCriticAgent;
    plot_inputs.fscores_RLDeterministicActorCriticAgent_UA = fscores_RLDeterministicActorCriticAgent_UA;

    plot_inputs.precision_NC = precision_NC;
    plot_inputs.precision_inst_opt_FDC = precision_inst_opt_FDC;
    plot_inputs.precision_inst_opt_FDC_UA = precision_inst_opt_FDC_UA;
    plot_inputs.precision_subopt_det_SP_FDC = precision_subopt_det_SP_FDC;
    plot_inputs.precision_subopt_det_SP_FDC_UA = precision_subopt_det_SP_FDC_UA;
    plot_inputs.precision_subopt_DBS_FDC = precision_subopt_DBS_FDC;
    plot_inputs.precision_subopt_DBS_FDC_UA = precision_subopt_DBS_FDC_UA;
    plot_inputs.precision_UA_subopt_DBS_FDC = precision_UA_subopt_DBS_FDC;
    plot_inputs.precision_UA_subopt_DBS_FDC_UA = precision_UA_subopt_DBS_FDC_UA;
    plot_inputs.precision_RLDeterministicActorCriticAgent = precision_RLDeterministicActorCriticAgent;
    plot_inputs.precision_RLDeterministicActorCriticAgent_UA = precision_RLDeterministicActorCriticAgent_UA;

    plot_inputs.recall_NC = recall_NC;
    plot_inputs.recall_inst_opt_FDC = recall_inst_opt_FDC;
    plot_inputs.recall_inst_opt_FDC_UA = recall_inst_opt_FDC_UA;
    plot_inputs.recall_subopt_det_SP_FDC = recall_subopt_det_SP_FDC;
    plot_inputs.recall_subopt_det_SP_FDC_UA = recall_subopt_det_SP_FDC_UA;
    plot_inputs.recall_subopt_DBS_FDC = recall_subopt_DBS_FDC;
    plot_inputs.recall_subopt_DBS_FDC_UA = recall_subopt_DBS_FDC_UA;
    plot_inputs.recall_UA_subopt_DBS_FDC = recall_UA_subopt_DBS_FDC;
    plot_inputs.recall_UA_subopt_DBS_FDC_UA = recall_UA_subopt_DBS_FDC_UA;
    plot_inputs.recall_RLDeterministicActorCriticAgent = recall_RLDeterministicActorCriticAgent;
    plot_inputs.recall_RLDeterministicActorCriticAgent_UA = recall_RLDeterministicActorCriticAgent_UA;

    plot_inputs.mean_PYkgY12kn1_NC = mean_PYkgY12kn1_NC;
    plot_inputs.mean_PYkgY12kn1_inst_opt_FDC = mean_PYkgY12kn1_inst_opt_FDC;
    plot_inputs.mean_PYkgY12kn1_inst_opt_FDC_UA = mean_PYkgY12kn1_inst_opt_FDC_UA;
    plot_inputs.mean_PYkgY12kn1_subopt_det_SP_FDC = mean_PYkgY12kn1_subopt_det_SP_FDC;
    plot_inputs.mean_PYkgY12kn1_subopt_det_SP_FDC_UA = mean_PYkgY12kn1_subopt_det_SP_FDC_UA;
    plot_inputs.mean_PYkgY12kn1_subopt_DBS_FDC = mean_PYkgY12kn1_subopt_DBS_FDC;
    plot_inputs.mean_PYkgY12kn1_subopt_DBS_FDC_UA = mean_PYkgY12kn1_subopt_DBS_FDC_UA;
    plot_inputs.mean_PYkgY12kn1_UA_subopt_DBS_FDC = mean_PYkgY12kn1_UA_subopt_DBS_FDC;
    plot_inputs.mean_PYkgY12kn1_UA_subopt_DBS_FDC_UA = mean_PYkgY12kn1_UA_subopt_DBS_FDC_UA;
    plot_inputs.mean_PYkgY12kn1_RLDeterministicActorCriticAgent = mean_PYkgY12kn1_RLDeterministicActorCriticAgent;
    plot_inputs.mean_PYkgY12kn1_RLDeterministicActorCriticAgent_UA = mean_PYkgY12kn1_RLDeterministicActorCriticAgent_UA;

    plot_fig_sweep_hmm_models(plot_inputs);
    %     set(gcf,'SelectionHighlight','off')
    %     exportgraphics(gcf,'mean_correct_detection_comparision_binary_toy.pdf','ContentType','vector');
end