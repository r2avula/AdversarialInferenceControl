function [agent, policy_fileFullPath] = get_DeterministicActorCriticAgent(params_in,fileNamePrefix, showGUI, UseGPU, pp_data_BEM, BEM_initialized_actor_fileNamePrefix, useparpool)
params = struct;
used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset','u_num',...
    'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','P_Zp1gZD','C_HgHh_design','paramsPrecision','minLikelihoodFilter',...
    'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0','learning_rate_Ac','y_num_for_exploration','learning_rate_C',...
    'P_XHgHn1','actor_net_hidden_layers','actor_net_hidden_layer_neurons_ratio_obs','actor_net_hidden_layer_neurons_ratio_act','critic_net_hidden_layers'...
    'critic_net_hidden_layer_neurons_ratio_obs','discountFactor','TargetSmoothFactor_Ac','with_mean_reward',...
    'MiniBatchSize','C_HgHh_homogeneous','with_bem_initialized_actor','noise_sd','with_controller_reward',...
    's_num','GradientDecayFactor_Ac','SquaredGradientDecayFactor_Ac','num_rand_adv_strats_for_exploration',...
    'Epsilon_adam_Ac','GradientDecayFactor_C','SquaredGradientDecayFactor_C','Epsilon_adam_C',...
    'critic_net_hidden_layer_neurons_ratio_act','logistic_param_limit','y_p_pu','y_offset','with_a_nncells',...
    'minPowerDemandInW','exploration_epsilon','noise_epsilon','TargetSmoothFactor_C'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end

ReplayBufferLength = params_in.ReplayBufferLength;
numTrainHorizons = params_in.numTrainHorizons;

[policy_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    saved_data = load(policy_fileFullPath,'trainStats');
    trainStats = saved_data.trainStats;
    if(numTrainHorizons>length(trainStats.EpisodeIndex))
        saved_data = load(policy_fileFullPath,'agent','rngState');
        [policy] = compute_DeterministicActorCriticAgent_policy(params,saved_data.agent,trainStats, showGUI, pp_data_BEM, saved_data.rngState);
        agent = policy.agent;
        trainStats = policy.trainStats;
        rngState = rng;
        save(policy_fileFullPath,'agent','trainStats','params','rngState',"-v7.3")
    end
else
    [policy] = compute_DeterministicActorCriticAgent_policy(params,[],[], showGUI, pp_data_BEM);
    agent = policy.agent;
    trainStats = policy.trainStats;
    rngState = rng;
    save(policy_fileFullPath,'agent','trainStats','params','rngState',"-v7.3")
end

if(numTrainHorizons<=length(trainStats.EpisodeIndex))
    saved_data = load(policy_fileFullPath,'agent');
    agent = saved_data.agent;
else
    error('\tTraining incomplete! Train Horizons completed: %d! [Required: %d]\n',length(policy.trainStats.EpisodeIndex),numTrainHorizons);
end

%% Supporting functions

    function [policy] = compute_DeterministicActorCriticAgent_policy(params,agent,trainStats, showGUI, pp_data_BEM, rngState)
        agentClass = 'DeterministicActorCriticAgent';
        LoggedSignalsForExpStruct = ["Observation","NextObservation", "Reward", "AdversarialRewardEstimate","Action", "IsDone"];
        if params.with_mean_reward
            if params.with_controller_reward
                LoggedSignalsForExpStruct = [LoggedSignalsForExpStruct, "MeanReward", "P_Aks", "P_YksgSk"];
            else
                LoggedSignalsForExpStruct = [LoggedSignalsForExpStruct, "MeanAdversarialRewardEstimate", "P_Aks", "P_YksgY12kn1"];
            end
        end
        [agent, env] = initializeAgentAndEnv(agent, params, pp_data_BEM, LoggedSignalsForExpStruct, agentClass);
        env.Params.useparpool = useparpool;
        trainingOptions = rlTrainingOptions;
        trainingOptions.MaxEpisodes = numTrainHorizons;
        trainingOptions.ScoreAveragingWindowLength = 100;
        trainingOptions.MaxStepsPerEpisode = params.k_num;
        trainingOptions.StopTrainingCriteria = "EpisodeCount";
        trainingOptions.StopTrainingValue = numTrainHorizons;
        trainingOptions.UseParallel = false;
        trainingOptions.Verbose = ~showGUI;
        trainingOptions.ParallelizationOptions.DataToSendFromWorkers = "gradients";
        trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = 1;

        if(isempty(trainStats))
            trainStats = trainingOptions;
        else
            trainStats.TrainingOptions.MaxEpisodes = numTrainHorizons;
            trainStats.TrainingOptions.StopTrainingValue = numTrainHorizons;
            rng(rngState);
        end
        agent.useparpool = useparpool;
        [trainStats, agent] = train.customTrain(agent, env, showGUI, trainStats);

        agent.Actor.UseDevice = 'cpu';
        agent.Critic.UseDevice = 'cpu';
        agent.P_AgU_YcAkn1 = [];

        policy = struct;
        policy.agent = agent;
        policy.trainStats = trainStats;
    end

    function [agent, env] = initializeAgentAndEnv(agent, params, pp_data_BEM, LoggedSignalsForExpStruct, agentClass)
        env = SmartGridUserEnv_FD(params, agentClass, pp_data_BEM ,true,LoggedSignalsForExpStruct);
        params_env = env.Params;
        a_num = params_env.a_num;
        subpolicy_params_num_con = params_env.subpolicy_params_num_con;
        actor_nnet_intput_size = a_num;
        actor_nnet_output_size = subpolicy_params_num_con;
        params_env.actor_nnet_intput_size = actor_nnet_intput_size;
        params_env.actor_nnet_output_size = actor_nnet_output_size;
        if(isempty(agent))
            logistic_param_limit = params_env.logistic_param_limit;
            controlObservationInfo = rlNumericSpec([actor_nnet_intput_size  1],'LowerLimit',0,'UpperLimit',1);
            controlActionInfo = rlNumericSpec([actor_nnet_output_size 1],'LowerLimit',-logistic_param_limit,'UpperLimit',logistic_param_limit);

            actor_net_hidden_layers = params_env.actor_net_hidden_layers;
            actor_net_hidden_layer_neurons_ratio_obs = params_env.actor_net_hidden_layer_neurons_ratio_obs;
            actor_net_hidden_layer_neurons_ratio_act = params_env.actor_net_hidden_layer_neurons_ratio_act;
            observation_dim = controlObservationInfo.Dimension(1);
            action_dim = controlActionInfo.Dimension(1);

            hiddel_layer_neurons = round(actor_net_hidden_layer_neurons_ratio_obs*observation_dim+...
                actor_net_hidden_layer_neurons_ratio_act*action_dim);

            actorPath = featureInputLayer(observation_dim,'Normalization','none','Name','actorInputLayer');
            for hidden_layer_idx_1 = 1:actor_net_hidden_layers
                actorPath = [actorPath fullyConnectedLayer(hiddel_layer_neurons, 'Name',strcat('actorHiddenLayer_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
                actorPath = [actorPath reluLayer('Name',strcat('actorHiddenLayerAct_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
            end
            actorPath = [actorPath fullyConnectedLayer(action_dim, 'Name','actorPre2OutputLayer')];
            actorPath = [actorPath tanhLayer('Name','actorPre1OutputLayer')];
            actorPath = [actorPath scalingLayer('Scale', logistic_param_limit, 'Name','actorOutputLayer')];
            actor_net = layerGraph(actorPath);
            if params_env.with_bem_initialized_actor
                [actor_net] = getBEMActorNet(actor_net, params, params_env, pp_data_BEM, controlObservationInfo, controlActionInfo);
            end

            actor = actors.ContinuousDeterministicActor(actor_net,controlObservationInfo,controlActionInfo);
            actor = accelerate(actor,true);
            if(UseGPU && gpuDeviceCount("available")>0)
                actor.UseDevice = 'gpu';
            end

            actor_optOpts = rlOptimizerOptions;
            actor_optOpts.LearnRate = params_env.learning_rate_Ac;
            actor_optOpts.OptimizerParameters.Epsilon = params_env.Epsilon_adam_Ac;
            actor_optOpts.OptimizerParameters.GradientDecayFactor = params_env.GradientDecayFactor_Ac;
            actor_optOpts.OptimizerParameters.SquaredGradientDecayFactor = params_env.SquaredGradientDecayFactor_Ac;

            %% Critic design
            criticObservationInfo = rlNumericSpec([actor_nnet_intput_size  1],'LowerLimit',0,'UpperLimit',1);
            criticActionInfo = rlNumericSpec([actor_nnet_output_size 1],'LowerLimit',-logistic_param_limit,'UpperLimit',logistic_param_limit);

            observation_dim = criticObservationInfo.Dimension(1);
            action_dim = criticActionInfo.Dimension(1);

            critic_net_hidden_layers = params_env.critic_net_hidden_layers;
            hiddel_layer_neurons_obs = round(observation_dim*params_env.critic_net_hidden_layer_neurons_ratio_obs);
            hiddel_layer_neurons_act = round(action_dim*params_env.critic_net_hidden_layer_neurons_ratio_act);
            hiddel_layer_neurons = hiddel_layer_neurons_obs + hiddel_layer_neurons_act;
            critic_obsPath = featureInputLayer(observation_dim,'Normalization','none', 'Name','criticObsInputLayer');
            critic_actPath = featureInputLayer(action_dim,'Normalization','none', 'Name','criticActInputLayer');
            for hidden_layer_idx_1 = 1:critic_net_hidden_layers
                critic_obsPath = [critic_obsPath fullyConnectedLayer(hiddel_layer_neurons_obs, 'Name',strcat('criticObsHiddenLayer_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
                critic_actPath = [critic_actPath fullyConnectedLayer(hiddel_layer_neurons_act, 'Name',strcat('criticActHiddenLayer_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
                critic_obsPath = [critic_obsPath softplusLayer( 'Name',strcat('criticObsHiddenLayerAct_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
                critic_actPath = [critic_actPath softplusLayer( 'Name',strcat('criticActHiddenLayerAct_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
            end
            hidden_layer_idx = hidden_layer_idx_1 + 1;
            critic_obsPath = [critic_obsPath fullyConnectedLayer(hiddel_layer_neurons, 'Name',strcat('criticObsHiddenLayer_',num2str(hidden_layer_idx)))];
            critic_actPath = [critic_actPath fullyConnectedLayer(hiddel_layer_neurons, 'Name',strcat('criticActHiddenLayer_',num2str(hidden_layer_idx)))];
            critic_obsPath = [critic_obsPath softplusLayer( 'Name',strcat('criticObsHiddenLayerAct_',num2str(hidden_layer_idx)))];
            critic_actPath = [critic_actPath softplusLayer( 'Name',strcat('criticActHiddenLayerAct_',num2str(hidden_layer_idx)))];            
            critic_obsPath = [critic_obsPath sigmoidLayer( 'Name',strcat('criticObsHiddenLayerAct_tall_',num2str(1)))];
            critic_actPath = [critic_actPath sigmoidLayer( 'Name',strcat('criticActHiddenLayerAct_tall_',num2str(1)))];
            criticPath = additionLayer(2,'Name', 'criticObsActLayersAddition');
            for hidden_layer_idx = 1:critic_net_hidden_layers
                criticPath = [criticPath fullyConnectedLayer(hiddel_layer_neurons, 'Name',strcat('criticHiddenLayer_',num2str(hidden_layer_idx)))]; %#ok<AGROW>
                criticPath = [criticPath reluLayer( 'Name',strcat('criticHiddenLayerAct_',num2str(hidden_layer_idx)))]; %#ok<AGROW>
            end
            criticPath = [criticPath, fullyConnectedLayer(1, 'Name','criticPre2OutputLayer')];
            criticPath = [criticPath, softplusLayer('Name','criticPre1OutputLayer')];
            criticPath = [criticPath, scalingLayer('Scale',-1, 'Name','criticOutputLayer')];
            critic_net = addLayers(layerGraph(critic_obsPath),critic_actPath);
            critic_net = addLayers(critic_net,criticPath);
            critic_net = connectLayers(critic_net,strcat('criticObsHiddenLayerAct_tall_',num2str(1)),'criticObsActLayersAddition/in1');
            critic_net = connectLayers(critic_net,strcat('criticActHiddenLayerAct_tall_',num2str(1)),'criticObsActLayersAddition/in2');

            critic = rlQValueFunction(critic_net,criticObservationInfo,criticActionInfo);

            critic = accelerate(critic,true);
            if(UseGPU && gpuDeviceCount("available")>0)
                critic.UseDevice = 'gpu';
            end

            critic_optOpts = rlOptimizerOptions;
            critic_optOpts.LearnRate = params_env.learning_rate_C;
            critic_optOpts.OptimizerParameters.Epsilon = params_env.Epsilon_adam_C;
            critic_optOpts.OptimizerParameters.GradientDecayFactor = params_env.GradientDecayFactor_C;
            critic_optOpts.OptimizerParameters.SquaredGradientDecayFactor = params_env.SquaredGradientDecayFactor_C;

            %% Agent
            agentOpts = struct;
            agentOpts.ActorOptimizerOptions = actor_optOpts;
            agentOpts.CriticOptimizerOptions = critic_optOpts;
            agentOpts.MiniBatchSize = params_env.MiniBatchSize;
            agentOpts.ExperienceBufferLength = ReplayBufferLength;
            agentOpts.DiscountFactor = params_env.discountFactor;

            agent = feval(agentClass, actor, critic, agentOpts, params_env);
        else
            if agent.AgentOptions.ExperienceBufferLength ~= ReplayBufferLength
                new_ReplayBuffer = replay.RLReplayMemory(agent.Actor.ObservationInfo,agent.Actor.ActionInfo,ReplayBufferLength);
                old_BufferLength = agent.ReplayBuffer.Length;
                if old_BufferLength>ReplayBufferLength
                    [experiences, ~] = sample(agent.ReplayBuffer, ReplayBufferLength);
                else
                    experiences = allExperiences(agent.ReplayBuffer);
                end
                appendWithoutSampleValidation(new_ReplayBuffer,experiences);
                agent.ReplayBuffer = new_ReplayBuffer;
            end
            agent.Params = params_env;
            if(UseGPU && gpuDeviceCount("available")>0)
                agent.Actor.UseDevice = 'gpu';
                agent.Critic.UseDevice = 'gpu';
            end
        end
        agent.P_AgU_YcAkn1 = pp_data_BEM.P_AgU_YcAkn1;        
    end

    function [actor_net] = getBEMActorNet(actor_net, params, params_env, pp_data_BEM, controlObservationInfo, controlActionInfo) %#ok<INUSD>
        params_in_ = params;
        params = struct;
        used_fieldnames_ = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset','u_num',...
            'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','C_HgHh_design','paramsPrecision','minLikelihoodFilter',...
            'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0',...
            'P_XHgHn1','actor_net_hidden_layers','actor_net_hidden_layer_neurons_ratio_obs','actor_net_hidden_layer_neurons_ratio_act',...
            'C_HgHh_homogeneous','s_num', 'logistic_param_limit','y_p_pu','y_offset','minPowerDemandInW'};
        for fn_ = used_fieldnames_
            params.(fn_{1}) = params_in_.(fn_{1});
        end   
        %% Actor design
        [BEM_initialized_acto_fileFullPath,fileExists_] = findFileName(params,BEM_initialized_actor_fileNamePrefix,'params');
        if(fileExists_)
            saved_data_ = load(BEM_initialized_acto_fileFullPath,'actor_net');
            actor_net = saved_data_.actor_net;
        else
            %% Actor nnet initialization
            evalParams = struct;
            evalParams.params = params;
            evalParams.numHorizons = 3000;
            evalParams.storeAuxData = false;
            evalParams.in_debug_mode = false;
            evalParams.storeBeliefs = true;
            evalParams.storeYkn1Idxs = true;
            subpolicy_params_num_con = controlActionInfo.Dimension(1);

            responses_g_Y = zeros(params.y_control_num,subpolicy_params_num_con);
            for y_idx = 1:params.y_control_num
                responses_g_Y(y_idx,:) = DeterministicActorCriticAgent.conSubPolicy2Action(params_env,pp_data_BEM.P_UkgYckn1Idx{y_idx})';
            end

            [simulationData] = simulate_BEM_FDC(evalParams, pp_data_BEM);
            nnet_init_train_obs = reshape(simulationData.belief_states,params.a_num,[])';
            rand_y_idxs = reshape(simulationData.yc_kn1_idxs,[],1);
            responses = responses_g_Y(rand_y_idxs,:);

            nnet_init_train_options = trainingOptions('adam', ...
                'MiniBatchSize',32, ...
                'Shuffle','every-epoch', ...
                'Plots','training-progress', ...
                'MaxEpochs',10,... 
                'Verbose',false);

            actor_net = addLayers(actor_net,regressionLayer);       
            actor_net = connectLayers(actor_net,"actorOutputLayer","regressionoutput");
            actor_net = trainNetwork(nnet_init_train_obs,responses,actor_net,nnet_init_train_options);
            actor_net = layerGraph(actor_net);
            actor_net = removeLayers(actor_net,'regressionoutput');

            % actor = actors.ContinuousDeterministicActor(actor_net,controlObservationInfo,controlActionInfo);
            % evalParams = struct;
            % evalParams.params = params;
            % evalParams.numHorizons = 3000;
            % evalParams.storeAuxData = false;
            % evalParams.in_debug_mode = false;
            % evalParams.storeBeliefs = true;
            % evalParams.storeYkn1Idxs = true;
            % validate_BEM_FDC_trained_nnet(evalParams, pp_data_BEM, actor);

            save(BEM_initialized_acto_fileFullPath,'actor_net','params');
        end
    end
end

