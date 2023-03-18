function [agent, policy_fileFullPath] = get_DeterministicActorCriticAgent_RDC(params_in,fileNamePrefix, showGUI, pp_data, UseGPU)
params = struct;
used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
    'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','P_Zp1gZD','C_HgHh_design','paramsPrecision','minLikelihoodFilter',...
    'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0','learning_rate_Ac','learning_rate_C','C_HgHh_homogeneous',...
    'discountFactor','P_XHgHn1','actor_net_hidden_layers','actor_net_hidden_layer_neurons_ratio','critic_net_hidden_layers',...
    'critic_net_hidden_layer_neurons_ratio_obs','exploration_epsilon','TargetSmoothFactor_C','penalty_factor',...
    'MiniBatchSize','InMemoryUpdateInterval','ReplayBufferLength','TargetSmoothFactor_Ac',...
    'l_num','w_num','LIdxgZIdx','ZIdxsgLIdx','z_num_per_level','b_num','t_num','P_HgB','P_BgA'...
    ,'y_num_for_exploration','num_rand_adv_strats_for_exploration','noise_epsilon',...
    'logistic_param_limit','GradientDecayFactor_Ac','SquaredGradientDecayFactor_Ac','Epsilon_adam_Ac','GradientDecayFactor_C',...
    'SquaredGradientDecayFactor_C','Epsilon_adam_C','critic_net_hidden_layer_neurons_ratio_act'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end

numTrainHorizons = params_in.numTrainHorizons;

[policy_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    saved_data = load(policy_fileFullPath,'trainStats');
    trainStats = saved_data.trainStats;
    if(numTrainHorizons>length(trainStats.EpisodeIndex))
        saved_data = load(policy_fileFullPath,'agent', 'rngState');
        [policy] = compute_DeterministicActorCriticAgent_RDC_policy(params,saved_data.agent,trainStats, showGUI,pp_data, saved_data.rngState);
        agent = policy.agent;
        trainStats = policy.trainStats;
        rngState = rng;
        save(policy_fileFullPath,'agent','trainStats','params','rngState')
    end
else
    [policy] = compute_DeterministicActorCriticAgent_RDC_policy(params,[],[],showGUI,pp_data,[]);
    agent = policy.agent;
    trainStats = policy.trainStats;
    rngState = rng;
    save(policy_fileFullPath,'agent','trainStats','params','rngState')
end

if(numTrainHorizons<=length(trainStats.EpisodeIndex))
    saved_data = load(policy_fileFullPath,'agent');
    agent = saved_data.agent;
else
    error('\tTraining incomplete! Train Horizons completed: %d! [Required: %d]\n',length(policy.trainStats.EpisodeIndex),numTrainHorizons);
end

%% Supporting functions

    function [policy] = compute_DeterministicActorCriticAgent_RDC_policy(params,agent,trainStats, showGUI, pp_data, rngState)
        agentClass = 'DeterministicActorCriticAgent_RDC';
        LoggedSignalsForExpStruct = ["Observation", "NextObservation", "Reward", "Action", "AdversarialRewardEstimate", "IsDone"];
        env = SmartGridUserEnv_RD(params, agentClass, pp_data, true, LoggedSignalsForExpStruct);
        
        [agent] = initializeAgent(agent,env);

        if agent.Params.exploration_epsilon>0
            [~] = evalc('gcp;');
        end
        
        trainingOptions = rlTrainingOptions;
        trainingOptions.MaxEpisodes = numTrainHorizons;
        trainingOptions.ScoreAveragingWindowLength = 100;
        trainingOptions.MaxStepsPerEpisode = params.k_num;
        trainingOptions.StopTrainingCriteria = "EpisodeCount";
        trainingOptions.StopTrainingValue = trainingOptions.MaxEpisodes;
        trainingOptions.UseParallel = false;
        trainingOptions.Verbose = ~showGUI;

        if(isempty(trainStats))
            trainOpts = trainingOptions;
        else
            trainStats.TrainingOptions = trainingOptions;
            trainStats.Information.TrainingOpts = trainingOptions;
            trainOpts = train.RLTrainingResult.struct2class(trainStats);
            rng(rngState);
        end
        [trainStats, agent] = train.customTrain(agent, env, showGUI, trainOpts);

        trainStats_saved.EpisodeIndex = trainStats.EpisodeIndex;
        trainStats_saved.EpisodeReward = trainStats.EpisodeReward;
        trainStats_saved.AdversarialEpisodeRewardEstimate = trainStats.AdversarialEpisodeRewardEstimate;
        trainStats_saved.EpisodeSteps = trainStats.EpisodeSteps;
        trainStats_saved.AverageEpisodeReward = trainStats.AverageEpisodeReward;
        trainStats_saved.AverageAdversarialEpisodeRewardEstimate = trainStats.AverageAdversarialEpisodeRewardEstimate;
        trainStats_saved.TotalAgentSteps = trainStats.TotalAgentSteps;
        trainStats_saved.AverageSteps = trainStats.AverageSteps;
        trainStats_saved.Loss = trainStats.Loss;
        trainStats_saved.EpisodeQ0 = trainStats.EpisodeQ0;
        trainStats_saved.SimulationInfo = trainStats.SimulationInfo;
        trainStats_saved.TimeStamp = trainStats.TimeStamp;
        trainStats_saved.Information = rmfield(trainStats.Information,'TrainingOpts');

        policy = struct;
        policy.agent = agent;
        policy.trainStats = trainStats_saved;
    end

    function [agent] = initializeAgent(agent,env)
        if(isempty(agent))
            params_ = env.Params;
            b_num = params_.b_num;
            logistic_param_limit = params_.logistic_param_limit;
            subpolicy_params_num_con = params_.subpolicy_params_num_con;
            actor_net_hidden_layers = params_.actor_net_hidden_layers;
            actor_net_hidden_layer_neurons_ratio = params_.actor_net_hidden_layer_neurons_ratio;

            %% Actor design
            controlObservationInfo = rlNumericSpec([b_num  1],'LowerLimit',0,'UpperLimit',1);
            controlActionInfo = rlNumericSpec([subpolicy_params_num_con 1],'LowerLimit',0,'UpperLimit',1);

            observation_dim = controlObservationInfo.Dimension(1);
            action_dim = controlActionInfo.Dimension(1);

            hiddel_layer_neurons = round(mean([observation_dim,action_dim])*actor_net_hidden_layer_neurons_ratio);

            actorPath = featureInputLayer(observation_dim,'Normalization','none','Name','actorInputLayer');
            for hidden_layer_idx_1 = 1:actor_net_hidden_layers
                actorPath = [actorPath fullyConnectedLayer(hiddel_layer_neurons, 'Name',strcat('actorHiddenLayer_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
                actorPath = [actorPath reluLayer('Name',strcat('actorHiddenLayerAct_',num2str(hidden_layer_idx_1)))]; %#ok<AGROW>
            end
            actorPath = [actorPath fullyConnectedLayer(action_dim, 'Name','actorPre4OutputLayer')];
            actorPath = [actorPath tanhLayer('Name','actorPre1OutputLayer')];
            actorPath = [actorPath scalingLayer('Scale', logistic_param_limit, 'Name','actorOutputLayer')];
            
            %% 
            actor_net = layerGraph(actorPath);

            actor = actors.ContinuousDeterministicActor(actor_net,controlObservationInfo,controlActionInfo);
            actor = accelerate(actor,true);
            if(UseGPU && gpuDeviceCount("available")>0)
                actor.UseDevice = 'gpu';
            end

            actor_optOpts = rlOptimizerOptions;
            actor_optOpts.LearnRate = params_.learning_rate_Ac;
            actor_optOpts.OptimizerParameters.Epsilon = params_.Epsilon_adam_Ac;
            actor_optOpts.OptimizerParameters.GradientDecayFactor = params_.GradientDecayFactor_Ac;
            actor_optOpts.OptimizerParameters.SquaredGradientDecayFactor = params_.SquaredGradientDecayFactor_Ac;

            %% Critic design
            criticObservationInfo = rlNumericSpec([b_num  1],'LowerLimit',0,'UpperLimit',1);
            criticActionInfo = rlNumericSpec([subpolicy_params_num_con 1],'LowerLimit',0,'UpperLimit',1);

            observation_dim = criticObservationInfo.Dimension(1);
            action_dim = criticActionInfo.Dimension(1);

            critic_net_hidden_layers = params_.critic_net_hidden_layers;
            hiddel_layer_neurons_obs = round(observation_dim*params_.critic_net_hidden_layer_neurons_ratio_obs);
            hiddel_layer_neurons_act = round(action_dim*params_.critic_net_hidden_layer_neurons_ratio_act);
            hiddel_layer_neurons = round(mean([hiddel_layer_neurons_obs,hiddel_layer_neurons_act]));

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
            criticPath = [criticPath, fullyConnectedLayer(1, 'Name','criticPre2OutputLayer'),...
                softplusLayer('Name','criticPre1OutputLayer'),...
                scalingLayer('Scale', -1, 'Name','criticOutputLayer')];
            %             criticPath = [criticPath, fullyConnectedLayer(1, 'Name','criticOutputLayer')];

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
            critic_optOpts.LearnRate = params_.learning_rate_C;
            critic_optOpts.OptimizerParameters.Epsilon = params_.Epsilon_adam_C;
            critic_optOpts.OptimizerParameters.GradientDecayFactor = params_.GradientDecayFactor_C;
            critic_optOpts.OptimizerParameters.SquaredGradientDecayFactor = params_.SquaredGradientDecayFactor_C;

            %% Agent
            agentOpts = struct;
            agentOpts.ActorOptimizerOptions = actor_optOpts;
            agentOpts.CriticOptimizerOptions = critic_optOpts;
            agentOpts.MiniBatchSize = params_.MiniBatchSize;
            agentOpts.ReplayBufferLength = params_.ReplayBufferLength;
            agentOpts.EpisodeBufferLength = params_.k_num;
            agentOpts.InMemoryUpdateInterval = params_.InMemoryUpdateInterval;

            agent = feval(env.AgentClassString_or_Action2SubPolicyFcn, actor, critic, agentOpts, params_);
        end
        setP_AgW_YcAkn1(agent, env.P_AgW_YcAkn1)

        if(UseGPU && gpuDeviceCount("available")>0)
            agent.Actor.UseDevice = 'gpu';
            agent.Critic.UseDevice = 'gpu';
        end
    end

end

