function [simulationData] = validate_BEM_FDC_trained_nnet_2(evalParams, pp_data, actor)
storeAuxData = evalParams.storeAuxData;
storeBeliefs = evalParams.storeBeliefs;
numHorizons = evalParams.numHorizons;
params = evalParams.params;

a_num = params.a_num;
k_num = params.k_num;

%% Initialization
LoggedSignalsForExpStruct = ["AdversarialRewardEstimate", "Hhk_idx", "Dk_idx", "YkIdx","Zk_idx","P_YksgY12kn1",...
    "Hhk_idx","Dk_idx","YkIdx","Zk_idx","Hk_idx","x_k_idx_obs"];
env = SmartGridUserEnv_FD(params, @loopBackFcn, pp_data, true, LoggedSignalsForExpStruct);

%% simulation
modifiedSMdata = zeros(k_num,numHorizons);
estimatedhypStateData = zeros(k_num,numHorizons);
Ykn1_idxs = zeros(k_num,numHorizons);
mean_correct_detection = 0;
bayesian_reward = 0;
mean_PYkgY12kn1 = 0;

if(storeAuxData)
    d_k_idxs = zeros(k_num,numHorizons);
    yc_k_idxs = zeros(k_num,numHorizons);
    z_k_idxs = zeros(k_num,numHorizons);
end
if(storeBeliefs)
    belief_states = zeros(a_num,k_num,numHorizons);
end

[progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;

worketData = struct;
worketData.env = env;
worketData.storeAuxData = storeAuxData;
worketData.storeBeliefs = storeBeliefs;
worketData = parallel.pool.Constant(worketData);

parfor horizon_idx = 1:numHorizons
    Y_kn1_idx = 1;
    [mean_correct_detection_, bayesian_reward_,mean_PYkgY12kn1_, Ykn1_idxs(:, horizon_idx), estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
        feval(internal_routine_fn, worketData, Y_kn1_idx); %#ok<*FVAL>

    if(storeAuxData)
        d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
        yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
        z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
    end
    if(storeBeliefs)
        belief_states(:,:,horizon_idx) = auxdata.belief_states;
    end
    mean_correct_detection = mean_correct_detection + mean_correct_detection_;
    bayesian_reward = bayesian_reward + bayesian_reward_;
    mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;

    send(progressDataQueue, incPercent);
end

mean_correct_detection = mean_correct_detection/numHorizons;
bayesian_reward = bayesian_reward/numHorizons;
mean_PYkgY12kn1 = mean_PYkgY12kn1/numHorizons;
progressData.terminate(sprintf('Bayes risk : %f; Bayes reward : %f ',mean_correct_detection, bayesian_reward));


% store
simulationData = struct;
simulationData.modifiedSMdata = modifiedSMdata;
simulationData.estimatedhypStateData = estimatedhypStateData;
simulationData.Ykn1_idxs = Ykn1_idxs;
if(storeAuxData)
    simulationData.d_k_idxs = d_k_idxs;
    simulationData.yc_k_idxs = yc_k_idxs;
    simulationData.z_k_idxs = z_k_idxs;
end
if(storeBeliefs)
    simulationData.belief_states = belief_states;
end
simulationData.mean_correct_detection = mean_correct_detection;
simulationData.bayesian_reward = bayesian_reward;
simulationData.mean_PYkgY12kn1 = mean_PYkgY12kn1;

%% Supporting functions
    function [P_Uk_control] = loopBackFcn(~, P_Uk,~)        
        P_Uk_control = P_Uk;
    end

    function [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1,Ykn1_idxs_, estimatedhypStateData_, modifiedSMdata_, auxdata] =...
            internal_routine(worketData, Y_kn1_idx)

        worketData = worketData.Value;
        env_ = worketData.env;
        storeAuxData_ = worketData.storeAuxData;
        storeBeliefs_ = worketData.storeBeliefs;

        params_ = env_.Params;
        k_num_ = params_.k_num;
        a_num_ = params_.a_num;
        P_Akn1 = reset(env_);
        C_HgHh_homogeneous = params_.C_HgHh_homogeneous;

        mean_correct_detection_ = 0;
        bayesian_reward_ = 0;
        mean_PYkgY12kn1 = 0;

        Ykn1_idxs_ = zeros(k_num_,1);
        modifiedSMdata_ = zeros(k_num_,1);
        estimatedhypStateData_ = zeros(k_num_,1);

        auxdata = [];
        if(storeAuxData_)
            auxdata.d_k_idxs = zeros(k_num_,1);
            auxdata.yc_k_idxs = zeros(k_num_,1);
            auxdata.z_k_idxs = zeros(k_num_,1);
        end
        if(storeBeliefs_)
            auxdata.belief_states = zeros(a_num_,k_num_);
        end

        
        for k_in_horizon=1:k_num_
            action = getAction(actor, {P_Akn1});
            P_Uk = DeterministicActorCriticAgent.conAction2SubPolicy(params_,action{1});
            Ykn1_idxs_(k_in_horizon) = Y_kn1_idx;
            
            action_info.P_Uk = P_Uk;
            action_info.belief_trans_info = [];
            [P_Ak, Reward, ~, LoggedSignals] = env_.step(P_Uk,action_info);

            Hhk_idx = LoggedSignals.Hhk_idx;
            d_k_idx_star = LoggedSignals.Dk_idx;
            yc_k_idx_star = LoggedSignals.YkIdx;
            z_k_idx = LoggedSignals.Zk_idx;
            Hk_idx= LoggedSignals.Hk_idx;
            x_k_idx_obs = LoggedSignals.x_k_idx_obs;

            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(Hk_idx,Hhk_idx);
            bayesian_reward_ = bayesian_reward_ - Reward;
            y_k_idx_obs = round(((x_k_idx_obs+params_.x_offset)*params_.x_p_pu + (d_k_idx_star+params_.d_offset)*params_.d_p_pu)/params_.y_p_pu - params_.y_offset);
            modifiedSMdata_(k_in_horizon) = params_.minPowerDemandInW + (y_k_idx_obs + params_.y_offset)*params_.y_p_pu;
            estimatedhypStateData_(k_in_horizon) = Hhk_idx;

            if(storeAuxData_)
                auxdata.d_k_idxs(k_in_horizon) = d_k_idx_star;
                auxdata.yc_k_idxs(k_in_horizon) = yc_k_idx_star;
                auxdata.z_k_idxs(k_in_horizon) = z_k_idx;
            end
            if(storeBeliefs_)
                auxdata.belief_states(:,k_in_horizon) = P_Ak;
            end

            P_Akn1 = P_Ak;
            mean_PYkgY12kn1 = mean_PYkgY12kn1 + LoggedSignals.P_YksgY12kn1(yc_k_idx_star);
            Y_kn1_idx = yc_k_idx_star;
        end
        mean_PYkgY12kn1 = mean_PYkgY12kn1/k_num_;
        mean_correct_detection_ = mean_correct_detection_/k_num_;
        bayesian_reward_ = bayesian_reward_/k_num_;
    end
end

