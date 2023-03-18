function [optimalControlData] = simulate_ActorCriticAgent_RDC(evalParams, pp_data_RD)
params = evalParams.params;
agent = evalParams.rl_agent;
storeAuxData = evalParams.storeAuxData;
storeBeliefs = evalParams.storeBeliefs;
sm_data = evalParams.sm_data;
gt_data = evalParams.gt_data;
h_0_idxs = evalParams.h_0_idxs;
numHorizons = evalParams.numHorizons;

minPowerDemandInW = params.minPowerDemandInW;
k_num = params.k_num;
x_num = params.x_num;
b_num = params.b_num;
x_p_pu = params.x_p_pu;
x_offset = params.x_offset;

%% simulation
x_k_idxs = min(max(1,round((sm_data-minPowerDemandInW)/x_p_pu)-x_offset),x_num);
h_k_idxs = gt_data;
modifiedSMdata = zeros(k_num,numHorizons);
estimatedhypStateData = zeros(k_num,numHorizons);
mean_correct_detection = 0;
bayesian_reward = 0;
mean_PYkgY12kn1 = 0;

if(storeAuxData)
    d_k_idxs = zeros(k_num,numHorizons);
    yc_k_idxs = zeros(k_num,numHorizons);
    z_k_idxs = zeros(k_num,numHorizons);
end
if(storeBeliefs)
    belief_states = zeros(b_num,k_num,numHorizons);
end
actor = agent.Actor;

LoggedSignalsForExpStruct = ["AdversarialRewardEstimate", "Hhk_idx", "Dk_idx", "YkIdx","Zk_idx","P_YksgY12kn1"];
env_RD = SmartGridUserEnv_RD(params, class(agent), pp_data_RD, false, LoggedSignalsForExpStruct);
delete pp_data_RD;

[progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
incPercent = (1/numHorizons/k_num)*100;
internal_routine_fn = @internal_routine;

for horizon_idx = 1:numHorizons
    [mean_correct_detection_, bayesian_reward_,mean_PYkgY12kn1_, estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
        feval(internal_routine_fn, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<FVAL>

    if(storeAuxData)
        d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
        yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
        z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
    end
    if(storeBeliefs)
        belief_states(:,:, horizon_idx) = auxdata.belief_states;
    end
    mean_correct_detection = mean_correct_detection + mean_correct_detection_;
    bayesian_reward = bayesian_reward + bayesian_reward_;
    mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;
end

mean_correct_detection = mean_correct_detection/numHorizons;
bayesian_reward = bayesian_reward/numHorizons;
mean_PYkgY12kn1 = mean_PYkgY12kn1/numHorizons;
progressData.terminate(sprintf('Bayes risk : %f; Bayes reward : %f ',mean_correct_detection, bayesian_reward));

optimalControlData = struct;
optimalControlData.modifiedSMdata = modifiedSMdata;
optimalControlData.estimatedhypStateData = estimatedhypStateData;
if(storeAuxData)
    optimalControlData.d_k_idxs = d_k_idxs;
    optimalControlData.yc_k_idxs = yc_k_idxs;
    optimalControlData.z_k_idxs = z_k_idxs;
end
if(storeBeliefs)
    optimalControlData.belief_states = belief_states;
end
optimalControlData.mean_correct_detection = mean_correct_detection;
optimalControlData.bayesian_reward = bayesian_reward;
optimalControlData.mean_PYkgY12kn1 = mean_PYkgY12kn1;

    function [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1, estimatedhypStateData_, modifiedSMdata_, auxdata] =...
            internal_routine(x_k_idxs_, h_k_idxs_, h0_idx)
        env_RD_ = env_RD;
        params_ = env_RD_.Params;
        action2SubPolicyFcn =@(varargin) feval(strcat(env_RD_.AgentClassString_or_Action2SubPolicyFcn, ".conAction2SubPolicy"), varargin{:});

        HZ2A_ = params_.Function_handles.HZ2A;
        k_num_ = params_.k_num;
        b_num_ = params_.b_num;
        P_HgA = params_.P_HgA;
        C_HgHh_design = params_.C_HgHh_design;
        P_BgA = params_.P_BgA;
        C_HgHh_homogeneous = params_.C_HgHh_homogeneous;

        [P_Akn1]  = reset(env_RD_);

        z_kn1_idx = env_RD_.z0_idx;
        a_kn1_idx = HZ2A_(h0_idx,z_kn1_idx);

        mean_correct_detection_ = 0;
        bayesian_reward_ = 0;
        mean_PYkgY12kn1 = 0;

        modifiedSMdata_ = zeros(k_num_,1);
        estimatedhypStateData_ = zeros(k_num_,1);

        auxdata = [];
        if(storeAuxData)
            auxdata.d_k_idxs = zeros(k_num_,1);
            auxdata.yc_k_idxs = zeros(k_num_,1);
            auxdata.z_k_idxs = zeros(k_num_,1);
        end
        if(storeBeliefs)
            auxdata.belief_states = zeros(b_num_, k_num_);
        end

        for k_in_horizon=1:k_num_
            P_Bkn1 = P_BgA*P_Akn1;
            [action, ~]= getAction(actor, {P_Bkn1});
            [P_Wk] = action2SubPolicyFcn(params_, action{1}, []);
            [P_Aks, Hhk_idxs, P_YksgY12kn1] = env_RD_.get_possible_belief_transitions(params_, P_Akn1, P_Wk, [], env_RD_.P_AgW_YcAkn1);

            x_k_idx_obs = x_k_idxs_(k_in_horizon);
            h_k_idx = h_k_idxs_(k_in_horizon);

            [z_k_idx, yc_k_idx_star, d_k_idx_star] = env_RD_.updateSystemState(x_k_idx_obs, h_k_idx, a_kn1_idx, P_Wk);
            P_Ak = P_Aks{yc_k_idx_star};
            Hhk_idx = Hhk_idxs(yc_k_idx_star);
            
            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(h_k_idx,Hhk_idx);
            bayesian_reward_ = bayesian_reward_ + C_HgHh_design(:, Hhk_idx)'*P_HgA*P_Ak;
            y_k_idx_obs = round(((x_k_idx_obs+params_.x_offset)*params_.x_p_pu + (d_k_idx_star+params_.d_offset)*params_.d_p_pu)/params_.y_p_pu - params_.y_offset);
            modifiedSMdata_(k_in_horizon) = params_.minPowerDemandInW + (y_k_idx_obs + params_.y_offset)*params_.y_p_pu;
            estimatedhypStateData_(k_in_horizon) = Hhk_idx;

            if(storeAuxData)
                auxdata.d_k_idxs(k_in_horizon) = d_k_idx_star;
                auxdata.yc_k_idxs(k_in_horizon) = yc_k_idx_star;
                auxdata.z_k_idxs(k_in_horizon) = z_k_idx;
            end
            if(storeBeliefs)
                auxdata.belief_states(:, k_in_horizon) = P_Akn1_idx;
            end

            P_Akn1 = P_Ak;
            a_kn1_idx = HZ2A_(h_k_idx,z_k_idx);
            mean_PYkgY12kn1 = mean_PYkgY12kn1 + P_YksgY12kn1(yc_k_idx_star);
            send(progressDataQueue, incPercent);
        end
        mean_PYkgY12kn1 = mean_PYkgY12kn1/k_num_;
        mean_correct_detection_ = mean_correct_detection_/k_num_;
        bayesian_reward_ = bayesian_reward_/k_num_;
    end
end

