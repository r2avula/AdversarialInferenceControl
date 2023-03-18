function [optimalControlData] = simulate_subopt_DBS_FDC(evalParams,policy,PP_data_filename,pp_data, useparpool)
params = evalParams.params;
storeAuxData = evalParams.storeAuxData;
storeBeliefs = evalParams.storeBeliefs;
sm_data = evalParams.sm_data;
gt_data = evalParams.gt_data;
h_0_idxs = evalParams.h_0_idxs;
numHorizons = evalParams.numHorizons;

minPowerDemandInW = params.minPowerDemandInW;
k_num = params.k_num;
x_num = params.x_num;
x_p_pu = params.x_p_pu;
x_offset = params.x_offset;

cached_data = load(PP_data_filename);
a_nn_cells = cached_data.('a_nn_cells');

beliefTransitionMap = policy.beliefTransitionMap;
P_Ug_PAkn1_Idx = policy.emu_strategy;


%% simulation
x_k_idxs = min(max(1,round((sm_data-minPowerDemandInW)/x_p_pu)-x_offset),x_num);
h_k_idxs = gt_data;
modifiedSMdata = zeros(k_num,numHorizons);
estimatedhypStateData = zeros(k_num,numHorizons);
mean_correct_detection = 0;
bayesian_reward = 0;
mean_PYkgY12kn1 = 0;

if(storeAuxData)
    yc_k_idxs = zeros(k_num,numHorizons);
    z_k_idxs = zeros(k_num,numHorizons);
    d_k_idxs = zeros(k_num,numHorizons);
end
if(storeBeliefs)
    PAkn1_idxs = zeros(k_num,numHorizons);
end

LoggedSignalsForExpStruct = ["AdversarialRewardEstimate", "Hhk_idx", "Dk_idx", "YkIdx","Zk_idx","P_YksgY12kn1"];
env = SmartGridUserEnv_FD(params, @loopBackFcn, pp_data, false,LoggedSignalsForExpStruct);

[progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;
[~,p_pool] = evalc('gcp(''nocreate'');');

worketData = struct;
worketData.env = env;
worketData.a_nn_cells = a_nn_cells;
worketData.P_Ug_PAkn1_Idx = P_Ug_PAkn1_Idx;
worketData.beliefTransitionMap = beliefTransitionMap;
worketData.storeAuxData = storeAuxData;
worketData.storeBeliefs = storeBeliefs;
if isempty(p_pool) || ~useparpool
    for horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_,mean_PYkgY12kn1_, estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
            feval(internal_routine_fn, worketData, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<FVAL> 

        if(storeAuxData)
            d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
            yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
            z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
        end
        if(storeBeliefs)
            PAkn1_idxs(:,horizon_idx) = auxdata.PAkn1_PolyIdxs;
        end
        mean_correct_detection = mean_correct_detection + mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;

        send(progressDataQueue, incPercent);
    end
else
    worketData = parallel.pool.Constant(worketData);
    parfor horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1_, estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
            feval(internal_routine_fn, worketData.Value, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<FVAL> 

        if(storeAuxData)
            d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
            yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
            z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
        end
        if(storeBeliefs)
            PAkn1_idxs(:,horizon_idx) = auxdata.PAkn1_idxs;
        end
        mean_correct_detection = mean_correct_detection + mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;

        send(progressDataQueue, incPercent);
    end
end


mean_correct_detection = mean_correct_detection/numHorizons;
bayesian_reward = bayesian_reward/numHorizons;
mean_PYkgY12kn1 = mean_PYkgY12kn1/numHorizons;
progressData.terminate(sprintf('Bayes risk : %f; Bayes reward : %f ',mean_correct_detection, bayesian_reward));

optimalControlData = struct;
optimalControlData.mean_correct_detection = mean_correct_detection;
optimalControlData.bayesian_reward = bayesian_reward;
optimalControlData.mean_PYkgY12kn1 = mean_PYkgY12kn1;
optimalControlData.modifiedSMdata = modifiedSMdata;
optimalControlData.estimatedhypStateData = estimatedhypStateData;
if(storeAuxData)
    optimalControlData.d_k_idxs = d_k_idxs;
    optimalControlData.z_k_idxs = z_k_idxs;
    optimalControlData.yc_k_idxs = yc_k_idxs;
end
if(storeBeliefs)
    optimalControlData.PAkn1_idxs = PAkn1_idxs;
end

    function [P_Uk_control] = loopBackFcn(~, P_Uk,~)        
        P_Uk_control = P_Uk;
    end

    function [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1, estimatedhypStateData_, modifiedSMdata_, auxdata] =...
            internal_routine(worketData, x_k_idxs_, h_k_idxs_, h0_idx)
        env_ = worketData.env;
        storeAuxData_ = worketData.storeAuxData;
        storeBeliefs_ = worketData.storeBeliefs;
        a_nn_cells_ = worketData.a_nn_cells;
        P_Ug_PAkn1_Idx_ = worketData.P_Ug_PAkn1_Idx;
        beliefTransitionMap_ = worketData.beliefTransitionMap;

        params_ = env_.Params;
        HZ2A_ = params_.Function_handles.HZ2A;
        XHAn1_2S_ = params_.Function_handles.XHAn1_2S;
        k_num_ = params_.k_num;
        C_HgHh_homogeneous = params_.C_HgHh_homogeneous;

        P_Akn1 = reset(env_);
        z_kn1_idx = env_.z0_idx;
        a_kn1_idx = HZ2A_(h0_idx,z_kn1_idx);

        mean_correct_detection_ = 0;
        bayesian_reward_ = 0;
        mean_PYkgY12kn1 = 0;

        modifiedSMdata_ = zeros(k_num_,1);
        estimatedhypStateData_ = zeros(k_num_,1);

        auxdata = [];
        if(storeAuxData_)
            auxdata.d_k_idxs = zeros(k_num_,1);
            auxdata.yc_k_idxs = zeros(k_num_,1);
            auxdata.z_k_idxs = zeros(k_num_,1);
        end
        if(storeBeliefs_)
            auxdata.PAkn1_idxs = zeros(k_num_, 1);
        end

        a_nn_roundOffBelief_fn = @(x)roundOffInSimplex(x,[],a_nn_cells_);
        [~,P_Akn1_idx] = a_nn_roundOffBelief_fn(P_Akn1);

        for k_in_horizon=1:k_num_
            P_Uk = P_Ug_PAkn1_Idx_(:,P_Akn1_idx);
            PAIdx_given_Yc = beliefTransitionMap_(:,P_Akn1_idx);

            x_k_idx_obs = x_k_idxs_(k_in_horizon);
            h_k_idx = h_k_idxs_(k_in_horizon);
            s_k_idx = XHAn1_2S_(x_k_idx_obs,h_k_idx,a_kn1_idx);

            env_.setState({P_Akn1, s_k_idx});
            action_info.belief_trans_info = [];
            [P_Ak, Reward, ~, LoggedSignals] = env_.step(P_Uk,action_info);

            Hhk_idx = LoggedSignals.Hhk_idx;
            d_k_idx_star = LoggedSignals.Dk_idx;
            yc_k_idx_star = LoggedSignals.YkIdx;
            z_k_idx = LoggedSignals.Zk_idx;

            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(h_k_idx,Hhk_idx);
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
                auxdata.PAkn1_idxs(k_in_horizon) = P_Akn1_idx;
            end

            P_Akn1 = P_Ak;
            a_kn1_idx = HZ2A_(h_k_idx,z_k_idx);
            P_Akn1_idx = PAIdx_given_Yc(yc_k_idx_star);
            mean_PYkgY12kn1 = mean_PYkgY12kn1 + LoggedSignals.P_YksgY12kn1(yc_k_idx_star);
        end
        mean_PYkgY12kn1 = mean_PYkgY12kn1/k_num_;
        mean_correct_detection_ = mean_correct_detection_/k_num_;
        bayesian_reward_ = bayesian_reward_/k_num_;
    end
end

