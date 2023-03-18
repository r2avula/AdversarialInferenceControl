function [optimalControlData_UA, detectionData_AA] = simulate_UA_subopt_DBS_FDC(evalParams,policy,PP_data_filename,pp_data, useparpool)
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
a_num = params.a_num;
x_p_pu = params.x_p_pu;
x_offset = params.x_offset;

cached_data = load(PP_data_filename);
h_nn_cells = cached_data.('h_nn_cells');
possible_PHIdxk_given_Yck_PHIdxkn1 = cached_data.('possible_PHIdxk_given_Yck_PHIdxkn1');
P_Ukg_PHIdxkn1 = policy.emu_strategy;
delete cached_data

%% simulation
x_k_idxs = min(max(1,round((sm_data-minPowerDemandInW)/x_p_pu)-x_offset),x_num);
h_k_idxs = gt_data;
modifiedSMdata = zeros(k_num,numHorizons);
estimatedhypStateData = zeros(k_num,numHorizons);
estimatedhypStateData_UA = zeros(k_num,numHorizons);
mean_correct_detection = 0;
bayesian_reward = 0;
mean_correct_detection_UA = 0;
bayesian_reward_UA = 0;

if(storeAuxData)
    yc_k_idxs = zeros(k_num,numHorizons);
    z_k_idxs = zeros(k_num,numHorizons);
    d_k_idxs = zeros(k_num,numHorizons);
end
if(storeBeliefs)
    belief_Idx_UA = zeros(k_num,numHorizons);
    belief_AA = zeros(a_num,k_num,numHorizons);
end

LoggedSignalsForExpStruct = ["AdversarialRewardEstimate", "Hhk_idx", "Dk_idx", "YkIdx","Zk_idx"];
env = SmartGridUserEnv_FD(params, @loopBackFcn, pp_data, false,LoggedSignalsForExpStruct);

[progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;
[~,p_pool] = evalc('gcp(''nocreate'');');

if isempty(p_pool) || ~useparpool
    for horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_,mean_correct_detection_UA_,bayesian_reward_UA_, auxdata] =...
            feval(internal_routine_fn, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<FVAL>

        estimatedhypStateData(:, horizon_idx) = auxdata.estimatedhypStateData;
        estimatedhypStateData_UA(:, horizon_idx) = auxdata.estimatedhypStateData_UA;

        modifiedSMdata(:, horizon_idx) = auxdata.modifiedSMdata;
        if(storeAuxData)
            d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
            yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
            z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
        end
        if(storeBeliefs)
            belief_Idx_UA(:,horizon_idx) = auxdata.belief_Idx_UA;
            belief_AA(:,:,horizon_idx) = auxdata.belief_AA;
        end
        mean_correct_detection =mean_correct_detection +mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_correct_detection_UA =mean_correct_detection_UA +mean_correct_detection_UA_;
        bayesian_reward_UA = bayesian_reward_UA + bayesian_reward_UA_;

        send(progressDataQueue, incPercent);
    end
else
    parfor horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_,mean_correct_detection_UA_,bayesian_reward_UA_, auxdata] =...
            feval(internal_routine_fn, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<FVAL>

        estimatedhypStateData(:, horizon_idx) = auxdata.estimatedhypStateData;
        estimatedhypStateData_UA(:, horizon_idx) = auxdata.estimatedhypStateData_UA;

        modifiedSMdata(:, horizon_idx) = auxdata.modifiedSMdata;
        if(storeAuxData)
            d_k_idxs(:,horizon_idx) = auxdata.d_k_idxs;
            yc_k_idxs(:,horizon_idx) = auxdata.yc_k_idxs;
            z_k_idxs(:,horizon_idx) = auxdata.z_k_idxs;
        end
        if(storeBeliefs)
            belief_Idx_UA(:,horizon_idx) = auxdata.belief_Idx_UA;
            belief_AA(:,:,horizon_idx) = auxdata.belief_AA;
        end
        mean_correct_detection =mean_correct_detection +mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_correct_detection_UA =mean_correct_detection_UA +mean_correct_detection_UA_;
        bayesian_reward_UA = bayesian_reward_UA + bayesian_reward_UA_;

        send(progressDataQueue, incPercent);
    end
end


mean_correct_detection =mean_correct_detection/k_num/numHorizons;
bayesian_reward = bayesian_reward/k_num/numHorizons;
mean_correct_detection_UA =mean_correct_detection_UA/k_num/numHorizons;
bayesian_reward_UA = bayesian_reward_UA/k_num/numHorizons;

progressData.terminate(sprintf('Bayes risk : %f; Bayes reward : %f ',mean_correct_detection, bayesian_reward));

optimalControlData_UA = struct;
optimalControlData_UA.mean_correct_detection =mean_correct_detection_UA;
optimalControlData_UA.bayesian_reward = bayesian_reward_UA;
optimalControlData_UA.modifiedSMdata = modifiedSMdata;
optimalControlData_UA.estimatedhypStateData = estimatedhypStateData_UA;
if(storeAuxData)
    optimalControlData_UA.d_k_idxs = d_k_idxs;
    optimalControlData_UA.z_k_idxs = z_k_idxs;
    optimalControlData_UA.yc_k_idxs = yc_k_idxs;
end
if(storeBeliefs)
    optimalControlData_UA.PHkn1_idxs = belief_Idx_UA;
end

detectionData_AA = struct;
detectionData_AA.mean_correct_detection =mean_correct_detection;
detectionData_AA.bayesian_reward = bayesian_reward;
detectionData_AA.estimatedhypStateData = estimatedhypStateData;
if(storeBeliefs)
    detectionData_AA.belief_states = belief_AA;
end


    function [P_Uk_control] = loopBackFcn(~, P_Uk,~)
        P_Uk_control = P_Uk;
    end

    function [mean_correct_detection_, bayesian_reward_, mean_correct_detection_UA_t, bayesian_reward_UA_t, auxdata] =...
            internal_routine(x_k_idxs_, h_k_idxs_, h0_idx)
        env_ = env;
        h_nn_cells_ = h_nn_cells;

        params_ = env_.Params;
        HZ2A_ = params_.Function_handles.HZ2A;
        XHAn1_2S_ = params_.Function_handles.XHAn1_2S;
        k_num_ = params_.k_num;
        a_num_ = params_.a_num;
        P_H0 = params_.P_H0;
        C_HgHh_design = params_.C_HgHh_design;
        C_HgHh_homogeneous = params_.C_HgHh_homogeneous;

        P_Akn1 = reset(env_);
        z_kn1_idx = env_.z0_idx;
        a_kn1_idx = HZ2A_(h0_idx,z_kn1_idx);

        mean_correct_detection_ = 0;
        bayesian_reward_ = 0;

        mean_correct_detection_UA_t = 0;
        bayesian_reward_UA_t = 0;

        modifiedSMdata_ = zeros(k_num_,1);
        estimatedhypStateData_ = zeros(k_num_,1);
        estimatedhypStateData_UA_ = zeros(k_num_,1);

        auxdata = struct;
        if(storeAuxData)
            auxdata.d_k_idxs = zeros(k_num_,1);
            auxdata.yc_k_idxs = zeros(k_num_,1);
            auxdata.z_k_idxs = zeros(k_num_,1);
        end
        if(storeBeliefs)
            auxdata.belief_Idx_UA = zeros(k_num_, 1);
            auxdata.belief_AA = zeros(a_num_,k_num_);
        end

        h_nn_roundOffBelief_fn = @(x)roundOffInSimplex(x,[],h_nn_cells);
        [~,P_HIdxkn1] = h_nn_roundOffBelief_fn(P_H0);

        for k_in_horizon=1:k_num_
            P_Uk = P_Ukg_PHIdxkn1{P_HIdxkn1};
            x_k_idx_obs = x_k_idxs_(k_in_horizon);
            h_k_idx = h_k_idxs_(k_in_horizon);
            s_k_idx = XHAn1_2S_(x_k_idx_obs,h_k_idx,a_kn1_idx);

            actionInfo.belief_trans_info = [];
            env_.setState({P_Akn1, s_k_idx});
            [P_Ak, Reward, ~, LoggedSignals] = env_.step(P_Uk,actionInfo);

            Hhk_idx = LoggedSignals.Hhk_idx;
            d_k_idx_star = LoggedSignals.Dk_idx;
            yc_k_idx_star = LoggedSignals.YkIdx;
            z_k_idx = LoggedSignals.Zk_idx;

            PHidxk = possible_PHIdxk_given_Yck_PHIdxkn1(yc_k_idx_star,P_HIdxkn1);
            Hhk_idx_UA = h_nn_cells_(PHidxk).Data.HhIdx;

            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(h_k_idx,Hhk_idx);
            bayesian_reward_ = bayesian_reward_ - Reward;
            y_k_idx_obs = round(((x_k_idx_obs+params_.x_offset)*params_.x_p_pu + (d_k_idx_star+params_.d_offset)*params_.d_p_pu)/params_.y_p_pu - params_.y_offset);
            modifiedSMdata_(k_in_horizon) = params_.minPowerDemandInW + (y_k_idx_obs + params_.y_offset)*params_.y_p_pu;
            estimatedhypStateData_(k_in_horizon) = Hhk_idx;
            estimatedhypStateData_UA_(k_in_horizon) = Hhk_idx_UA;

            mean_correct_detection_UA_t =mean_correct_detection_UA_t + C_HgHh_homogeneous(h_k_idx, Hhk_idx_UA);
            bayesian_reward_UA_t = bayesian_reward_UA_t + C_HgHh_design(h_k_idx, Hhk_idx_UA);

            if(storeAuxData)
                auxdata.d_k_idxs(k_in_horizon) = d_k_idx_star;
                auxdata.yc_k_idxs(k_in_horizon) = yc_k_idx_star;
                auxdata.z_k_idxs(k_in_horizon) = z_k_idx;
            end
            if(storeBeliefs)
                auxdata.belief_Idx_UA(k_in_horizon) = P_HIdxkn1;
                auxdata.belief_AA(:,k_in_horizon) = P_Akn1;
            end

            P_Akn1 = P_Ak;
            a_kn1_idx = HZ2A_(h_k_idx,z_k_idx);
            P_HIdxkn1 = PHidxk;
        end
        auxdata.estimatedhypStateData = estimatedhypStateData_;
        auxdata.estimatedhypStateData_UA = estimatedhypStateData_UA_;
        auxdata.modifiedSMdata = modifiedSMdata_;
    end
end

