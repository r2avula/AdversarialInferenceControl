function [optimalControlData] = simulate_OSG_FDC(evalParams,cache_fileName, pp_data_, useparpool)
storeAuxData = evalParams.storeAuxData;
storeBeliefs = evalParams.storeBeliefs;
sm_data = evalParams.sm_data;
gt_data = evalParams.gt_data;
h_0_idxs = evalParams.h_0_idxs;
numHorizons = evalParams.numHorizons;
params = evalParams.params;
max_cache_size = evalParams.max_cache_size;
in_debug_mode = evalParams.in_debug_mode;

x_num = params.x_num;
h_num = params.h_num;
a_num = params.a_num;
k_num = params.k_num;
minPowerDemandInW = params.minPowerDemandInW;

if(isfile(cache_fileName))
    cached_data = load(cache_fileName,'osg_strategy','valid_DgYZn1','valid_XgYZn1');
    osg_strategy = cached_data.('osg_strategy');
    valid_DgYZn1 = cached_data.('valid_DgYZn1');
    valid_XgYZn1 = cached_data.('valid_XgYZn1');
else
    osg_strategy = containers.Map;
    x_num = params.x_num;
    y_control_num = params.y_control_num;
    y_control_p_pu = params.y_control_p_pu;
    y_control_offset = params.y_control_offset;
    x_p_pu = params.x_p_pu;
    x_offset = params.x_offset;
    d_p_pu = params.d_p_pu;
    d_offset = params.d_offset;
    P_Zp1gZD = params.P_Zp1gZD;
    paramsPrecision = params.paramsPrecision;
    d_num = params.d_num;
    z_num = params.z_num;
    y_control_range = 1:y_control_num;
    x_range = 1:x_num;

    valid_DgYZn1 = cell(y_control_num, z_num);
    valid_XgYZn1 = cell(y_control_num, z_num);
    for z_kn1_idx_ = 1:z_num
        P_Zp1g_D = reshape(P_Zp1gZD(:,z_kn1_idx_,:),z_num,d_num);
        for Yck_idx = y_control_range
            valid_DIdxs = round(((Yck_idx+y_control_offset)*y_control_p_pu - (x_range+x_offset)*x_p_pu)/d_p_pu) - d_offset;
            valid_XIdxs_flag = valid_DIdxs>=1 & valid_DIdxs<=d_num;
            valid_XIdxs = x_range(valid_XIdxs_flag);
            valid_DIdxs = valid_DIdxs(valid_XIdxs_flag);
            valid_DIdxs_flag = sum(P_Zp1g_D(:,valid_DIdxs),1)>=paramsPrecision;
            valid_DIdxs = valid_DIdxs(valid_DIdxs_flag);
            valid_XIdxs = valid_XIdxs(valid_DIdxs_flag);

            valid_DgYZn1{Yck_idx, z_kn1_idx_} = valid_DIdxs;
            valid_XgYZn1{Yck_idx, z_kn1_idx_} = valid_XIdxs;
        end
    end
end

osg_strategy_dataSet_size = osg_strategy.Count;
osg_strategy_data_keySet = cell(max_cache_size,1);
osg_strategy_data_keySet(1:osg_strategy_dataSet_size) = keys(osg_strategy);
osg_strategy_data_keySet(osg_strategy_dataSet_size+1:max_cache_size) = {''};
osg_strategy_data_valueSet = cell(max_cache_size,1);
osg_strategy_data_valueSet(1:osg_strategy_dataSet_size) = values(osg_strategy);

params.valid_DgYZn1 = valid_DgYZn1;
params.valid_XgYZn1 = valid_XgYZn1;


%% Initialization
LoggedSignalsForExpStruct = ["AdversarialRewardEstimate", "Hhk_idx", "Dk_idx", "YkIdx","Zk_idx","P_YksgY12kn1"];
env = SmartGridUserEnv_FD(params, @loopBackFcn, pp_data_, false, LoggedSignalsForExpStruct);

[gurobi_model,gurobi_model_params,Aeq_cons,beq_cons] = get_gurobi_model_FDC(params, env.Params.Function_handles);
OSG_strategy_params = env.Params;
OSG_strategy_params.gurobi_model = gurobi_model;
OSG_strategy_params.gurobi_model_params = gurobi_model_params;
OSG_strategy_params.Aeq_cons = Aeq_cons;
OSG_strategy_params.beq_cons = beq_cons;

y_control_p_pu = params.y_control_p_pu;
y_control_offset = params.y_control_offset;
x_p_pu = params.x_p_pu;
x_offset = params.x_offset;

u_num = env.Params.u_num;
h_range = 1:h_num;
a_range = 1:a_num;

XHsAn1s_2S = env.Params.Function_handles.XHsAn1s_2S;
YcSs_2U = env.Params.Function_handles.YcSs_2U;
P_Uk_idle = zeros(u_num, 1);
for x_k_idx = 1:x_num
    s_k_idxs = XHsAn1s_2S(x_k_idx,h_range',a_range');
    yc_idx = round((x_k_idx+x_offset)*x_p_pu/y_control_p_pu) - y_control_offset;
    P_Uk_idle(YcSs_2U(yc_idx,s_k_idxs)) = 1;
end

OSG_strategy_params.P_Uk_idle = P_Uk_idle;
OSG_strategy_params.P_AgU_YcAkn1 = env.P_AgU_YcAkn1;

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
    belief_states = zeros(a_num,k_num,numHorizons);
end

osg_strategy_Queue = parallel.pool.DataQueue;
afterEach(osg_strategy_Queue, @update_cache);

[progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;

worketData = struct;
worketData.env = env;
worketData.OSG_strategy_params = OSG_strategy_params;
worketData.in_debug_mode = in_debug_mode;
worketData.storeAuxData = storeAuxData;
worketData.storeBeliefs = storeBeliefs;

if useparpool
    parfor horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1_, estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
            feval(internal_routine_fn, worketData, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<*FVAL>

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
else
    for horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_,mean_PYkgY12kn1_, estimatedhypStateData(:, horizon_idx), modifiedSMdata(:, horizon_idx), auxdata] =...
            feval(internal_routine_fn, worketData, x_k_idxs(:, horizon_idx), h_k_idxs(:, horizon_idx), h_0_idxs(horizon_idx)); %#ok<*FVAL>

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
end

mean_correct_detection = mean_correct_detection/numHorizons;
bayesian_reward = bayesian_reward/numHorizons;
mean_PYkgY12kn1 = mean_PYkgY12kn1/numHorizons;
progressData.terminate(sprintf('Bayes risk : %f; Bayes reward : %f ',mean_correct_detection, bayesian_reward));

osg_strategy_data_keySet(osg_strategy_dataSet_size+1:end) = [];
osg_strategy_data_valueSet(osg_strategy_dataSet_size+1:end) = [];

osg_strategy = containers.Map(osg_strategy_data_keySet,osg_strategy_data_valueSet);

params = evalParams.params;
save(cache_fileName,'osg_strategy','valid_DgYZn1','valid_XgYZn1','params')

% store
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

%% Supporting functions
    function update_cache(data_sent)
        strategy_data = data_sent.P_Uk;
        key_string_t = data_sent.key_string;
        [~,keyID_t] = ismember(key_string_t,osg_strategy_data_keySet);
        if(keyID_t==0)
            if(osg_strategy_dataSet_size<max_cache_size)
                osg_strategy_dataSet_size = osg_strategy_dataSet_size + 1;
            else
                osg_strategy_data_keySet(1:end-1) = osg_strategy_data_keySet(2:end);
                osg_strategy_data_valueSet(1:end-1) = osg_strategy_data_valueSet(2:end);
            end
        elseif(keyID_t<max_cache_size)
            osg_strategy_data_keySet(keyID_t:end-1) = osg_strategy_data_keySet(keyID_t+1:end);
            osg_strategy_data_valueSet(keyID_t:end-1) = osg_strategy_data_valueSet(keyID_t+1:end);
        end
        osg_strategy_data_keySet{osg_strategy_dataSet_size} = key_string_t;
        osg_strategy_data_valueSet{osg_strategy_dataSet_size} = strategy_data;
    end

    function [P_Uk_control] = loopBackFcn(~, P_Uk,~)        
        P_Uk_control = P_Uk;
    end

    function [mean_correct_detection_, bayesian_reward_, mean_PYkgY12kn1, estimatedhypStateData_, modifiedSMdata_, auxdata] =...
            internal_routine(worketData, x_k_idxs_, h_k_idxs_, h0_idx)

        env_ = worketData.env;
        OSG_strategy_params_ = worketData.OSG_strategy_params;
        in_debug_mode_ = worketData.in_debug_mode;
        storeAuxData_ = worketData.storeAuxData;
        storeBeliefs_ = worketData.storeBeliefs;

        params_ = env_.Params;
        HZ2A_ = params_.Function_handles.HZ2A;
        XHAn1_2S_ = params_.Function_handles.XHAn1_2S;
        k_num_ = params_.k_num;
        a_num_ = params_.a_num;
        P_Akn1 = reset(env_);
        z_kn1_idx = env_.z0_idx;
        a_kn1_idx = HZ2A_(h0_idx,z_kn1_idx);
        C_HgHh_homogeneous = params_.C_HgHh_homogeneous;

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
            auxdata.belief_states = zeros(a_num_,k_num_);
        end

        for k_in_horizon=1:k_num_
            key_string = num2str(P_Akn1');
            [keyExists,keyID] = ismember(key_string,osg_strategy_data_keySet);
            if(keyExists)
                P_Uk = osg_strategy_data_valueSet{keyID};
            else
                [P_Uk,~] = computeStrategy_OSG_FDC(P_Akn1,OSG_strategy_params_,in_debug_mode_, false);

                data_to_send = struct;
                data_to_send.P_Uk = P_Uk;
                data_to_send.key_string = key_string;
                send(osg_strategy_Queue,data_to_send);
            end

            x_k_idx_obs = x_k_idxs_(k_in_horizon);
            h_k_idx_ = h_k_idxs_(k_in_horizon);
            s_k_idx = XHAn1_2S_(x_k_idx_obs,h_k_idx_,a_kn1_idx);

            action_info.belief_trans_info = [];
            env_.setState({P_Akn1, s_k_idx});
            [P_Ak, Reward, ~, LoggedSignals] = env_.step(P_Uk,action_info);

            Hhk_idx = LoggedSignals.Hhk_idx;
            d_k_idx_star = LoggedSignals.Dk_idx;
            yc_k_idx_star = LoggedSignals.YkIdx;
            z_k_idx = LoggedSignals.Zk_idx;

            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(h_k_idx_,Hhk_idx);
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
            a_kn1_idx = HZ2A_(h_k_idx_,z_k_idx);
            mean_PYkgY12kn1 = mean_PYkgY12kn1 + LoggedSignals.P_YksgY12kn1(yc_k_idx_star);
        end
        mean_PYkgY12kn1 = mean_PYkgY12kn1/k_num_;
        mean_correct_detection_ = mean_correct_detection_/k_num_;
        bayesian_reward_ = bayesian_reward_/k_num_;
    end
end

