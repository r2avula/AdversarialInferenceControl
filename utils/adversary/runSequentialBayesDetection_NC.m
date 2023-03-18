function [detectionData] = runSequentialBayesDetection_NC(params, sm_data, gt_data, useparpool)
storeBeliefs = params.storeBeliefs;
x_num = params.x_num;
y_num = params.y_num;
h_num = params.h_num;
k_num = params.k_num;
x_p_pu = params.x_p_pu;
y_p_pu = params.y_p_pu;
x_offset = params.x_offset;
y_offset = params.y_offset;
minPowerDemandInW = params.minPowerDemandInW;
P_XHgHn1 = params.P_XHgHn1;
params.DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);


P_YgX = zeros(y_num,x_num);
for x_k_idx = 1:x_num
    y_k_idx = round(((x_k_idx+x_offset)*x_p_pu)/y_p_pu - y_offset);
    P_YgX(y_k_idx,x_k_idx) = 1;
end

belief_trans_mats = zeros(h_num,h_num,y_num);
for y_k_idx = 1:y_num
    for h_kn1_idx = 1:h_num
        belief_trans_mats(:,h_kn1_idx,y_k_idx) =  reshape(P_YgX(y_k_idx,:)*P_XHgHn1(:,:,h_kn1_idx),h_num,1);
    end
end
params.belief_trans_mats = belief_trans_mats;

%% simulation
y_k_idxs = min(max(1,round((sm_data-minPowerDemandInW)/y_p_pu)-y_offset),y_num);
numHorizons = size(y_k_idxs,2);
if(storeBeliefs)
    belief_states = zeros(h_num,k_num,numHorizons);
end
estimatedhypStateData = zeros(k_num,numHorizons);

mean_correct_detection = 0;
bayesian_reward = 0;
mean_PYkgY12kn1 = 0;


cache_data_queue = parallel.pool.DataQueue;
afterEach(cache_data_queue, @update_cache);
belief_transition_cache = containers.Map;
h_hat_cache = containers.Map;      
P_YksgY12kn1_cache = containers.Map;   

[progressData, progressDataQueue] = ProgressData('\t\t\tRunning Bayesian detection without controller : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;
[~,p_pool] = evalc('gcp(''nocreate'');');
if isempty(p_pool) || ~useparpool
    for horizon_idx = 1:numHorizons
        [mean_correct_detection_, bayesian_reward_, estimatedhypStateData(:, horizon_idx), belief_states_, mean_PYkgY12kn1_] =...
            feval(internal_routine_fn, params, y_k_idxs(:, horizon_idx), gt_data(:, horizon_idx), belief_transition_cache, h_hat_cache, P_YksgY12kn1_cache); %#ok<*FVAL> 
        mean_correct_detection = mean_correct_detection + mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;
        if(storeBeliefs)
            belief_states(:, :, horizon_idx) = belief_states_;
        end
        send(progressDataQueue, incPercent);
    end
else
    params = parallel.pool.Constant(params);
    parfor horizon_idx = 1:numHorizons
        params_ = params.Value;
        [mean_correct_detection_, bayesian_reward_, estimatedhypStateData(:, horizon_idx), belief_states_, mean_PYkgY12kn1_] =...
            feval(internal_routine_fn, params_, y_k_idxs(:, horizon_idx), gt_data(:, horizon_idx), belief_transition_cache, h_hat_cache, P_YksgY12kn1_cache);
        mean_correct_detection = mean_correct_detection + mean_correct_detection_;
        bayesian_reward = bayesian_reward + bayesian_reward_;
        mean_PYkgY12kn1 = mean_PYkgY12kn1 + mean_PYkgY12kn1_;
        if(storeBeliefs)
            belief_states(:, :, horizon_idx) = belief_states_;
        end
        send(progressDataQueue, incPercent);
    end
end
mean_correct_detection = mean_correct_detection/numHorizons;
bayesian_reward = bayesian_reward/numHorizons;
mean_PYkgY12kn1 = mean_PYkgY12kn1/numHorizons;
progressData.terminate();

detectionData = struct;
if(storeBeliefs)
    detectionData.belief_states = belief_states;
end
detectionData.estimatedhypStateData = estimatedhypStateData;
detectionData.mean_correct_detection = mean_correct_detection;
detectionData.bayesian_reward = bayesian_reward;
detectionData.mean_PYkgY12kn1 = mean_PYkgY12kn1;

%% Supporting functions
    function update_cache(strategy_data)
        belief_transition_cache(strategy_data.key_string) = strategy_data.belief_ks;
        P_YksgY12kn1_cache(strategy_data.key_string) = strategy_data.P_YksgY12kn1;
        h_hat_cache(strategy_data.key_string) = strategy_data.h_hat_k_idxs;
    end

    function [mean_correct_detection_, bayesian_reward_, estimatedhypStateData_, belief_states_, mean_PYkgY12kn1] =...
            internal_routine(params, y_k_idxs_, gt_data_, belief_transition_cache, h_hat_cache, P_YksgY12kn1_cache)
        storeBeliefs_ = params.storeBeliefs;
        k_num_ = params.k_num;
        y_num_ = params.y_num;
        h_num_ = params.h_num;
        minLikelihoodFilter = params.minLikelihoodFilter;
        getHypothesisGuess_fn = @(x)getHypothesisGuess(x, params.DRs_in_Rn);
        beliefPrecision_roundOff_fn = @(x)roundOffInSimplex(x, params.beliefSpacePrecision_adv);
        paramsPrecision_roundOff_fn = @(x)roundOffInSimplex(x, params.paramsPrecision);
        P_H0 = params.P_H0;
        C_HgHh_design = params.C_HgHh_design;
        C_HgHh_homogeneous = params.C_HgHh_homogeneous;
        belief_trans_mats_ = params.belief_trans_mats;

        if(storeBeliefs_)
            belief_states_ = zeros(h_num_, k_num_,1);
        else
            belief_states_ = [];
        end
        estimatedhypStateData_ = zeros(k_num_,1);
        mean_correct_detection_ = 0;
        bayesian_reward_ = 0;
        mean_PYkgY12kn1 = 0;

        belief_kn1 = P_H0;
        for k_in_horizon = 1:k_num_
            key_string = num2str(belief_kn1');
            y_k_idx_obs = y_k_idxs_(k_in_horizon);

            if(isKey(belief_transition_cache,key_string))
                belief_ks = belief_transition_cache(key_string);
                P_YksgY12kn1 = P_YksgY12kn1_cache(key_string);
                h_hat_k_idxs = h_hat_cache(key_string);
            else
                belief_ks = cell(1,y_num_);
                h_hat_k_idxs = nan(y_num_,1);
                P_YksgY12kn1 = zeros(y_num_,1);
                for y_k_idx_ = 1:y_num_
                    belief_k_ = belief_trans_mats_(:,:,y_k_idx_)*belief_kn1;
                    P_YkgY12kn1 = sum(belief_k_);
                    P_YksgY12kn1(y_k_idx_) = P_YkgY12kn1;
                    if(P_YkgY12kn1>minLikelihoodFilter)
                        belief_k_ = belief_k_/P_YkgY12kn1;
                        belief_k_ = beliefPrecision_roundOff_fn(belief_k_);
                        belief_ks{y_k_idx_} = belief_k_;
                        h_hat_k_idxs(y_k_idx_) = getHypothesisGuess_fn(belief_k_);
                    end
                end

                P_YksgY12kn1 = paramsPrecision_roundOff_fn(P_YksgY12kn1./sum(P_YksgY12kn1));
                infeasible_y_idxs_flag = P_YksgY12kn1 < minLikelihoodFilter | isnan(h_hat_k_idxs);
                if(any(infeasible_y_idxs_flag))
                    [P_Ak_sum_max_t,Yck_idx_t] = max(P_YksgY12kn1);
                    if(P_Ak_sum_max_t>=minLikelihoodFilter)
                        h_hat_k_idxs(infeasible_y_idxs_flag) = h_hat_k_idxs(Yck_idx_t);
                        belief_ks(infeasible_y_idxs_flag) = repmat(belief_ks(Yck_idx_t), [1,sum(infeasible_y_idxs_flag)]);
                    else
                        belief_ks(infeasible_y_idxs_flag) = repmat({belief_kn1}, [1,sum(infeasible_y_idxs_flag)]);
                        h_hat_k_idxs(infeasible_y_idxs_flag) = getHypothesisGuess_fn(P_HgA*P_Akn1);
                    end
                end

                strategy_data_k = struct;
                strategy_data_k.belief_ks = belief_ks;
                strategy_data_k.P_YksgY12kn1 = P_YksgY12kn1;
                strategy_data_k.h_hat_k_idxs = h_hat_k_idxs;
                strategy_data_k.key_string = key_string;
                belief_transition_cache(key_string) = belief_ks;
                P_YksgY12kn1_cache(key_string) = P_YksgY12kn1;
                h_hat_cache(key_string) = h_hat_k_idxs;
                send(cache_data_queue, strategy_data_k);
            end
            belief_k_ = belief_ks{y_k_idx_obs};
            h_hat_k_idx = h_hat_k_idxs(y_k_idx_obs);
            mean_correct_detection_ = mean_correct_detection_ + C_HgHh_homogeneous(gt_data_(k_in_horizon),h_hat_k_idx);
            bayesian_reward_ = bayesian_reward_ + C_HgHh_design(gt_data_(k_in_horizon), h_hat_k_idx);

            if(storeBeliefs_)
                belief_states_(:,k_in_horizon) = belief_k_;
            end
            estimatedhypStateData_(k_in_horizon) = h_hat_k_idx;
            belief_kn1 = belief_k_;
            mean_PYkgY12kn1 = mean_PYkgY12kn1 + P_YksgY12kn1(y_k_idx_obs);
        end
        mean_PYkgY12kn1 = mean_PYkgY12kn1/k_num_;
        mean_correct_detection_ = mean_correct_detection_/k_num_;
        bayesian_reward_ = bayesian_reward_/k_num_;
    end
end

