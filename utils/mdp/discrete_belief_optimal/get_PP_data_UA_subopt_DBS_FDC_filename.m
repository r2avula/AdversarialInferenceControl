function [fileFullPath] = get_PP_data_UA_subopt_DBS_FDC_filename(params_in,fileNamePrefix,in_debug_mode, pp_data, useparpool) %#ok<INUSD>
params = struct;
used_fieldnames = {'P_H0','paramsPrecision','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num','x_p_pu','x_offset',...
    'P_HgA','C_HgHh_design','beliefSpacePrecision_EMU_subopt_DBS','minLikelihoodFilter','y_control_p_pu',...
    'y_control_num','y_control_offset','d_num','d_p_pu','u_num','s_num'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end


[fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(~fileExists)
    fprintf('\t\tPre processing data not found in cache...\n');
    function_handles = get_vectorizing_function_handles(params);
    beliefSpacePrecision_EMU_subopt_DBS = params.beliefSpacePrecision_EMU_subopt_DBS;
    h_num = params.h_num;
    y_control_num = params.y_control_num;

    %% Belief space discretization
    prob_int_sum = floor(1/beliefSpacePrecision_EMU_subopt_DBS);
    prob_temp1 = nchoosek(1:(prob_int_sum+h_num-1), h_num-1);
    prob_ndividers = size(prob_temp1, 1);
    prob_temp2 = cat(2, zeros(prob_ndividers, 1), prob_temp1, (prob_int_sum+h_num)*ones(prob_ndividers, 1));
    h_dbs = beliefSpacePrecision_EMU_subopt_DBS*(diff(prob_temp2, 1, 2) - 1)';
    h_dbs_count = size(h_dbs,2);

    h_nn_cells = get_NN_ClassifyingConstraints(h_dbs,beliefSpacePrecision_EMU_subopt_DBS);

    DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);
    get_adv_guess_g_belief_k_fn = @(x)getHypothesisGuess(x,DRs_in_Rn);

    %% Possible adversarial DR intersections
    for PHkIdx = 1:h_dbs_count
        h_nn_cells(PHkIdx).Data.HhIdx = get_adv_guess_g_belief_k_fn(h_dbs(:,PHkIdx));
    end
    params_t = params;
    params_t.DRs_in_Rn = DRs_in_Rn;

    %% Possible belief transitions map
    possible_PHIdxk_given_Yck_PHIdxkn1 = zeros(y_control_num,h_dbs_count);
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing possible belief transitions : ');
    incPercent = (1/h_dbs_count)*100;
    internal_routine_fn = @internal_routine;
    [~,p_pool] = evalc('gcp(''nocreate'');');


    S2XHAn1 = function_handles.S2XHAn1;
    A2HZ = function_handles.A2HZ;
    XsHAn1_2S = function_handles.XsHAn1_2S;
    HZ2A= function_handles.HZ2A;
    s_num = params.s_num;
    x_num= params.x_num;
    P_XHgHn1_ = params.P_XHgHn1;
    P_Zp1gZD = params.P_Zp1gZD;
    P_Skp1gYk_Sk = cell(1, s_num);
    valid_YgXZn1 = pp_data.('valid_YgXZn1');
    valid_DgXZn1 = pp_data.('valid_DgXZn1');

    worketData = struct;
    worketData.h_nn_cells = h_nn_cells;
    worketData.h_dbs = h_dbs;
    worketData.params = params_t;
    worketData.function_handles = function_handles;
    if isempty(p_pool) || ~useparpool
        for PHIdxkn1 = 1:h_dbs_count
            possible_PHIdxk_given_Yck_PHIdxkn1(:,PHIdxkn1) = feval(internal_routine_fn, worketData, PHIdxkn1); %#ok<FVAL>
            send(progressDataQueue, incPercent);
        end

        for sk_idx = 1:s_num
            P_Skp1gYk_ = zeros(s_num, y_control_num);
            [x_k_idx_obs, h_k_idx, a_kn1_idx] = feval(S2XHAn1,sk_idx);
            [~,z_kn1_idx] = feval(A2HZ,a_kn1_idx);
            valid_YIdxs = valid_YgXZn1{x_k_idx_obs, z_kn1_idx};
            valid_DIdxs = valid_DgXZn1{x_k_idx_obs, z_kn1_idx};
            for Yck_idx_ = valid_YIdxs(:)'
                P_Zk = P_Zp1gZD(:,z_kn1_idx,valid_DIdxs(Yck_idx_));
                valid_z_kidxs = find(P_Zk'>0);
                for zk_idx= valid_z_kidxs
                    for hkp1_idx = 1:h_num
                        skp1_idxs = feval(XsHAn1_2S,1:x_num, hkp1_idx, feval(HZ2A,h_k_idx, zk_idx));
                        P_Skp1gYk_(skp1_idxs,Yck_idx_) = P_XHgHn1_(:, hkp1_idx, h_k_idx)*P_Zk(zk_idx);
                    end
                end
            end
            P_Skp1gYk_Sk{sk_idx} = sparse(P_Skp1gYk_);
        end
    else
        [~,~] = evalc('gcp;');
        worketData = parallel.pool.Constant(worketData);
        for PHIdxkn1 = 1:h_dbs_count
            possible_PHIdxk_given_Yck_PHIdxkn1(:,PHIdxkn1) = feval(internal_routine_fn, worketData.Value, PHIdxkn1); %#ok<FVAL>
            send(progressDataQueue, incPercent);
        end

        parfor sk_idx = 1:s_num
            P_Skp1gYk_ = zeros(s_num, y_control_num);
            valid_YgXZn1_ = valid_YgXZn1;
            valid_DgXZn1_ = valid_DgXZn1;
            P_XHgHn1_t = P_XHgHn1_;
            P_Zp1gZD_ = P_Zp1gZD;
            [x_k_idx_obs, h_k_idx, a_kn1_idx] = feval(S2XHAn1,sk_idx);
            [~,z_kn1_idx] = feval(A2HZ,a_kn1_idx);
            valid_YIdxs = valid_YgXZn1_{x_k_idx_obs, z_kn1_idx};
            valid_DIdxs = valid_DgXZn1_{x_k_idx_obs, z_kn1_idx};
            for Yck_idx_ = valid_YIdxs(:)'
                P_Zk = P_Zp1gZD_(:,z_kn1_idx,valid_DIdxs(Yck_idx_));
                valid_z_kidxs = find(P_Zk'>0);
                for zk_idx= valid_z_kidxs
                    for hkp1_idx = 1:h_num
                        skp1_idxs = feval(XsHAn1_2S,1:x_num, hkp1_idx, feval(HZ2A,h_k_idx, zk_idx));
                        P_Skp1gYk_(skp1_idxs,Yck_idx_) = P_XHgHn1_t(:, hkp1_idx, h_k_idx)*P_Zk(zk_idx);
                    end
                end
            end
            P_Skp1gYk_Sk{sk_idx} = sparse(P_Skp1gYk_);
        end
    end
    progressData.terminate();


    %% Store pre processing
    save(fileFullPath,'params',...
        getVarName(h_nn_cells),...
        getVarName(P_Skp1gYk_Sk),...
        getVarName(possible_PHIdxk_given_Yck_PHIdxkn1));
    fprintf('\t\tPre processing complete. Data saved in: %s\n',fileFullPath);
end


    function [PHIdxks] = internal_routine(worketData, PHIdxkn1)
        h_dbs_ = worketData.h_dbs;
        h_nn_cells_ = worketData.h_nn_cells;
        P_Hkn1 = h_dbs_(:,PHIdxkn1);
        params_ = worketData.params;
        y_control_num_ = params_.y_control_num;
        y_control_range = 1:y_control_num_;
        h_num_ = params_.h_num;
        P_H0 = params_.P_H0;
        P_XHgHn1 = params_.P_XHgHn1;
        minLikelihoodFilter = params_.minLikelihoodFilter;
        paramsPrecision = params_.paramsPrecision;
        roundOffBelief_beliefSpacePrecision_fn = @(x)roundOffInSimplex(x,paramsPrecision,h_nn_cells_);

        PHIdxks = nan(y_control_num_,1);
        P_YksgY12kn1 = zeros(y_control_num_, 1);
        for Yck_idx = y_control_range
            P_Hk = reshape(P_XHgHn1(Yck_idx,:,:), h_num_, h_num_)*P_Hkn1;
            P_Hk_sum = sum(P_Hk);
            if P_Hk_sum>minLikelihoodFilter
                [~,PHIdxkn1] = roundOffBelief_beliefSpacePrecision_fn(P_Hk/P_Hk_sum);
                PHIdxks(Yck_idx) = PHIdxkn1;
                P_YksgY12kn1(Yck_idx) = P_Hk_sum;
            end
        end
        infeasible_y_idxs_flag = isnan(PHIdxks);
        if(any(infeasible_y_idxs_flag))
            [P_Hk_sum_max_t,Yck_idx_t] = max(P_YksgY12kn1);
            if(P_Hk_sum_max_t>=minLikelihoodFilter)
                PHIdxks(infeasible_y_idxs_flag) = PHIdxks(Yck_idx_t);
            else
                [~,PHIdx0] = roundOffBelief_beliefSpacePrecision_fn(P_H0);
                PHIdxks(infeasible_y_idxs_flag) = PHIdx0;
            end
        end
    end
end

