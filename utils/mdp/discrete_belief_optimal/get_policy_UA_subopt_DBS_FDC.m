function [policy] = get_policy_UA_subopt_DBS_FDC(params_in,max_valueFnIterations,value_iter_conv_threshold,PP_data_filename,fileNamePrefix, pp_data)
PP_data_params_fields = fieldnames(load(PP_data_filename,'params').('params'));
params = struct;
used_fieldnames = {'y_control_num','a_num','P_HgA','C_HgHh_design','discountFactor','s_num'};
for fn = union(used_fieldnames,PP_data_params_fields)'
    params.(fn{1}) = params_in.(fn{1});
end

possible_PHIdxk_given_Yck_PHIdxkn1 = [];
worketData = [];
h_dbs_count=[];

[policy_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    load(policy_fileFullPath,'policy');
    valueFnIterationsComplete = policy.iter_idx;
    if(max_valueFnIterations>valueFnIterationsComplete)
        for iter_idx  = valueFnIterationsComplete + 1:max_valueFnIterations
            if(policy.max_val_inc<=value_iter_conv_threshold)
                policy.isTerminated = false;
                return;
            end
            policy = iterate_policy_UA_subopt_DBS_NDRC_FDC(PP_data_filename,policy);
            save(policy_fileFullPath,'policy','params')
        end
    else
        if(policy.max_val_inc<=value_iter_conv_threshold)
            policy.isTerminated = false;
            return;
        end
    end
else
    policy = iterate_policy_UA_subopt_DBS_NDRC_FDC(PP_data_filename,[]);
    save(policy_fileFullPath,'policy','params')
    for iter_idx  = 2:max_valueFnIterations
        if(policy.max_val_inc<=value_iter_conv_threshold)
            policy.isTerminated = false;
            return;
        end
        policy = iterate_policy_UA_subopt_DBS_NDRC_FDC(PP_data_filename,policy);
        save(policy_fileFullPath,'policy','params')
    end    
end
policy.isTerminated = ~policy.isConverged;

%% Supporting functions

    function [policy_i] = iterate_policy_UA_subopt_DBS_NDRC_FDC(PP_data_filename,policy_in1)
        if isempty(worketData)
            cached_data = load(PP_data_filename);
            h_nn_cells = cached_data.('h_nn_cells');
            possible_PHIdxk_given_Yck_PHIdxkn1 = cached_data.('possible_PHIdxk_given_Yck_PHIdxkn1');
            P_Skp1gYk_Sk = cached_data.('P_Skp1gYk_Sk');
            h_dbs_count = length(h_nn_cells);
            delete cached_data

            worketData = struct;
            worketData.params = params;
            worketData.h_nn_cells = h_nn_cells;
            worketData.function_handles = get_vectorizing_function_handles(params);
            worketData.valid_YgXZn1 = pp_data.('valid_YgXZn1');
            worketData.valid_DgXZn1 = pp_data.('valid_DgXZn1');
            worketData.P_Skp1gYk_Sk = P_Skp1gYk_Sk;
            delete P_Skp1gYk_Sk
        end

        s_num = params.s_num;

        if(isempty(policy_in1))
            iter_idx_t = 1;
            valueFunction_in1 = zeros(s_num, h_dbs_count);
        else
            iter_idx_t = policy_in1.iter_idx + 1;
            valueFunction_in1 = policy_in1.valueFunction;
        end
        

        %% Value Iteration
        emu_strategy_i = cell(1,h_dbs_count);
        valueFunction_i = inf(s_num, h_dbs_count);

        [~,~] = evalc('gcp;');
        [progressData, progressDataQueue] = ProgressData(sprintf('\t\tPerforming value iteration # %d --',iter_idx_t));
        incPercent = (1/h_dbs_count)*100;
        internal_routine_fn = @internal_routine;

        worketData.valueFunction_in1 = valueFunction_in1;
        parfor PHIdxkn1 = 1:h_dbs_count
            [emu_strategy_i{PHIdxkn1},valueFunction_i(:,PHIdxkn1)] =...
                feval(internal_routine_fn, worketData, possible_PHIdxk_given_Yck_PHIdxkn1(:,PHIdxkn1),progressDataQueue,incPercent); %#ok<FVAL>
        end

        valueFunction_diff = valueFunction_i - valueFunction_in1;
        max_val_inc = max(valueFunction_diff(:));
        isConverged = max_val_inc<=value_iter_conv_threshold ;
        progressData.terminate(sprintf('Max value inc.:%f',max_val_inc));
        

        %% Save policy data
        policy_i = struct;
        policy_i.iter_idx = iter_idx_t;
        policy_i.emu_strategy = emu_strategy_i;
        policy_i.valueFunction = valueFunction_i;
        policy_i.max_val_inc = max_val_inc;
        policy_i.isConverged = isConverged;
    end


    function [P_Uk, valueFunction_i] =...
            internal_routine(worketData, possible_PHIdxk_given_Yck,progressDataQueue,incPercent)
        valueFunction_in1 =  worketData.valueFunction_in1;
        h_nn_cells_ = worketData.h_nn_cells;
        function_handles = worketData.function_handles;
        valid_YgXZn1 = worketData.valid_YgXZn1;
        params_ = worketData.params;
        P_Skp1gYk_Sk= worketData.P_Skp1gYk_Sk;
        y_control_num = params_.y_control_num;
        s_num = params_.s_num;
        u_num = params_.u_num;
        C_HgHh_design = params_.C_HgHh_design;
        discountFactor =  params_.discountFactor;

        Data = [h_nn_cells_.Data];
        HhkIdxs = [Data.HhIdx];

        S2XHAn1 = function_handles.S2XHAn1;
        A2HZ = function_handles.A2HZ;
        YcS_2U= function_handles.YcS_2U;

        P_Uk_ = zeros(1,s_num);
        valueFunction_i = inf(s_num, 1);
        parfor sk_idx = 1:s_num
            valid_YgXZn1_ = valid_YgXZn1;
            possible_PHIdxk_given_Yck_ = possible_PHIdxk_given_Yck;
            HhkIdxs_t = HhkIdxs;
            C_HgHh_design_ = C_HgHh_design;
            valueFunction_in1_ =valueFunction_in1;

            [x_k_idx_obs, h_k_idx, a_kn1_idx] = feval(S2XHAn1, sk_idx);
            [~,z_kn1_idx] = feval(A2HZ, a_kn1_idx);
            alphaVector_k = inf(y_control_num,1);
            valid_YIdxs = valid_YgXZn1_{x_k_idx_obs, z_kn1_idx};
            valid_YIdxs = valid_YIdxs(:)';
            P_Skp1gYk_ = (P_Skp1gYk_Sk{sk_idx});
            PHidxks = possible_PHIdxk_given_Yck_(valid_YIdxs);
            HhkIdxs_ = HhkIdxs_t(PHidxks);
            alphaVector_k(valid_YIdxs) = C_HgHh_design_(h_k_idx,HhkIdxs_) + discountFactor*sum(valueFunction_in1_(:,PHidxks).*P_Skp1gYk_(:,valid_YIdxs),1);            
            [min_val, opt_y_idx] = min(alphaVector_k);
            if isinf(min_val)
                error('min_val is inf')
            else
                P_Uk_(sk_idx) = opt_y_idx;
                valueFunction_i(sk_idx) = min_val;
            end
            send(progressDataQueue, incPercent/s_num);
        end

        P_Uk = zeros(u_num,1);
        P_Uk(YcS_2U(P_Uk_,1:s_num)) = 1;
        P_Uk = sparse(P_Uk);
    end
end

