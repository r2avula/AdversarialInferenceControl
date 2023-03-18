function [policy] = get_policy_subopt_DBS_FDC(params_in,max_valueFnIterations,value_iter_conv_threshold,PP_data_filename,fileNamePrefix, pp_data, useparpool)
PP_data_params_fields = fieldnames(load(PP_data_filename,'params').('params'));
params = struct;
used_fieldnames = {'y_control_num','a_num','P_HgA','C_HgHh_design','discountFactor'};
for fn = union(used_fieldnames,PP_data_params_fields)'
    params.(fn{1}) = params_in.(fn{1});
end

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
            policy = iterate_policy_subopt_DBS_NDRC_FDC(PP_data_filename,policy);
            save(policy_fileFullPath,'policy','params')
        end
    else
        if(policy.max_val_inc<=value_iter_conv_threshold)
            policy.isTerminated = false;
            return;
        end
    end
else
    policy = iterate_policy_subopt_DBS_NDRC_FDC(PP_data_filename,[]);
    save(policy_fileFullPath,'policy','params')
    for iter_idx  = 2:max_valueFnIterations
        if(policy.max_val_inc<=value_iter_conv_threshold)
            policy.isTerminated = false;
            return;
        end
        policy = iterate_policy_subopt_DBS_NDRC_FDC(PP_data_filename,policy);
        save(policy_fileFullPath,'policy','params')
    end    
end
policy.isTerminated = ~policy.isConverged;

%% Supporting functions

    function [policy_i] = iterate_policy_subopt_DBS_NDRC_FDC(PP_data_filename,policy_in1)
        y_control_num = params.y_control_num;
        function_handles = get_vectorizing_function_handles(params);
        A2HZ = function_handles.A2HZ;
        YcsS_2U = function_handles.YcsS_2U;
        u_num = params.u_num;

        %% Pre processing
        cached_data = load(PP_data_filename);
        a_nn_cells = cached_data.('a_nn_cells');
        possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1 = cached_data.('possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1');
        P_AgU_YcAkn1 = pp_data.('P_AgU_YcAkn1');
        valid_XgYZn1 = pp_data.('valid_XgYZn1');
        schedulable_Yc_idx_PAIdxkn1_flag = cached_data.('schedulable_Yc_idx_PAIdxkn1_flag');
        suppressable_Yc_idxs_PAIdxkn1_flag = cached_data.('suppressable_Yc_idxs_PAIdxkn1_flag');
        DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);
        a_dbs_count = length(a_nn_cells);
        params_ = params;
        params_.valid_XgYZn1 = valid_XgYZn1;

        if(isfield(cached_data,'osg_PU'))
            osg_PU = cached_data.osg_PU;
        else
            osg_PU = cell(a_dbs_count,1);
        end
        init_osg_PU_empty_cells = sum(cellfun(@isempty,osg_PU));

        [gurobi_model,gurobi_model_params,Aeq_cons,beq_cons] = get_gurobi_model_FDC(params_,function_handles);

        if(isempty(policy_in1))
            iter_idx_t = 1;
            valueFunction_in1 = zeros(a_dbs_count,1);
        else
            iter_idx_t = policy_in1.iter_idx + 1;
            valueFunction_in1 = policy_in1.valueFunction;
        end

        %% Value Iteration
        emu_strategy_i = zeros(u_num,a_dbs_count);
        valueFunction_i = inf(a_dbs_count,1);
        beliefTransitionMap_i = nan(y_control_num,a_dbs_count);

        OSG_strategy_params = params;
        OSG_strategy_params.gurobi_model = gurobi_model;
        OSG_strategy_params.gurobi_model_params = gurobi_model_params;
        OSG_strategy_params.Aeq_cons = Aeq_cons;
        OSG_strategy_params.beq_cons = beq_cons;
        OSG_strategy_params.P_AgU_YcAkn1 = P_AgU_YcAkn1;
        OSG_strategy_params.DRs_in_Rn = DRs_in_Rn;

        DBS_strategy_params = params;
        DBS_strategy_params.a_nn_cells = a_nn_cells;
        DBS_strategy_params.P_AgU_YcAkn1 = P_AgU_YcAkn1;
        DBS_strategy_params.gurobi_model = gurobi_model;
        DBS_strategy_params.gurobi_model_params = gurobi_model_params;
        DBS_strategy_params.Aeq_cons = Aeq_cons;
        DBS_strategy_params.beq_cons = beq_cons;
        DBS_strategy_params.A2HZ = A2HZ;
        DBS_strategy_params.YcsS_2U = YcsS_2U;


        [progressData, progressDataQueue] = ProgressData(sprintf('\t\tPerforming value iteration # %d --',iter_idx_t));
        incPercent = (1/a_dbs_count)*100;
        internal_routine_fn = @internal_routine;
        [~,p_pool] = evalc('gcp(''nocreate'');');

        worketData = struct;
        worketData.DBS_strategy_params = DBS_strategy_params;
        worketData.OSG_strategy_params = OSG_strategy_params;
        worketData.a_nn_cells = a_nn_cells;
        worketData.valueFunction_in1 = valueFunction_in1;

        if isempty(p_pool)||useparpool
            for PAkn1_idx = 1:a_dbs_count
                osg_PU_ = osg_PU{PAkn1_idx};
                [osg_PU_, emu_strategy_i(:,PAkn1_idx),valueFunction_i(PAkn1_idx), beliefTransitionMap_i(:,PAkn1_idx)] =...
                    feval(internal_routine_fn, worketData, PAkn1_idx, osg_PU_, ...
                    schedulable_Yc_idx_PAIdxkn1_flag(:,PAkn1_idx), suppressable_Yc_idxs_PAIdxkn1_flag(:,PAkn1_idx),...
                    possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1{PAkn1_idx}); %#ok<FVAL>
                osg_PU{PAkn1_idx} = osg_PU_;
                send(progressDataQueue, incPercent);
            end
        else
            worketData = parallel.pool.Constant(worketData);
            parfor PAkn1_idx = 1:a_dbs_count
                osg_PU_ = osg_PU{PAkn1_idx};
                [osg_PU_, emu_strategy_i(:,PAkn1_idx),valueFunction_i(PAkn1_idx), beliefTransitionMap_i(:,PAkn1_idx)] =...
                    feval(internal_routine_fn, worketData.Value,PAkn1_idx, osg_PU_, ...
                    schedulable_Yc_idx_PAIdxkn1_flag(:,PAkn1_idx), suppressable_Yc_idxs_PAIdxkn1_flag(:,PAkn1_idx),...
                    possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1{PAkn1_idx}); %#ok<FVAL>
                osg_PU{PAkn1_idx} = osg_PU_;
                send(progressDataQueue, incPercent);
            end
        end

        valueFunction_diff = valueFunction_i - valueFunction_in1;
        max_val_inc = max(valueFunction_diff);
        isConverged = max_val_inc<=value_iter_conv_threshold ;
        progressData.terminate(sprintf('Max value inc.:%f',max_val_inc));
        

        %% Save policy data
        policy_i = struct;
        policy_i.iter_idx = iter_idx_t;
        policy_i.emu_strategy = emu_strategy_i;
        policy_i.valueFunction = valueFunction_i;
        policy_i.beliefTransitionMap = beliefTransitionMap_i;
        policy_i.max_val_inc = max_val_inc;
        policy_i.isConverged = isConverged;

        if(sum(cellfun(@isempty,osg_PU))<init_osg_PU_empty_cells)
            cached_data.osg_PU = osg_PU;
            save(PP_data_filename, '-struct', 'cached_data');
        end
    end


    function [osg_PU_, emu_strategy_i, valueFunction_i, beliefTransitionMap_i] =...
            internal_routine(worketData, PAkn1_idx, osg_PU_,...
            schedulable_yc_idxs_flag, suppressable_yc_idxs_flag, possible_PAIdx_given_Yc_vecs)
        DBS_strategy_params_ = worketData.DBS_strategy_params;
        OSG_strategy_params = worketData.DBS_strategy_params;
        valueFunction_in1 =  worketData.valueFunction_in1;
        a_nn_cells_ = worketData.a_nn_cells;
        y_control_num = DBS_strategy_params_.y_control_num;
        paramsPrecision = DBS_strategy_params_.paramsPrecision;
        beliefSpacePrecision_EMU_subopt_DBS = DBS_strategy_params_.beliefSpacePrecision_EMU_subopt_DBS;
        x_num = DBS_strategy_params_.x_num;
        h_num = DBS_strategy_params_.h_num;
        a_num = DBS_strategy_params_.a_num;
        C_HgHh_design = DBS_strategy_params_.C_HgHh_design;
        P_HgA = DBS_strategy_params_.P_HgA;
        discountFactor =  DBS_strategy_params_.discountFactor;
        y_control_range = 1:y_control_num;
        s_num = x_num*h_num*a_num;
        u_num = y_control_num*s_num;
        
        a_nn_roundOffBelief_fn = @(x)roundOffInSimplex(x,[],a_nn_cells_);
        roundOffBelief_beliefSpacePrecision_fn = @(x)roundOffInSimplex(x,beliefSpacePrecision_EMU_subopt_DBS);
        roundOffBelief_paramsPrecision_fn = @(x)roundOffInSimplex(x,paramsPrecision);

        P_Akn1 = a_nn_cells_(PAkn1_idx).Data.randomInteriorPoint;
        DBS_strategy_params_.schedulable_yc_idxs_flag = schedulable_yc_idxs_flag;
        DBS_strategy_params_.suppressable_yc_idxs_flag = suppressable_yc_idxs_flag;
        DBS_strategy_params_.P_Akn1 = P_Akn1;

        DBS_strategy_params_.possible_PAIdx_given_Yc_vecs = possible_PAIdx_given_Yc_vecs;
        [strategy_data,P_AgU_given_Yc,P_YcgU_sum] = compute_strategy_subopt_DBS_FDC(DBS_strategy_params_,valueFunction_in1);

        if(isempty(strategy_data))
            if(isempty(osg_PU_))
                [osg_PU_] = computeStrategy_OSG_FDC(P_Akn1,OSG_strategy_params,false, useparpool);
            end

            alpha_vector_t = zeros(1,u_num);
            for Yk_idx = y_control_range
                P_AgU = P_AgU_given_Yc{Yk_idx};
                P_YgU = sum(P_AgU,1);
                P_Ak = P_AgU*osg_PU_;
                P_Ak_sum = sum(P_Ak);
                if(P_Ak_sum>=paramsPrecision)
                    P_Ak = P_Ak/P_Ak_sum;
                    P_Ak = roundOffBelief_beliefSpacePrecision_fn(roundOffBelief_paramsPrecision_fn(P_Ak));
                    [~,PAk_idx] = a_nn_roundOffBelief_fn(P_Ak);
                    Hhk_idx = a_nn_cells_(PAk_idx).Data.HhIdx;
                    alpha_vector_t = alpha_vector_t + C_HgHh_design(Hhk_idx,:)*P_HgA*P_AgU + discountFactor*valueFunction_in1(PAk_idx)*P_YgU;
                end
            end

            PAIdx_given_Y = nan(y_control_num,1);
            for Yk_idx = y_control_range
                P_AgU = P_AgU_given_Yc{Yk_idx};
                P_Ak = P_AgU*osg_PU_;
                P_Ak_sum = sum(P_Ak);
                if(P_Ak_sum>=paramsPrecision)
                    P_Ak = P_Ak/P_Ak_sum;
                    P_Ak = roundOffBelief_beliefSpacePrecision_fn(roundOffBelief_paramsPrecision_fn(P_Ak));
                    [~,PAk_idx] = a_nn_roundOffBelief_fn(P_Ak);
                    PAIdx_given_Y(Yk_idx) = PAk_idx;
                end
            end

            if(any(isnan(PAIdx_given_Y)))
                P_Ak_sum_t = zeros(y_control_num,1);
                for Yk_idx = y_control_range
                    P_Ak_sum_t(Yk_idx) = sum(P_AgU_given_Yc{Yk_idx}*osg_PU_);
                end
                [P_Ak_sum_max_t,Yk_idx_t] = max(P_Ak_sum_t);
                if(P_Ak_sum_max_t>=paramsPrecision)
                    PAIdx_given_Y(isnan(PAIdx_given_Y)) = PAIdx_given_Y(Yk_idx_t);
                else
                    error('opt_mismatch')
                end
            end

            emu_strategy_i = osg_PU_;
            valueFunction_i = roundOff(alpha_vector_t*osg_PU_/(P_YcgU_sum*osg_PU_),paramsPrecision);
            beliefTransitionMap_i = PAIdx_given_Y;
        else
            emu_strategy_i = strategy_data.P_U;
            valueFunction_i = strategy_data.min_value;
            beliefTransitionMap_i = strategy_data.PAIdx_given_Y;
        end
    end
end

