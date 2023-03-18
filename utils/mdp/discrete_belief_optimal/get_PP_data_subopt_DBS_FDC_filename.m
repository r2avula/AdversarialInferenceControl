function [fileFullPath] = get_PP_data_subopt_DBS_FDC_filename(params_in,fileNamePrefix,in_debug_mode, pp_data, useparpool) %#ok<INUSD>
params = struct;
used_fieldnames = {'paramsPrecision','h_num','z_num','x_num','d_offset','P_Zp1gZD','P_XHgHn1','a_num','x_p_pu','x_offset',...
    'P_HgA','C_HgHh_design','beliefSpacePrecision_EMU_subopt_DBS','minLikelihoodFilter','y_control_p_pu','y_control_num','y_control_offset','d_num','d_p_pu','u_num','s_num'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end


[fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(~fileExists)
    fprintf('\t\tPre processing data not found in cache...\n');
    function_handles = get_vectorizing_function_handles(params);
    beliefSpacePrecision_EMU_subopt_DBS = params.beliefSpacePrecision_EMU_subopt_DBS;
    a_num = params.a_num;
    P_HgA = params.P_HgA;
    y_control_num = params.y_control_num;

    %% Belief space discretization
    prob_int_sum = floor(1/beliefSpacePrecision_EMU_subopt_DBS);
    prob_temp1 = nchoosek(1:(prob_int_sum+a_num-1), a_num-1);
    prob_ndividers = size(prob_temp1, 1);
    prob_temp2 = cat(2, zeros(prob_ndividers, 1), prob_temp1, (prob_int_sum+a_num)*ones(prob_ndividers, 1));
    a_dbs = beliefSpacePrecision_EMU_subopt_DBS*(diff(prob_temp2, 1, 2) - 1)';
    a_dbs_count = size(a_dbs,2);

    a_nn_cells = get_NN_ClassifyingConstraints(a_dbs,beliefSpacePrecision_EMU_subopt_DBS);

    DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);
    get_adv_guess_g_belief_k_fn = @(x)getHypothesisGuess(x,DRs_in_Rn);

    %% Possible adversarial DR intersections
    for PAk_idx = 1:a_dbs_count
        a_nn_cells(PAk_idx).Data.HhIdx = get_adv_guess_g_belief_k_fn(P_HgA*a_dbs(:,PAk_idx));
    end
    params_t = params;
    params_t.valid_XgYZn1 = pp_data.valid_XgYZn1;

    %% Possible belief transitions map
    [gurobi_model,gurobi_model_params,Aeq_cons,beq_cons] = get_gurobi_model_FDC(params_t,function_handles);
    P_AgU_YcAkn1 = pp_data.P_AgU_YcAkn1;

    possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1 = cell(a_dbs_count,1);
    num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1 = zeros(a_dbs_count,1);
    schedulable_Yc_idx_PAIdxkn1_flag = zeros(y_control_num,a_dbs_count);
    suppressable_Yc_idxs_PAIdxkn1_flag = zeros(y_control_num,a_dbs_count);

    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing possible belief transitions : ');
    incPercent = (1/a_dbs_count)*100;
    internal_routine_fn = @internal_routine;
    [~,p_pool] = evalc('gcp(''nocreate'');');
    
    worketData = struct;
    worketData.gurobi_model = gurobi_model;
    worketData.P_AgU_YcAkn1 = P_AgU_YcAkn1;
    worketData.a_nn_cells = a_nn_cells;
    worketData.a_dbs = a_dbs;
    worketData.params = params_t;
    worketData.function_handles = function_handles;
    worketData.gurobi_model_params = gurobi_model_params;
    worketData.Aeq_cons = Aeq_cons;
    worketData.beq_cons = beq_cons;
    if isempty(p_pool) || useparpool
        for PAkn1_idx = 1:a_dbs_count
            [out_data] = feval(internal_routine_fn, worketData, PAkn1_idx); %#ok<FVAL>
            possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1{PAkn1_idx} = out_data.possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1;
            num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1(PAkn1_idx) = out_data.num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1;
            schedulable_Yc_idx_PAIdxkn1_flag(:,PAkn1_idx) = out_data.schedulable_Yc_idx_PAIdxkn1_flag;
            suppressable_Yc_idxs_PAIdxkn1_flag(:,PAkn1_idx) = out_data.suppressable_Yc_idxs_PAIdxkn1_flag;
            send(progressDataQueue, incPercent);
        end
    else
        worketData = parallel.pool.Constant(worketData);
        parfor PAkn1_idx = 1:a_dbs_count
            [out_data] = feval(internal_routine_fn, worketData.Value, PAkn1_idx); %#ok<FVAL>
            possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1{PAkn1_idx} = out_data.possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1;
            num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1(PAkn1_idx) = out_data.num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1;
            schedulable_Yc_idx_PAIdxkn1_flag(:,PAkn1_idx) = out_data.schedulable_Yc_idx_PAIdxkn1_flag;
            suppressable_Yc_idxs_PAIdxkn1_flag(:,PAkn1_idx) = out_data.suppressable_Yc_idxs_PAIdxkn1_flag;
            send(progressDataQueue, incPercent);
        end
    end
    progressData.terminate();

    %% Store pre processing
    save(fileFullPath,'params',...
        getVarName(a_nn_cells),...
        getVarName(possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1),...
        getVarName(num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1),...
        getVarName(schedulable_Yc_idx_PAIdxkn1_flag),...
        getVarName(suppressable_Yc_idxs_PAIdxkn1_flag));
    fprintf('\t\tPre processing complete. Data saved in: %s\n',fileFullPath);
end


    function [out_data] = internal_routine(worketData, PAkn1_idx)
        gurobi_model_ = worketData.gurobi_model;
        a_nn_cells_ = worketData.a_nn_cells;
        a_dbs_ = worketData.a_dbs;
        P_Akn1 = a_dbs_(:,PAkn1_idx);
        P_AgU_YcAkn1_ = worketData.P_AgU_YcAkn1;
        Aeq_cons_ = worketData.Aeq_cons;
        beq_cons_ = worketData.beq_cons;
        function_handles_ = worketData.function_handles;
        gurobi_model_params_ = worketData.gurobi_model_params;
        eq_cons_num = length(beq_cons_);
        a_dbs_count_ = size(a_dbs_,2);

        params_ = worketData.params;
        y_control_num_ = params_.y_control_num;
        y_control_range = 1:y_control_num_;
        h_num_ = params_.h_num;
        x_num_ = params_.x_num;
        a_num_ = params_.a_num;
        s_num_ = h_num_*x_num_*a_num_;
        u_num_ = y_control_num_*s_num_;
        minLikelihoodFilter = params_.minLikelihoodFilter;
        paramsPrecision = params_.paramsPrecision;
        s_range = 1:s_num_;

        YcsS_2U_ = function_handles_.YcsS_2U;

        schedulable_yc_idxs_flag = false(y_control_num_,1);
        suppressable_yc_idxs_flag = false(y_control_num_,1);
        possible_PAIdxk_g_Yck_flag = false(a_dbs_count_,y_control_num_);
        P_AgU_given_Yc = cell(y_control_num_,1);

        NZ_P_Akn1_idxs = find(P_Akn1>0)';
        P_YcgU_sum = zeros(1,u_num_);
        for Yck_idx = y_control_range
            P_AgU = sparse(a_num_,u_num_);
            for Akn1_idx = NZ_P_Akn1_idxs
                P_AgU = P_AgU + P_AgU_YcAkn1_{Yck_idx,Akn1_idx}*P_Akn1(Akn1_idx);
            end

            P_AgU = full(P_AgU);
            P_AgU_given_Yc{Yck_idx} = P_AgU;
            P_YcgU = sum(P_AgU,1);
            P_YcgU_sum = P_YcgU_sum + P_YcgU;

            gurobi_model_tt = gurobi_model_;
            gurobi_model_tt.obj  = -full(P_YcgU);
            gurobi_result_tt = gurobi(gurobi_model_tt, gurobi_model_params_);
            if strcmp(gurobi_result_tt.status, 'OPTIMAL') && -gurobi_result_tt.objval >= minLikelihoodFilter
                schedulable_yc_idxs_flag(Yck_idx) = true;
                for PAk_idx_ = 1:a_dbs_count_
                    Aineq_cons_PAk_idx = a_nn_cells_(PAk_idx_).A*P_AgU - a_nn_cells_(PAk_idx_).b*P_YcgU;
                    bineq_cons_PAk_idx = -paramsPrecision*ones(size(Aineq_cons_PAk_idx,1),1);

                    Aineq_cons_t = [Aineq_cons_PAk_idx;-P_YcgU];
                    bineq_cons_t = [bineq_cons_PAk_idx;-(minLikelihoodFilter+paramsPrecision)];

                    gurobi_model_tt.A = sparse([Aeq_cons_;Aineq_cons_t]);
                    gurobi_model_tt.rhs   = [beq_cons_;bineq_cons_t];
                    gurobi_model_tt.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
                    gurobi_result_tt = gurobi(gurobi_model_tt, gurobi_model_params_);
                    if(strcmp(gurobi_result_tt.status, 'OPTIMAL'))
                        possible_PAIdxk_g_Yck_flag(PAk_idx_,Yck_idx) = true;
                    end
                end
                if(~any(possible_PAIdxk_g_Yck_flag(:,Yck_idx)))
                    error('~any(possible_PAIdxk_g_Yck_flag(:,Yck_idx))')
                end
            end

            gurobi_model_tt.obj  = full(P_YcgU);
            Aineq_cons_t = P_YcgU;
            bineq_cons_t = (minLikelihoodFilter-paramsPrecision);
            gurobi_model_tt.A = sparse([Aeq_cons_;Aineq_cons_t]);
            gurobi_model_tt.rhs   = [beq_cons_;bineq_cons_t];
            gurobi_model_tt.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
            gurobi_result_tt = gurobi(gurobi_model_tt, gurobi_model_params_);
            if strcmp(gurobi_result_tt.status, 'OPTIMAL')
                suppressable_yc_idxs_flag(Yck_idx) = true;
            end
        end

        if(~any(schedulable_yc_idxs_flag))
            error('~any(valid_yc_idxs_flag)')
        end

        feasible_yc_idxs_flag = reshape((schedulable_yc_idxs_flag|suppressable_yc_idxs_flag),1,[]);
        feasible_yc_idxs = find(feasible_yc_idxs_flag);
        if(any(~feasible_yc_idxs_flag))
            gurobi_var_dim = u_num_;
            Aeq_cons_2 = zeros(1,gurobi_var_dim);
            infeasible_yc_idxs = find(~feasible_yc_idxs_flag);
            for s_k_idx = s_range
                Aeq_cons_2(s_k_idx,feval(YcsS_2U_, infeasible_yc_idxs,s_k_idx)) = 1;
            end
            if(any(Aeq_cons_2>0))
                Aeq_cons_ = [Aeq_cons_;Aeq_cons_2];
                beq_cons_ = [beq_cons_;0];
                eq_cons_num = eq_cons_num + 1;
            end
        end

        possible_PAIdx_given_Yc_vecs = find(possible_PAIdxk_g_Yck_flag(:,1))';
        if(isempty(possible_PAIdx_given_Yc_vecs)||suppressable_yc_idxs_flag(1))
            possible_PAIdx_given_Yc_vecs=[possible_PAIdx_given_Yc_vecs,nan];
        end
        for yc_idx = 2:y_control_num_
            temp_PAIdxs = find(possible_PAIdxk_g_Yck_flag(:,yc_idx))';
            if(isempty(temp_PAIdxs)||suppressable_yc_idxs_flag(yc_idx))
                temp_PAIdxs=[temp_PAIdxs,nan]; %#ok<AGROW> 
            end
            possible_PAIdx_given_Yc_vecs = combvec(possible_PAIdx_given_Yc_vecs,temp_PAIdxs);
        end

        possible_PAIdx_given_Yc_vecs(:,all(isnan(possible_PAIdx_given_Yc_vecs),1)) = [];
        num_possible_PAIdx_given_Yc_vecs = size(possible_PAIdx_given_Yc_vecs,2);
        if(num_possible_PAIdx_given_Yc_vecs>0)
            isValid_PAIdx_given_Yc_vec_idx_flag = false(num_possible_PAIdx_given_Yc_vecs,1);
            for PAIdx_given_Yc_vec_idx = 1:num_possible_PAIdx_given_Yc_vecs
                PAIdx_given_Y = possible_PAIdx_given_Yc_vecs(:,PAIdx_given_Yc_vec_idx);

                Aineq_cons_t = [];
                bineq_cons_t = [];
                for Yck_idx = feasible_yc_idxs
                    P_AgU = P_AgU_given_Yc{Yck_idx};
                    P_YcgU = sum(P_AgU,1);

                    PAk_idx_t = PAIdx_given_Y(Yck_idx);
                    if(~isnan(PAk_idx_t))
                        Aineq_cons_PAk_idx = a_nn_cells_(PAk_idx_t).A*P_AgU - a_nn_cells_(PAk_idx_t).b*P_YcgU;
                        bineq_cons_PAk_idx = -paramsPrecision*ones(size(Aineq_cons_PAk_idx,1),1);

                        Aineq_cons_t = [Aineq_cons_t;Aineq_cons_PAk_idx;-P_YcgU]; %#ok<AGROW> 
                        bineq_cons_t = [bineq_cons_t;bineq_cons_PAk_idx;-(minLikelihoodFilter+paramsPrecision)]; %#ok<AGROW> 
                    else
                        Aineq_cons_t = [Aineq_cons_t;P_YcgU]; %#ok<AGROW> 
                        bineq_cons_t = [bineq_cons_t;minLikelihoodFilter-paramsPrecision]; %#ok<AGROW> 
                    end
                end

                gurobi_model_.A = sparse([Aeq_cons_;Aineq_cons_t]);
                gurobi_model_.rhs   = [beq_cons_;bineq_cons_t];
                gurobi_model_.sense = [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
                gurobi_model_.obj  = zeros(1,u_num_);
                gurobi_result_t = gurobi(gurobi_model_, gurobi_model_params_);
                if (strcmp(gurobi_result_t.status, 'OPTIMAL'))
                    isValid_PAIdx_given_Yc_vec_idx_flag(PAIdx_given_Yc_vec_idx) = true;
                end
            end
            possible_PAIdx_given_Yc_vecs = possible_PAIdx_given_Yc_vecs(:,isValid_PAIdx_given_Yc_vec_idx_flag);
            num_possible_PAIdx_given_Yc_vecs = size(possible_PAIdx_given_Yc_vecs,2);

            out_data = struct;
            out_data.possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1 = possible_PAIdx_given_Yc_vecs;
            out_data.num_possible_PAkIdx_given_Yck_vecs_g_PAIdxkn1 = num_possible_PAIdx_given_Yc_vecs;
            out_data.schedulable_Yc_idx_PAIdxkn1_flag = schedulable_yc_idxs_flag;
            out_data.suppressable_Yc_idxs_PAIdxkn1_flag = suppressable_yc_idxs_flag;
        end
    end
end

