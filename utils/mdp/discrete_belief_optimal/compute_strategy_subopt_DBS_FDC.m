function [strategy_data,P_AgU_given_Yc,P_YcgU_sum] = compute_strategy_subopt_DBS_FDC(params,valueFunction_in1) %#ok<*AGROW> 
paramsPrecision = params.paramsPrecision;
beliefSpacePrecision_EMU_subopt_DBS = params.beliefSpacePrecision_EMU_subopt_DBS;
discountFactor =  params.discountFactor;
C_HgHh_design_t = params.C_HgHh_design;
y_control_num = params.y_control_num;
P_HgA = params.P_HgA;
a_num = params.a_num;
P_Akn1 = params.P_Akn1;
a_nn_cells_t = params.a_nn_cells;
possible_PAIdx_given_Yc_vecs = params.possible_PAIdx_given_Yc_vecs;
P_AgU_YcAkn1 = params.P_AgU_YcAkn1;
gurobi_model = params.gurobi_model;
gurobi_model_params = params.gurobi_model_params;
Aeq_cons_t = params.Aeq_cons;
beq_cons_t = params.beq_cons;
eq_cons_num = length(beq_cons_t);
minLikelihoodFilter = params.minLikelihoodFilter;
schedulable_yc_idxs_flag = params.schedulable_yc_idxs_flag;
suppressable_yc_idxs_flag = params.suppressable_yc_idxs_flag;

YcsS_2U = params.YcsS_2U;
u_num = params.u_num;
h_num = params.h_num;
x_num = params.x_num;
z_num = params.z_num;
s_num = x_num*h_num*z_num*h_num;
s_range = 1:s_num;

%% Pre processing
a_nn_roundOffBelief_fn = @(x)roundOffInSimplex(x,[],a_nn_cells_t);
roundOffBelief_beliefSpacePrecision_fn = @(x)roundOffInSimplex(x,beliefSpacePrecision_EMU_subopt_DBS);
roundOffBelief_paramsPrecision_fn = @(x)roundOffInSimplex(x,paramsPrecision);

%% Value Iteration
P_AgU_given_Yc = cell(y_control_num,1);
P_YcgU_sum = sparse(1,u_num);
NZ_P_Akn1_idxs = find(P_Akn1>0)';
for Yck_idx = 1:y_control_num
    P_AgU = sparse(a_num,u_num);
    for Akn1_idx = NZ_P_Akn1_idxs
        P_AgU = P_AgU + P_AgU_YcAkn1{Yck_idx,Akn1_idx}*P_Akn1(Akn1_idx);
    end
    P_AgU = full(P_AgU);
    P_AgU_given_Yc{Yck_idx} = P_AgU;
    P_YcgU_sum = P_YcgU_sum + sum(P_AgU,1);
end

feasible_yc_idxs_flag = reshape((schedulable_yc_idxs_flag|suppressable_yc_idxs_flag),1,[]);
feasible_yc_idxs = find(feasible_yc_idxs_flag);
if(any(~feasible_yc_idxs_flag))
    gurobi_var_dim = u_num;
    Aeq_cons_2 = zeros(1,gurobi_var_dim);
    infeasible_yc_idxs = find(~feasible_yc_idxs_flag);
    for s_k_idx = s_range
        Aeq_cons_2(s_k_idx,feval(YcsS_2U, infeasible_yc_idxs,s_k_idx)) = 1;
    end
    if(any(Aeq_cons_2>0))
        Aeq_cons_t = [Aeq_cons_t;Aeq_cons_2];
        beq_cons_t = [beq_cons_t;0];
        eq_cons_num = eq_cons_num + 1;
    end
end

strategy_data = [];
num_possible_PAIdx_given_Yc_vecs = size(possible_PAIdx_given_Yc_vecs,2);
if(num_possible_PAIdx_given_Yc_vecs>0)
    value_function_t = inf(num_possible_PAIdx_given_Yc_vecs,1);
    for PAIdx_given_Yc_vec_idx = 1:num_possible_PAIdx_given_Yc_vecs
        gurobi_model_t = gurobi_model;
        PAIdx_given_Yc = possible_PAIdx_given_Yc_vecs(:,PAIdx_given_Yc_vec_idx);

        alpha_vector_t = sparse(1,u_num);
        Aineq_cons_t = [];
        bineq_cons_t = [];
        for Yck_idx = feasible_yc_idxs
            P_AgU = P_AgU_given_Yc{Yck_idx};
            P_YcgU = sum(P_AgU,1);
            PAk_idx = PAIdx_given_Yc(Yck_idx);
            if(~isnan(PAk_idx))
                Hhk_idx = a_nn_cells_t(PAk_idx).Data.HhIdx;

                Aineq_cons_PAk_idx = a_nn_cells_t(PAk_idx).A*P_AgU - a_nn_cells_t(PAk_idx).b*P_YcgU;
                bineq_cons_PAk_idx = -paramsPrecision*ones(size(Aineq_cons_PAk_idx,1),1);

                Aineq_cons_t = [Aineq_cons_t;Aineq_cons_PAk_idx;-P_YcgU];
                bineq_cons_t = [bineq_cons_t;bineq_cons_PAk_idx;-(minLikelihoodFilter+paramsPrecision)];

                alpha_vector_t = alpha_vector_t + C_HgHh_design_t(Hhk_idx,:)*P_HgA*P_AgU + discountFactor*valueFunction_in1(PAk_idx)*P_YcgU;
            else
                Aineq_cons_t = [Aineq_cons_t;P_YcgU];
                bineq_cons_t = [bineq_cons_t;minLikelihoodFilter-paramsPrecision];
            end
        end

        gurobi_model_t.A = sparse([Aeq_cons_t;Aineq_cons_t]);
        gurobi_model_t.rhs   = [beq_cons_t;bineq_cons_t];
        gurobi_model_t.sense = [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];

        gurobi_model_t.obj  = full(alpha_vector_t);
        gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
        if (strcmp(gurobi_result_t.status, 'OPTIMAL'))
            value_function_t(PAIdx_given_Yc_vec_idx) = roundOff(gurobi_result_t.objval/(P_YcgU_sum*gurobi_result_t.x),paramsPrecision);
        end
    end
    [min_value,PAIdx_given_Y_vec_opt_idx]= min(value_function_t);
    if(~isinf(min_value))
        gurobi_model_t = gurobi_model;
        PAIdx_given_Yc = possible_PAIdx_given_Yc_vecs(:,PAIdx_given_Y_vec_opt_idx);

        alpha_vector_t = zeros(1,u_num);
        Aineq_cons_t = [];
        bineq_cons_t = [];
        for Yck_idx = feasible_yc_idxs
            P_AgU = P_AgU_given_Yc{Yck_idx};
            P_YcgU = sum(P_AgU,1);

            PAk_idx = PAIdx_given_Yc(Yck_idx);
            if(~isnan(PAk_idx))
                Hhk_idx = a_nn_cells_t(PAk_idx).Data.HhIdx;

                Aineq_cons_PAk_idx = a_nn_cells_t(PAk_idx).A*P_AgU - a_nn_cells_t(PAk_idx).b*P_YcgU;
                bineq_cons_PAk_idx = -paramsPrecision*ones(size(Aineq_cons_PAk_idx,1),1);

                Aineq_cons_t = [Aineq_cons_t;Aineq_cons_PAk_idx;-P_YcgU];
                bineq_cons_t = [bineq_cons_t;bineq_cons_PAk_idx;-(minLikelihoodFilter+paramsPrecision)];

                alpha_vector_t = alpha_vector_t + C_HgHh_design_t(Hhk_idx,:)*P_HgA*P_AgU + discountFactor*valueFunction_in1(PAk_idx)*P_YcgU;
            else
                Aineq_cons_t = [Aineq_cons_t;P_YcgU];
                bineq_cons_t = [bineq_cons_t;minLikelihoodFilter-paramsPrecision];
            end
        end

        gurobi_model_t.A = sparse([Aeq_cons_t;Aineq_cons_t]);
        gurobi_model_t.rhs   = [beq_cons_t;bineq_cons_t];
        gurobi_model_t.sense = [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];

        gurobi_model_t.obj  = full(alpha_vector_t);
        gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
        P_U = gurobi_result_t.x;

        %% correction of finite precision effects
        PAIdx_given_Yc = nan(y_control_num,1);
        for Yck_idx = feasible_yc_idxs
            P_AgU = P_AgU_given_Yc{Yck_idx};
            P_Ak = P_AgU*P_U;
            P_Ak_sum = sum(P_Ak);
            if(P_Ak_sum>=minLikelihoodFilter)
                P_Ak = P_Ak/P_Ak_sum;
                P_Ak = roundOffBelief_beliefSpacePrecision_fn(roundOffBelief_paramsPrecision_fn(P_Ak));
                [~,PAk_idx] = a_nn_roundOffBelief_fn(P_Ak);
                PAIdx_given_Yc(Yck_idx) = PAk_idx;
            end
        end

        if(any(isnan(PAIdx_given_Yc)))
            P_Ak_sum_t = zeros(y_control_num,1);
            for Yck_idx = feasible_yc_idxs
                P_Ak_sum_t(Yck_idx) = sum(P_AgU_given_Yc{Yck_idx}*P_U);
            end
            [P_Ak_sum_max_t,Yck_idx_t] = max(P_Ak_sum_t);
            if(P_Ak_sum_max_t>=minLikelihoodFilter)
                PAIdx_given_Yc(isnan(PAIdx_given_Yc)) = PAIdx_given_Yc(Yck_idx_t);
            else
                error('opt_mismatch')
            end
        end

        %% Save policy data
        strategy_data = struct;
        strategy_data.P_U = P_U;
        strategy_data.PAIdx_given_Y = PAIdx_given_Yc;
        strategy_data.min_value = min_value;
    end
end
end


