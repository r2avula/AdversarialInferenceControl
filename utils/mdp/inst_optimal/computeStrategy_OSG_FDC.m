function [P_U_out, default_strat_flag] = computeStrategy_OSG_FDC(P_Akn1,params,in_debug_mode,useparpool, computeAgainstARandomAdvStrategy, HhIdx_given_Yc_in)
if nargin == 4
    computeAgainstARandomAdvStrategy = false;
    HhIdx_given_Yc_in = [];
end
if nargin == 5
    HhIdx_given_Yc_in = [];
end
gurobi_model = params.gurobi_model;
gurobi_model_params = params.gurobi_model_params;
Aeq_cons = params.Aeq_cons;
beq_cons = params.beq_cons;
P_AgU_YcAkn1 = params.P_AgU_YcAkn1;
DRs_in_Rn = params.DRs_in_Rn;

function_handles = get_vectorizing_function_handles(params);
YcsS_2U = function_handles.YcsS_2U;

P_U_out = params.P_Uk_idle;
default_strat_flag = true;

x_num = params.x_num;
a_num = params.a_num;
y_control_num = params.y_control_num;
h_num = params.h_num;
paramsPrecision = params.paramsPrecision;
minLikelihoodFilter = params.minLikelihoodFilter;
P_HgA = params.P_HgA;
s_num = x_num*h_num*a_num;
u_num = y_control_num*s_num;

s_range = 1:s_num;
y_control_range = 1:y_control_num;

possible_HhIdxk_g_Yck_flag = false(h_num,y_control_num);
P_AgU_Yc = cell(y_control_num,1);
NZ_P_Akn1_idxs = find(P_Akn1>0)';
for Yck_idx = y_control_range
    P_AgU = sparse(a_num,u_num);
    for Akn1_idx = NZ_P_Akn1_idxs
        P_AgU = P_AgU + P_AgU_YcAkn1{Yck_idx,Akn1_idx}*P_Akn1(Akn1_idx);
    end
    P_AgU = full(P_AgU);
    P_AgU_Yc{Yck_idx} = P_AgU;
end
eq_cons_num = length(beq_cons);

schedulable_yc_idxs_flag = false(y_control_num,1);
suppressable_yc_idxs_flag = false(y_control_num,1);
for Yck_idx = y_control_range
    P_HgU = P_HgA*P_AgU_Yc{Yck_idx};
    P_YcgU = sum(P_HgU,1);

    gurobi_model_t = gurobi_model;
    gurobi_model_t.obj  = double(-full(P_YcgU));
    gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
    if strcmp(gurobi_result_t.status, 'OPTIMAL') && -gurobi_result_t.objval >= minLikelihoodFilter
        schedulable_yc_idxs_flag(Yck_idx) = true;
        for Hhk_idx = 1:h_num
            Aineq_cons_HhIdx = DRs_in_Rn(Hhk_idx).A*P_HgU - DRs_in_Rn(Hhk_idx).b*P_YcgU;
            bineq_cons_HhIdx = -paramsPrecision*ones(size(Aineq_cons_HhIdx,1),1);

            Aineq_cons_t = [Aineq_cons_HhIdx;-P_YcgU];
            bineq_cons_t = [bineq_cons_HhIdx;-(minLikelihoodFilter+paramsPrecision)];

            gurobi_model_t.A = sparse([Aeq_cons;double(Aineq_cons_t)]);
            gurobi_model_t.rhs   = [beq_cons;bineq_cons_t];
            gurobi_model_t.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
            gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
            if(strcmp(gurobi_result_t.status, 'OPTIMAL'))
                if(in_debug_mode)
                    P_U = gurobi_result_t.x;
                    P_Hk = P_HgU*P_U;
                    P_Hk_sum = sum(P_Hk);
                    if(P_Hk_sum<minLikelihoodFilter)
                        error('opt_mismatch')
                    end
                end
                possible_HhIdxk_g_Yck_flag(Hhk_idx,Yck_idx) = true;
            end
        end
        if(~any(possible_HhIdxk_g_Yck_flag(:,Yck_idx)))
            error('~any(possible_HhIdxk_g_Yck_flag(:,Yck_idx))')
        end
    end

    gurobi_model_t.obj  = double(full(P_YcgU));
    Aineq_cons_t = P_YcgU;
    bineq_cons_t = (minLikelihoodFilter-paramsPrecision);
    gurobi_model_t.A = sparse([Aeq_cons;double(Aineq_cons_t)]);
    gurobi_model_t.rhs   = [beq_cons;bineq_cons_t];
    gurobi_model_t.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
    gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
    if strcmp(gurobi_result_t.status, 'OPTIMAL')
        suppressable_yc_idxs_flag(Yck_idx) = true;
    end
end

if(~any(schedulable_yc_idxs_flag))
    error('~any(schedulable_yc_idxs_flag)')
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
        Aeq_cons = [Aeq_cons;Aeq_cons_2];
        beq_cons = [beq_cons;0];
        eq_cons_num = eq_cons_num + 1;
    end
end

params.P_AgU_Yc = P_AgU_Yc;
params.feasible_yc_idxs = feasible_yc_idxs;
params.u_num = u_num;
params.Aeq_cons = Aeq_cons;
params.beq_cons = beq_cons;
params.eq_cons_num = eq_cons_num;
if (computeAgainstARandomAdvStrategy)
    num_tries = 10;
    for try_idx = 1:num_tries
        [HhIdx_given_Yc] = getRandomAdvStrategy(params, possible_HhIdxk_g_Yck_flag);
        [risk_, P_U_out] = internal_routine(params, HhIdx_given_Yc);
        if ~isinf(risk_)
            default_strat_flag = false;
            break;
        end
    end
elseif ~isempty(HhIdx_given_Yc_in)
    [risk_, P_U_out] = internal_routine(params, HhIdx_given_Yc_in);
    default_strat_flag = isinf(risk_);
else
    possible_HhIdx_given_Yc_vecs = find(possible_HhIdxk_g_Yck_flag(:,1))';
    if(isempty(possible_HhIdx_given_Yc_vecs)||suppressable_yc_idxs_flag(1))
        possible_HhIdx_given_Yc_vecs=[possible_HhIdx_given_Yc_vecs,nan];
    end
    for yc_idx = 2:y_control_num
        temp_idxs = find(possible_HhIdxk_g_Yck_flag(:,yc_idx))';
        if(isempty(temp_idxs)||suppressable_yc_idxs_flag(yc_idx))
            temp_idxs=[temp_idxs,nan];
        end
        possible_HhIdx_given_Yc_vecs = combvec(possible_HhIdx_given_Yc_vecs,temp_idxs);
    end
    possible_HhIdx_given_Yc_vecs(:,all(isnan(possible_HhIdx_given_Yc_vecs),1)) = [];
    num_possible_HhIdx_given_Yc_vecs = size(possible_HhIdx_given_Yc_vecs,2);

    if(num_possible_HhIdx_given_Yc_vecs>0)
        max_risk_function_t = inf(num_possible_HhIdx_given_Yc_vecs,1);
        internal_routine_fn = @(x)internal_routine(params,x);
        [~,p_pool] = evalc('gcp(''nocreate'');');
        if isempty(p_pool) || ~useparpool
            for HhIdx_given_Yc_vec_idx = 1:num_possible_HhIdx_given_Yc_vecs
                [max_risk_function_t(HhIdx_given_Yc_vec_idx), ~] =...
                    feval(internal_routine_fn, possible_HhIdx_given_Yc_vecs(:,HhIdx_given_Yc_vec_idx));
            end
        else
            parfor HhIdx_given_Yc_vec_idx = 1:num_possible_HhIdx_given_Yc_vecs
                [max_risk_function_t(HhIdx_given_Yc_vec_idx), ~] =...
                    feval(internal_routine_fn, possible_HhIdx_given_Yc_vecs(:,HhIdx_given_Yc_vec_idx)); %#ok<*FVAL>
            end
        end

        [min_value,HhIdx_given_Y_vec_opt_idx] = min(max_risk_function_t);
        if(~isinf(min_value))
            [~, P_U_out] = feval(internal_routine_fn, possible_HhIdx_given_Yc_vecs(:,HhIdx_given_Y_vec_opt_idx));
            default_strat_flag = false;
        end
    end
end

    function [risk_, P_Uk] = internal_routine(params_, HhIdx_given_Yc)
        gurobi_model_ = params_.gurobi_model;
        gurobi_model_params_ = params_.gurobi_model_params;
        P_AgU_Yc_ = params_.P_AgU_Yc;
        feasible_yc_idxs_ = params_.feasible_yc_idxs;
        Aeq_cons_ = params_.Aeq_cons;
        beq_cons_ = params_.beq_cons;
        eq_cons_num_ = params_.eq_cons_num;
        DRs_in_Rn_ = params_.DRs_in_Rn;
        u_num_ = params_.u_num;
        C_HgHh_design_ = params_.C_HgHh_design;
        paramsPrecision_ = params_.paramsPrecision;
        minLikelihoodFilter_ = params_.minLikelihoodFilter;
        y_control_num_ = params_.y_control_num;
        P_Uk_idle = params_.P_Uk_idle;
        P_HgA_ = params_.P_HgA;

        P_Uk = P_Uk_idle;
        risk_ = inf;

        Aineq_cons_ = sparse([]);
        bineq_cons_ = [];
        alpha_vector_ = zeros(1,u_num_);
        P_YcgU_all = zeros(y_control_num_, u_num_);
        for Yck_idx_ = feasible_yc_idxs_
            Hhk_idx_ = HhIdx_given_Yc(Yck_idx_);
            P_HgU_ = P_HgA_*P_AgU_Yc_{Yck_idx_};
            P_YcgU_ = sum(P_HgU_,1);
            P_YcgU_all(Yck_idx_,:) = P_YcgU_;
            if(~isnan(Hhk_idx_))
                Aineq_cons_HhIdx_ = DRs_in_Rn_(Hhk_idx_).A*P_HgU_ - DRs_in_Rn_(Hhk_idx_).b*P_YcgU_;
                bineq_cons_HhIdx_ = -paramsPrecision_*ones(size(Aineq_cons_HhIdx_,1),1);

                Aineq_cons_ = [Aineq_cons_;Aineq_cons_HhIdx_;-P_YcgU_];
                bineq_cons_ = [bineq_cons_;bineq_cons_HhIdx_;-(minLikelihoodFilter_+paramsPrecision_)]; %#ok<*AGROW>
                alpha_vector_ = alpha_vector_ + C_HgHh_design_(Hhk_idx_,:)*P_HgU_;
            else
                Aineq_cons_ = [Aineq_cons_;P_YcgU_];
                bineq_cons_ = [bineq_cons_;(minLikelihoodFilter_-paramsPrecision_)];
            end
        end

        gurobi_model_.A = [Aeq_cons_;double(Aineq_cons_)];
        gurobi_model_.rhs   = [beq_cons_;bineq_cons_];
        gurobi_model_.sense =  [repmat('=',[1,eq_cons_num_]),repmat('<',[1,length(bineq_cons_)])];
        gurobi_model_.obj  = double(full(alpha_vector_));
        gurobi_result_ = gurobi(gurobi_model_, gurobi_model_params_);
        if strcmp(gurobi_result_.status, 'OPTIMAL')
            risk_ = gurobi_result_.objval;
            P_Uk = sparse(gurobi_result_.x);
        end
    end

    function [HhIdx_given_Yc] = getRandomAdvStrategy(params, possible_HhIdxk_g_Yck_flag)
        y_control_num_ = params.y_control_num;
        HhIdx_given_Yc = nan(y_control_num_,1);
        for yc_idx_ = 1:y_control_num_
            temp_idxs_ = find(possible_HhIdxk_g_Yck_flag(:,yc_idx_))';
            if(~isempty(temp_idxs_))
                HhIdx_given_Yc(yc_idx_) = temp_idxs_(randi(length(temp_idxs_)));
            end
        end
    end
end

