function [policy] = get_policy_opt_det_SP_FDC(params_in,max_num_gamma_vectors,max_valueFnIterations,PP_data_filename,fileNamePrefix, useparpool)
PP_data_params_fields = fieldnames(load(PP_data_filename,'params').('params'));
params = struct;
used_fieldnames = {'y_control_num','a_num','P_HgA','C_HgHh_design','doPruning','gamma_vec_conv_threshold','discountFactor'};
for fn = union(used_fieldnames,PP_data_params_fields)'
    params.(fn{1}) = params_in.(fn{1});
end
num_gamma_vectors_sum = 0;

[policy_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    cached_data_ = load(policy_fileFullPath,'policy');
    policy = cached_data_.policy;
    valueFnIterationsComplete = policy.iter_idx;
    if(max_valueFnIterations>valueFnIterationsComplete)
        if(policy.isTerminated)
            if(sum(policy.num_gamma_vectors,'all')<max_num_gamma_vectors && ~policy.out_of_memory)
                policy_in1 = struct;
                if(~isempty(policy.gamma_vectors_in1))
                    policy_in1.iter_idx = policy.iter_idx - 1;
                    policy_in1.gamma_vectors = policy.gamma_vectors_in1;
                    policy_in1.num_gamma_vectors = policy.num_gamma_vectors_in1;
                else
                    policy_in1.gamma_vectors = [];
                    policy_in1.num_gamma_vectors = 0;
                end
                [policy] = iterate_policy_opt_det_SP_FDC(params, PP_data_filename,policy_in1);
                policy.isTerminated = policy.out_of_memory  || sum(policy.num_gamma_vectors,'all')>=max_num_gamma_vectors;
                if(policy.isTerminated)
                    policy.gamma_vectors_in1 = policy_in1.gamma_vectors;
                    policy.num_gamma_vectors_in1 = policy_in1.num_gamma_vectors;
                    policy = rmfield(policy,'gamma_vectors');
                    save(policy_fileFullPath,'policy','params')
                end
            end
        end
        for iter_idx  = valueFnIterationsComplete + 1:max_valueFnIterations
            if(policy.isConverged || policy.isTerminated)
                return
            end
            policy_in1 = policy;
            [policy] = iterate_policy_opt_det_SP_FDC(params, PP_data_filename,policy_in1);
            policy.isTerminated =  policy.out_of_memory || sum(policy.num_gamma_vectors,'all')>=max_num_gamma_vectors;
            if(policy.isTerminated)
                policy.gamma_vectors_in1 = policy_in1.gamma_vectors;
                policy.num_gamma_vectors_in1 = policy_in1.num_gamma_vectors;
                policy = rmfield(policy,'gamma_vectors');
            end
            save(policy_fileFullPath,'policy','params')
        end
    end
else
    [policy] = iterate_policy_opt_det_SP_FDC(params, PP_data_filename,[]);
    policy.isTerminated = policy.out_of_memory || sum(policy.num_gamma_vectors,'all')>=max_num_gamma_vectors;
    if(policy.isTerminated)
        policy.gamma_vectors_in1 = [];
        policy.num_gamma_vectors_in1 = 0;
        policy = rmfield(policy,'gamma_vectors');
    end
    save(policy_fileFullPath,'policy','params')
    for iter_idx  = 2:max_valueFnIterations
        if(policy.isConverged || policy.isTerminated)
            return
        end
        policy_in1 = policy;
        [policy] = iterate_policy_opt_det_SP_FDC(params, PP_data_filename,policy_in1);
        policy.isTerminated = policy.out_of_memory || sum(policy.num_gamma_vectors,'all')>=max_num_gamma_vectors;
        if(policy.isTerminated)
            policy.gamma_vectors_in1 = policy_in1.gamma_vectors;
            policy.num_gamma_vectors_in1 = policy_in1.num_gamma_vectors;
            policy = rmfield(policy,'gamma_vectors');
        end
        save(policy_fileFullPath,'policy','params')
    end
end

%% Supporting functions
    function [policy_i] = iterate_policy_opt_det_SP_FDC(params, PP_data_filename,policy_in1)
        a_num = params.a_num;
        cached_data = load(PP_data_filename);
        EMUsubpolicies_vec_space = cached_data.('EMUsubpolicies_vec_space');
        P_AkgAkn1_Yck_EMUsubpolicy_idx = cached_data.('P_AkgAkn1_Yck_EMUsubpolicy_idx');
        polyhedralCones = cached_data.('polyhedralCones');
        polyhedralCone_DR = cached_data.('polyhedralCone_DR');
        possiblePartitionTransitionFlag = cached_data.('possiblePartitionTransitionFlag');
        EMUsubpolicies_vec_space_num = size(EMUsubpolicies_vec_space,2);
        polyhedralCones_num = length(polyhedralCones);

        if(isempty(policy_in1) || isempty(policy_in1.gamma_vectors))
            iter_idx_t = 1;
            gamma_vectors_in1 = cell(EMUsubpolicies_vec_space_num,polyhedralCones_num);
            gamma_vectors_in1(:,:) = {zeros(a_num,1)};
            num_gamma_vectors_in1 = ones(EMUsubpolicies_vec_space_num,polyhedralCones_num);
        else
            iter_idx_t = policy_in1.iter_idx + 1;
            gamma_vectors_in1 = policy_in1.gamma_vectors;
            num_gamma_vectors_in1 = policy_in1.num_gamma_vectors;
        end

        num_gamma_vectors_sum = 0;
        params.P_AkgAkn1_Yck_EMUsubpolicy_idx = P_AkgAkn1_Yck_EMUsubpolicy_idx;
        params.polyhedralCone_DR = polyhedralCone_DR;
        params.gamma_vectors_in1 = gamma_vectors_in1;
        params.num_gamma_vectors_in1 = num_gamma_vectors_in1;
        params.EMUsubpolicies_vec_space_num = EMUsubpolicies_vec_space_num;
        params.polyhedralCones = polyhedralCones;

        gamma_vectors = cell(EMUsubpolicies_vec_space_num,polyhedralCones_num);
        num_gamma_vectors = ones(EMUsubpolicies_vec_space_num,polyhedralCones_num);
        num_pruned_gamma_vectors = zeros(polyhedralCones_num,1);

        [progressData, progressDataQueue] = ProgressData(sprintf('\t\tPerforming value iteration # %d --',iter_idx_t));
        incPercent = (1/polyhedralCones_num)*100;

        internal_routine_fn = @internal_routine;
        [~,p_pool] = evalc('gcp(''nocreate'');');

        if isempty(p_pool)
            for g_idx_k = 1:polyhedralCones_num
                [gamma_vectors(:,g_idx_k), num_gamma_vectors(:,g_idx_k) , num_pruned_gamma_vectors(g_idx_k), out_of_memory] =...
                    feval(internal_routine_fn, params, g_idx_k, possiblePartitionTransitionFlag(:,:,:,g_idx_k)); %#ok<FVAL> 
                send(progressDataQueue, incPercent);
                if out_of_memory
                    break;
                end
            end
        else
            params = parallel.pool.Constant(params);
            out_of_memort_flag_ = false(polyhedralCones_num,1);
            out_of_memory = false;
            try
                if useparpool
                    parfor g_idx_k = 1:polyhedralCones_num
                        [gamma_vectors(:,g_idx_k), num_gamma_vectors(:,g_idx_k) , num_pruned_gamma_vectors(g_idx_k), out_of_memort_flag_(g_idx_k)] =...
                            feval(internal_routine_fn, params.Value, g_idx_k, possiblePartitionTransitionFlag(:,:,:,g_idx_k)); %#ok<FVAL>
                        send(progressDataQueue, incPercent);
                    end
                else
                    for g_idx_k = 1:polyhedralCones_num
                        [gamma_vectors(:,g_idx_k), num_gamma_vectors(:,g_idx_k) , num_pruned_gamma_vectors(g_idx_k), out_of_memort_flag_(g_idx_k)] =...
                            feval(internal_routine_fn, params.Value, g_idx_k, possiblePartitionTransitionFlag(:,:,:,g_idx_k)); %#ok<FVAL>
                        send(progressDataQueue, incPercent);
                    end
                end
            catch ex_
                if contains(ex_.message, 'Out of Memory') || contains(ex_.message, 'Out of memory')
                    out_of_memory = true;
                else
                    error(ex_.message)
                end
            end
            if ~out_of_memory
                out_of_memory = any(out_of_memort_flag_);
            end
        end

        isConverged = ~isempty(policy_in1) && (sum(num_gamma_vectors(:)) == sum(num_gamma_vectors_in1(:)));
        progressData.terminate(sprintf('Gamma vectors count: %.2e',sum(num_gamma_vectors,'all')));

        % Save policy data
        policy_i = struct;
        policy_i.iter_idx = iter_idx_t;
        policy_i.gamma_vectors = gamma_vectors;
        policy_i.num_gamma_vectors = num_gamma_vectors;
        policy_i.num_pruned_gamma_vectors = num_pruned_gamma_vectors;
        policy_i.isConverged = isConverged;
        policy_i.out_of_memory = out_of_memory;
    end

    function [gamma_vectors_, num_gamma_vectors_, pruned_gamma_vectors_num, out_of_memory_flag] =...
            internal_routine(params, g_idx_k, possiblePartitionTransitionFlag_)
        y_control_num = params.y_control_num;
        a_num = params.a_num;
        P_HgA = params.P_HgA;
        C_HgHh_design = params.C_HgHh_design;
        doPruning = params.doPruning;
        gamma_vec_conv_threshold = params.gamma_vec_conv_threshold;
        discountFactor = params.discountFactor;
        C_HhgA_adv = C_HgHh_design*P_HgA;

        polyhedralCone = params.polyhedralCones(g_idx_k);
        P_AkgAkn1_Yck_EMUsubpolicy_idx = params.P_AkgAkn1_Yck_EMUsubpolicy_idx;
        gamma_vectors_in1 = params.gamma_vectors_in1;
        polyhedralCone_DR = params.polyhedralCone_DR;
        num_gamma_vectors_in1 = params.num_gamma_vectors_in1;
        EMUsubpolicies_vec_space_num = params.EMUsubpolicies_vec_space_num;

        pruned_gamma_vectors_num = 0;
        gamma_vectors_ = cell(EMUsubpolicies_vec_space_num,1);
        num_gamma_vectors_ = ones(EMUsubpolicies_vec_space_num,1);

        out_of_memory_flag = false;
        try
            for emu_sub_strat_idx = 1:EMUsubpolicies_vec_space_num
                gamma_vectors_k_given_strategy = [];
                for yc_idx = 1:y_control_num
                    possiblePartitionTransitionFlag_tt = possiblePartitionTransitionFlag_(:,yc_idx,emu_sub_strat_idx);
                    max_num_gamma_vectors_k_given_strategy_Yk = sum(num_gamma_vectors_in1(:,possiblePartitionTransitionFlag_tt),"all");
                    gamma_vectors_k_given_strategy_Yk = zeros(a_num,max_num_gamma_vectors_k_given_strategy_Yk);
                    possiblePartitionTransitionIdxs = find(possiblePartitionTransitionFlag_tt)';
                    num_gamma_vectors_k_given_strategy_Yk  = 0;
                    for g_idx_kp1 = possiblePartitionTransitionIdxs
                        h_hat_k_idx = polyhedralCone_DR(g_idx_kp1);
                        per_step_cost_vector_given_Yk = (C_HhgA_adv(h_hat_k_idx,:)*P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx,emu_sub_strat_idx})';
                        gamma_vectors_kin1_reachable_t = cell2mat(reshape(gamma_vectors_in1(:,g_idx_kp1),1,[]));
                        discounted_future_cost_vector_given_Yk = discountFactor*transpose(P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx,emu_sub_strat_idx})*gamma_vectors_kin1_reachable_t;
                        gamma_vectors_k_given_strategy_Yk_t = (per_step_cost_vector_given_Yk + discounted_future_cost_vector_given_Yk);
                        gamma_vectors_k_given_strategy_Yk_t = transpose(unique(roundOff(gamma_vectors_k_given_strategy_Yk_t',gamma_vec_conv_threshold),'rows'));
                        fill_range = num_gamma_vectors_k_given_strategy_Yk+(1:size(gamma_vectors_k_given_strategy_Yk_t,2));
                        gamma_vectors_k_given_strategy_Yk(:,fill_range) = gamma_vectors_k_given_strategy_Yk_t;
                        num_gamma_vectors_k_given_strategy_Yk = fill_range(end);
                    end
                    gamma_vectors_k_given_strategy_Yk(:,num_gamma_vectors_k_given_strategy_Yk+1:end) = [];
                    gamma_vectors_k_given_strategy_Yk = transpose(unique(gamma_vectors_k_given_strategy_Yk','rows'));

                    if(isempty(gamma_vectors_k_given_strategy))
                        gamma_vectors_k_given_strategy = gamma_vectors_k_given_strategy_Yk;
                    else
                        mat_A = repelem(gamma_vectors_k_given_strategy, 1, size(gamma_vectors_k_given_strategy_Yk, 2));
                        mat_B = repmat(gamma_vectors_k_given_strategy_Yk, 1, size(gamma_vectors_k_given_strategy, 2));
                        gamma_vectors_k_given_strategy =  transpose(unique((mat_A + mat_B)','rows'));
                    end

                    if num_gamma_vectors_sum + size(gamma_vectors_k_given_strategy,2) >=max_num_gamma_vectors
                        error('Out of Memory')
                    end
                end
                if(doPruning && size(gamma_vectors_k_given_strategy,2)>1)
                    [gamma_vectors_k_given_strategy,inactive_vecs] = prune_POMDP(gamma_vectors_k_given_strategy,polyhedralCone);
                    pruned_gamma_vectors_num = pruned_gamma_vectors_num + sum(inactive_vecs);
                end
                gamma_vectors_{emu_sub_strat_idx} = gamma_vectors_k_given_strategy;
                num_gamma_vectors_(emu_sub_strat_idx) = size(gamma_vectors_k_given_strategy,2);
                num_gamma_vectors_sum = num_gamma_vectors_sum +  num_gamma_vectors_(emu_sub_strat_idx);
            end
        catch ex
            if contains(ex.message, 'Out of Memory') || contains(ex.message, 'Out of memory')
                out_of_memory_flag = true;
            else
                error(ex.message)
            end
        end
    end
end

