 function [policy] = get_policy_subopt_det_SP_FDC(params_in,max_valueFnIterations,PP_data_filename,fileNamePrefix, useparpool)
PP_data_params_fields = fieldnames(load(PP_data_filename,'params').('params'));
params = struct;
used_fieldnames = {'y_control_num','a_num','P_HgA','C_HgHh_design','gamma_vec_conv_threshold','discountFactor'};
for fn = union(used_fieldnames,PP_data_params_fields)'
    params.(fn{1}) = params_in.(fn{1});
end

[policy_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    load(policy_fileFullPath,'policy');
    valueFnIterationsComplete = policy.iter_idx;
    if(max_valueFnIterations>valueFnIterationsComplete)
        for iter_idx  = valueFnIterationsComplete + 1:max_valueFnIterations
            if(policy.isConverged)
                break;
            end
            policy = iterate_policy_subopt_det_SP_FDC(params, PP_data_filename,policy);
            save(policy_fileFullPath,'policy','params')
        end
    end
else
    policy = iterate_policy_subopt_det_SP_FDC(params, PP_data_filename,[]);
    save(policy_fileFullPath,'policy','params')
    for iter_idx  = 2:max_valueFnIterations
        if(policy.isConverged)
            break;
        end
        policy = iterate_policy_subopt_det_SP_FDC(params, PP_data_filename,policy);
        save(policy_fileFullPath,'policy','params')
    end
end

%% Supporting functions
    function [policy_i] = iterate_policy_subopt_det_SP_FDC(params, PP_data_filename,policy_in1)
        a_num = params.a_num;
        gamma_vec_conv_threshold = params.gamma_vec_conv_threshold;

        %% Pre processing
        cached_data = load(PP_data_filename);
        EMUsubpolicies_vec_space = cached_data.('EMUsubpolicies_vec_space');
        P_AkgAkn1_Yck_EMUsubpolicy_idx = cached_data.('P_AkgAkn1_Yck_EMUsubpolicy_idx');
        polyhedralCones = cached_data.('polyhedralCones');
        polyhedralCone_DR = cached_data.('polyhedralCone_DR');
        possiblePartitionTransitionFlag = cached_data.('possiblePartitionTransitionFlag');
        EMUsubpolicies_vec_space_num = size(EMUsubpolicies_vec_space,2);
        polyhedralCones_num = length(polyhedralCones);

        if(isempty(policy_in1))
            iter_idx_t = 1;
            gamma_vectors_in1 = zeros(a_num,EMUsubpolicies_vec_space_num,polyhedralCones_num);
        else
            iter_idx_t = policy_in1.iter_idx + 1;
            gamma_vectors_in1 = policy_in1.gamma_vectors;
        end

        params.P_AkgAkn1_Yck_EMUsubpolicy_idx = P_AkgAkn1_Yck_EMUsubpolicy_idx;
        params.polyhedralCone_DR = polyhedralCone_DR;
        params.gamma_vectors_in1 = gamma_vectors_in1;
        params.EMUsubpolicies_vec_space_num = EMUsubpolicies_vec_space_num;
        params.polyhedralCones = polyhedralCones;

        max_gamma_vectors_diff = 0;
        gamma_vectors = zeros(a_num,EMUsubpolicies_vec_space_num,polyhedralCones_num);

        [progressData, progressDataQueue] = ProgressData(sprintf('\t\tPerforming value iteration # %d --',iter_idx_t));
        incPercent = (1/polyhedralCones_num)*100;

        internal_routine_fn = @internal_routine;
        [~,p_pool] = evalc('gcp(''nocreate'');');

        if isempty(p_pool) || ~useparpool
            for g_idx_k = 1:polyhedralCones_num
                [gamma_vectors(:,:,g_idx_k), max_gamma_vectors_diff_] =...
                    feval(internal_routine_fn, params, g_idx_k, possiblePartitionTransitionFlag(:,:,:,g_idx_k)); %#ok<FVAL> 
                max_gamma_vectors_diff = max(max_gamma_vectors_diff,max_gamma_vectors_diff_);
                send(progressDataQueue, incPercent);
            end
        else
            params = parallel.pool.Constant(params);
            parfor g_idx_k = 1:polyhedralCones_num
                [gamma_vectors(:,:,g_idx_k), max_gamma_vectors_diff_] =...
                    feval(internal_routine_fn, params.Value, g_idx_k, possiblePartitionTransitionFlag(:,:,:,g_idx_k)); %#ok<FVAL> 
                max_gamma_vectors_diff = max(max_gamma_vectors_diff,max_gamma_vectors_diff_);
                send(progressDataQueue, incPercent);
            end
        end

        isConverged = max_gamma_vectors_diff<=gamma_vec_conv_threshold ;
        progressData.terminate(sprintf('Max. gamma vectors diff: %.2e',sum(max_gamma_vectors_diff,'all')));        

        % Save policy data
        policy_i = struct;
        policy_i.iter_idx = iter_idx_t;
        policy_i.gamma_vectors = gamma_vectors;
        policy_i.max_gamma_vectors_diff = max_gamma_vectors_diff;
        policy_i.isConverged = isConverged;
    end

     function [gamma_vectors_, max_gamma_vectors_diff] =...
             internal_routine(params, g_idx_k, possiblePartitionTransitionFlag_)
         y_control_num = params.y_control_num;
         a_num = params.a_num;
         P_HgA = params.P_HgA;
         C_HgHh_design = params.C_HgHh_design;
         gamma_vec_conv_threshold = params.gamma_vec_conv_threshold;
         discountFactor = params.discountFactor;
         C_HhgA_adv = C_HgHh_design*P_HgA;

         polyhedralCones_num = length(params.polyhedralCones);
         P_Akn1 = params.polyhedralCones(g_idx_k).Data.randomInteriorPoint;
         P_AkgAkn1_Yck_EMUsubpolicy_idx = params.P_AkgAkn1_Yck_EMUsubpolicy_idx;
         gamma_vectors_in1 = params.gamma_vectors_in1;
         polyhedralCone_DR = params.polyhedralCone_DR;
         EMUsubpolicies_vec_space_num = params.EMUsubpolicies_vec_space_num;

         max_gamma_vectors_diff = 0;
         gamma_vectors_ = zeros(a_num,EMUsubpolicies_vec_space_num);

         for EMUsubpolicy_idx = 1:EMUsubpolicies_vec_space_num
             gamma_vectors_k_given_strategy = [];
             for yc_idx = 1:y_control_num
                 possiblePartitionTransitionFlag_tt = possiblePartitionTransitionFlag_(:,yc_idx,EMUsubpolicy_idx);
                 max_num_gamma_vectors_k_given_strategy_Yk = EMUsubpolicies_vec_space_num*polyhedralCones_num;
                 gamma_vectors_k_given_strategy_Yk = zeros(a_num,max_num_gamma_vectors_k_given_strategy_Yk);
                 possiblePartitionTransitionIdxs = find(possiblePartitionTransitionFlag_tt)';
                 num_gamma_vectors_k_given_strategy_Yk  = 0;
                 for g_idx_kp1 = possiblePartitionTransitionIdxs
                     h_hat_k_idx = polyhedralCone_DR(g_idx_kp1);
                     per_step_cost_vector_given_Yk = (C_HhgA_adv(h_hat_k_idx,:)*P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx,EMUsubpolicy_idx})';
                     discounted_future_cost_vector_given_Yk = discountFactor*transpose(P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx,EMUsubpolicy_idx})*gamma_vectors_in1(:,:,g_idx_kp1);
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
             end

             if(size(gamma_vectors_k_given_strategy,2)>1)
                 [~,min_vec_idx] = min(P_Akn1'*gamma_vectors_k_given_strategy);
                 gamma_vectors_(:,EMUsubpolicy_idx) = gamma_vectors_k_given_strategy(:,min_vec_idx);
             else
                 gamma_vectors_(:,EMUsubpolicy_idx) = gamma_vectors_k_given_strategy;
             end

             gamma_vectors_diff = abs(gamma_vectors_(:,EMUsubpolicy_idx) - gamma_vectors_in1(:,EMUsubpolicy_idx,g_idx_k));
             max_gamma_vectors_diff = max(max_gamma_vectors_diff,max(gamma_vectors_diff));
         end
     end

end

