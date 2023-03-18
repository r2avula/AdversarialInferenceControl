    function [fileFullPath] = get_PP_data_det_SP_FDC_fileFullPath(params_in,fileNamePrefix, pp_data, useparpool)
params = struct;
used_fieldnames = {'paramsPrecision','h_num','z_num','x_num','d_num','x_p_pu','d_p_pu','x_offset','d_offset','P_Zp1gZD','P_XHgHn1','a_num',...
    'P_HgA','C_HgHh_design','minDet','max_num_EMUsubpolicies','y_control_p_pu','y_control_num','y_control_offset'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end

[fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(~fileExists)
    fprintf('\t\tPre processing data not found in cache...\n');
    function_handles = get_vectorizing_function_handles(params);
    paramsPrecision = params.paramsPrecision;
    h_num = params.h_num;
    x_num = params.x_num;
    d_num = params.d_num;
    x_p_pu = params.x_p_pu;
    d_p_pu = params.d_p_pu;
    x_offset = params.x_offset;
    d_offset = params.d_offset;
    P_Zp1gZD = params.P_Zp1gZD;
    a_num = params.a_num;
    A2HZ = function_handles.A2HZ;
    P_HgA = params.P_HgA;
    minDet = params.minDet;
    C_HgHh_design = params.C_HgHh_design;
    max_num_EMUsubpolicies = params.max_num_EMUsubpolicies;

    y_control_num = params.y_control_num;
    y_control_p_pu = params.y_control_p_pu;
    y_control_offset = params.y_control_offset;
    y_control_range = 1:y_control_num;

    s_num = x_num*h_num*a_num;
    u_num = y_control_num*s_num;
    s_range = 1:s_num;

    S2XHAn1 = function_handles.S2XHAn1;
    YcS_2U = function_handles.YcS_2U;

    P_AgU_YcAkn1 = pp_data.P_AgU_YcAkn1;

    polyHedralCone_a_dim = Polyhedron('A',-eye(a_num),'b',zeros(a_num,1));
    simplex_a_dim = Polyhedron(eye(a_num));
    polyHedralCone_a_dim_outer_hps = Polyhedron();
    eye_temp = eye(a_num);
    for hp_idx = 1:a_num
        R_matrix = eye_temp(hp_idx,:);
        polyHedralCone_a_dim_outer_hps(hp_idx) = Polyhedron('V',zeros(1,a_num),'R',R_matrix);
    end

    delta_vectors_adv = eye(h_num);
    decisionRegion_hp_Ae_T = [];
    for i_idx = 1:h_num
        delta_vectors_adv_t = delta_vectors_adv;
        for j_idx = 1:h_num
            if(j_idx ~=i_idx)
                M_temp = (delta_vectors_adv_t(i_idx,:)-delta_vectors_adv_t(j_idx,:))*C_HgHh_design*P_HgA;
                decisionRegion_hp_Ae_T = [decisionRegion_hp_Ae_T;M_temp]; %#ok<AGROW> 
            end
        end
    end

    decisionRegion_hp_Ae_T = roundOff(normr(decisionRegion_hp_Ae_T),paramsPrecision);
    decisionRegion_hp_Ae_T = unique(decisionRegion_hp_Ae_T, 'rows');
    unique_decisionRegionConstraints = ones(size(decisionRegion_hp_Ae_T,1),1);
    for hp_idx = 1:size(decisionRegion_hp_Ae_T,1)
        if(unique_decisionRegionConstraints(hp_idx))
            hp = decisionRegion_hp_Ae_T(hp_idx,:);
            neg_hp = -hp;
            diff_mat = abs(decisionRegion_hp_Ae_T - neg_hp);
            neg_hp_idx = (sum(diff_mat,2)<=paramsPrecision);
            unique_decisionRegionConstraints(neg_hp_idx) = 0;
        end
    end
    decisionRegion_hp_Ae_T = decisionRegion_hp_Ae_T(unique_decisionRegionConstraints==1,:);
    decisionRegion_hyperPlanes_num = size(decisionRegion_hp_Ae_T,1);

    %% Sub-policy space    
    num_EMUsubpolicies_init_limit = 1e6;
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing subpolicies space : ');
    incPercent = (1/s_num)*100;
    EMUsubpolicies_vec_space = [];
    for s_k_idx = s_range
        [x_k_idx,~,a_kn1_idx] = S2XHAn1(s_k_idx);
        [~,z_kn1_idx] = A2HZ(a_kn1_idx);
        d_range_t = round(((y_control_range+y_control_offset)*y_control_p_pu - (x_k_idx+x_offset)*x_p_pu)/d_p_pu) - d_offset;
        valid_range_idx_flag = d_range_t>=1 & d_range_t<=d_num; 
        valid_yc_idxs = find(valid_range_idx_flag);
        valid_d_range = d_range_t(valid_range_idx_flag);
        valid_yc_idxs = valid_yc_idxs(reshape(sum(P_Zp1gZD(:,z_kn1_idx,valid_d_range),1)>=paramsPrecision,1,[]));      
        if(isempty(valid_yc_idxs))
            error('here');
        end
        if(isempty(EMUsubpolicies_vec_space))
            EMUsubpolicies_vec_space = reshape(valid_yc_idxs,1,[]);
        else
            EMUsubpolicies_vec_space = combvec(EMUsubpolicies_vec_space,reshape(valid_yc_idxs,1,[]));
        end
        EMUsubpolicies_vec_space_num = size(EMUsubpolicies_vec_space,2);
        if(EMUsubpolicies_vec_space_num>num_EMUsubpolicies_init_limit)
            progressData.terminate();
            error('EMUsubpolicies_vec_space_num: %.2e,\ts_k_idx: %d/%d,\tPercent done:%.2f',EMUsubpolicies_vec_space_num,s_k_idx,s_num,(s_k_idx*100/s_num));
        end
        send(progressDataQueue, incPercent);
    end    
    progressData.terminate();
    
    P_AkgAkn1_Yck_EMUsubpolicy_idx = cell(y_control_num,EMUsubpolicies_vec_space_num);
    minDet_EMUsubpolicies = zeros(EMUsubpolicies_vec_space_num,1);
    for EMUsubpolicy_idx = 1:EMUsubpolicies_vec_space_num        
        P_U = zeros(u_num,1);
        P_U(YcS_2U(EMUsubpolicies_vec_space(s_range,EMUsubpolicy_idx)',s_range)) = 1;
        P_AkgAkn1_Yck = cell(y_control_num,1);

        P_AkgAkn1_det_g_Yck = zeros(y_control_num,1);
        for Yck_idx = y_control_range
            P_AkgAkn1 = zeros(a_num,a_num);
            for Akn1_idx = 1:a_num
                P_AkgAkn1(:,Akn1_idx) = P_AkgAkn1(:,Akn1_idx) + P_AgU_YcAkn1{Yck_idx,Akn1_idx}*P_U;
            end
            
            P_AkgAkn1 = full(P_AkgAkn1);
            P_AkgAkn1_Yck{Yck_idx} = P_AkgAkn1/s_num;
            P_AkgAkn1_det_g_Yck(Yck_idx) = abs(det(P_AkgAkn1));
        end
        %         minDet_EMUsubpolicies_ = min(P_AkgAkn1_det_g_Yck(P_AkgAkn1_det_g_Yck>0));
        %         if ~isempty(minDet_EMUsubpolicies_)
        %             minDet_EMUsubpolicies(EMUsubpolicy_idx) = minDet_EMUsubpolicies_;
        %         end
        minDet_EMUsubpolicies(EMUsubpolicy_idx) = min(P_AkgAkn1_det_g_Yck);
        P_AkgAkn1_Yck_EMUsubpolicy_idx(:,EMUsubpolicy_idx) = P_AkgAkn1_Yck;
    end
    
    [minDet_EMUsubpolicies,EMUsubpolicies_sorted_idxs] = sort(minDet_EMUsubpolicies,'descend');
    minDet_EMUsubpolicies = minDet_EMUsubpolicies(minDet_EMUsubpolicies>minDet);
    EMUsubpolicies_vec_space_num = min(max_num_EMUsubpolicies,length(minDet_EMUsubpolicies));
    EMUsubpolicies_vec_space = EMUsubpolicies_vec_space(:,EMUsubpolicies_sorted_idxs(1:EMUsubpolicies_vec_space_num));
    P_AkgAkn1_Yck_EMUsubpolicy_idx = P_AkgAkn1_Yck_EMUsubpolicy_idx(:,EMUsubpolicies_sorted_idxs(1:EMUsubpolicies_vec_space_num));

    M_pi_unique_num = EMUsubpolicies_vec_space_num*y_control_num;
    M_pi_unique = zeros(a_num,a_num,M_pi_unique_num);
    M_pi_unique_vec = zeros(a_num*a_num,M_pi_unique_num);
    mat_idx = 0;
    for EMUsubpolicy_idx = 1:EMUsubpolicies_vec_space_num
        for yc_idx = y_control_range
            mat_idx = mat_idx + 1;
            M_pi_unique(:,:,mat_idx) = P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx,EMUsubpolicy_idx};
            M_pi_unique_vec(:,mat_idx) =  roundOff(reshape(M_pi_unique(:,:,mat_idx), [], 1)/norm(M_pi_unique(:,:,mat_idx),'fro'),paramsPrecision);
        end
    end

    M_pi_unique(:,:,mat_idx+1:end) = [];
    M_pi_unique_vec(:,mat_idx+1:end) = [];
    [~,ia,~] = unique(M_pi_unique_vec','rows');
    M_pi_unique = M_pi_unique(:,:,ia);
    M_pi_unique_num = size(M_pi_unique,3);

    %% Processing partitions of belief_kn1
    hyperPlanes_kp1_Ae_T = decisionRegion_hp_Ae_T;
    hyperPlanes_num_kp1 = decisionRegion_hyperPlanes_num;

    [progressData, progressDataQueue] = ProgressData('\t\t\tPartitioning belief space : ');
    hyperPlanes_k_Ae_T = zeros(hyperPlanes_num_kp1*M_pi_unique_num + decisionRegion_hyperPlanes_num,a_num);
    for M_k_unique_idx = 1:M_pi_unique_num
        hp_range = (M_k_unique_idx-1)*hyperPlanes_num_kp1 + 1: M_k_unique_idx*hyperPlanes_num_kp1;
        hyperPlanes_k_Ae_T(hp_range,:) = roundOff(normr(hyperPlanes_kp1_Ae_T*M_pi_unique(:,:,M_k_unique_idx)),paramsPrecision);
    end

    hp_range = hyperPlanes_num_kp1*M_pi_unique_num + 1 : hyperPlanes_num_kp1*M_pi_unique_num  + decisionRegion_hyperPlanes_num;
    hyperPlanes_k_Ae_T(hp_range,:) = decisionRegion_hp_Ae_T;

    hyperPlanes_k_Ae_T = unique(hyperPlanes_k_Ae_T, 'rows');
    hyperPlanes_k_Ae_T = hyperPlanes_k_Ae_T(~(sum(hyperPlanes_k_Ae_T==0,2)==a_num-1),:); % removes trivial outer boundary constraints
    unique_hp = ones(size(hyperPlanes_k_Ae_T,1),1);
    for hp_idx = 1:size(hyperPlanes_k_Ae_T,1)
        if(unique_hp(hp_idx))
            hp = hyperPlanes_k_Ae_T(hp_idx,:);
            neg_hp = -hp;
            diff_mat = abs(hyperPlanes_k_Ae_T - neg_hp);
            neg_hp_idx = (sum(diff_mat,2)<=paramsPrecision);
            unique_hp(neg_hp_idx) = 0;
        end
    end    
    hyperPlanes_k_Ae_T = hyperPlanes_k_Ae_T(unique_hp==1,:);    
    hyperPlanes_num_k = size(hyperPlanes_k_Ae_T,1);
        
    polyhedralCones = polyHedralCone_a_dim;
    incPercent = (1/hyperPlanes_num_k)*100;
    for hp_idx = 1:hyperPlanes_num_k
        init_cones_num = length(polyhedralCones);
        slicedCones = [];
        residues_num = zeros(init_cones_num,1);
        hp_Ae_T = hyperPlanes_k_Ae_T(hp_idx,:);
        incPercent_t = incPercent/init_cones_num;
        if useparpool
            parfor cone_idx = 1:init_cones_num
                temp_cone = polyhedralCones(cone_idx).copy();
                [residues,residues_num_t] = cutPolyhedronWithHyperPlane(temp_cone,hp_Ae_T);
                residues_num(cone_idx) = residues_num_t;
                slicedCones = [slicedCones;residues];
                send(progressDataQueue, incPercent_t);
            end
        else
            for cone_idx = 1:init_cones_num
                temp_cone = polyhedralCones(cone_idx).copy();
                [residues,residues_num_t] = cutPolyhedronWithHyperPlane(temp_cone,hp_Ae_T);
                residues_num(cone_idx) = residues_num_t;
                slicedCones = [slicedCones;residues]; %#ok<AGROW>
                send(progressDataQueue, incPercent_t);
            end
        end
        residues_sum = sum(residues_num);
        if(residues_sum>init_cones_num)
            polyhedralCones = slicedCones;
        end
    end
    polyhedralCones.minHRep();
    polyhedralCones_num = length(polyhedralCones);
    progressData.terminate();
    fprintf('\t\t\tPartitions count : %d\n',polyhedralCones_num);

    %% Adversarial strategy, partition transition and containing maps
    DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);
    get_adv_guess_g_belief_k_fn = @(x)getHypothesisGuess(x,DRs_in_Rn);

    gurobi_var_dim = a_num;
    gurobi_model.modelsense = 'min';
    gurobi_model.vtype = repmat('C', gurobi_var_dim, 1);
    gurobi_model.lb    = zeros(a_num, 1);
    gurobi_model.ub   = ones(a_num, 1);
    gurobi_model.obj  = zeros(1,a_num);
    gurobi_model_params.outputflag = 0;

    possiblePartitionTransitionFlag = false(polyhedralCones_num,y_control_num,EMUsubpolicies_vec_space_num,polyhedralCones_num);
    polyhedralCone_DR = zeros(polyhedralCones_num,1);
    
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing adversarial strategy and partition map : ');
    roundOffBelief_paramsPrecision_fn = @(x)roundOffInSimplex(x,paramsPrecision);
    for g_idx_k = 1:polyhedralCones_num
        temp_polyhedron = polyhedralCones(g_idx_k)&simplex_a_dim;
        sol = interiorPoint(temp_polyhedron);
        randomInteriorPoint = roundOffBelief_paramsPrecision_fn(sol.x);
        polyhedralCone_DR(g_idx_k) =  get_adv_guess_g_belief_k_fn(P_HgA*randomInteriorPoint);

        polyhedralCones(g_idx_k).minHRep();
        polyhedralCones(g_idx_k).Data.randomInteriorPoint = randomInteriorPoint;
        polyhedralCones(g_idx_k).Data.HhIdx = polyhedralCone_DR(g_idx_k);
    end

    incPercent = (1/polyhedralCones_num/EMUsubpolicies_vec_space_num)*100;
    if useparpool
        parfor g_idx_k = 1:polyhedralCones_num
            P_AkgAkn1_Yk_EMUsubpolicy_idx_t = P_AkgAkn1_Yck_EMUsubpolicy_idx;
            polyhedralCones_t = polyhedralCones;
            temp_polyhedron = polyhedralCones_t(g_idx_k);
            gurobi_model_t = gurobi_model;
            possiblePartitionTransitionFlag_t = false(polyhedralCones_num,y_control_num,EMUsubpolicies_vec_space_num);

            for EMUsubpolicy_idx_t = 1:EMUsubpolicies_vec_space_num
                for yc_idx_t = y_control_range
                    new_A = temp_polyhedron.A/(P_AkgAkn1_Yk_EMUsubpolicy_idx_t{yc_idx_t,EMUsubpolicy_idx_t});
                    for g_idx_kp1 = 1:polyhedralCones_num
                        A_cons = [new_A;polyhedralCones_t(g_idx_kp1).A];
                        gurobi_model_t.A     = sparse([ones(1,a_num);A_cons]);
                        gurobi_model_t.rhs   = [1;zeros(size(A_cons,1),1)];
                        gurobi_model_t.sense = ['=',repelem('<',1,size(A_cons,1))];
                        optimizerResult = gurobi(gurobi_model_t, gurobi_model_params);
                        if strcmp(optimizerResult.status, 'OPTIMAL')
                            possiblePartitionTransitionFlag_t(g_idx_kp1,yc_idx_t,EMUsubpolicy_idx_t) = true;
                        end
                    end
                end
                send(progressDataQueue, incPercent);
            end
            possiblePartitionTransitionFlag(:,:,:,g_idx_k) = possiblePartitionTransitionFlag_t;
        end
    else
        for g_idx_k = 1:polyhedralCones_num
            temp_polyhedron = polyhedralCones(g_idx_k);
            for EMUsubpolicy_idx_t = 1:EMUsubpolicies_vec_space_num
                for yc_idx_t = y_control_range
                    new_A = temp_polyhedron.A/(P_AkgAkn1_Yck_EMUsubpolicy_idx{yc_idx_t,EMUsubpolicy_idx_t});
                    for g_idx_kp1 = 1:polyhedralCones_num
                        A_cons = [new_A;polyhedralCones(g_idx_kp1).A];
                        gurobi_model.A     = sparse([ones(1,a_num);A_cons]);
                        gurobi_model.rhs   = [1;zeros(size(A_cons,1),1)];
                        gurobi_model.sense = ['=',repelem('<',1,size(A_cons,1))];
                        optimizerResult = gurobi(gurobi_model, gurobi_model_params);
                        if strcmp(optimizerResult.status, 'OPTIMAL')
                            possiblePartitionTransitionFlag(g_idx_kp1,yc_idx_t,EMUsubpolicy_idx_t,g_idx_k) = true;
                        end
                    end
                end
                send(progressDataQueue, incPercent);
            end
        end
    end
    progressData.terminate();

    if(~all(any(possiblePartitionTransitionFlag,[1,2,3])))
        error('~all(any(partitionTransitionFlag,[1,2,3]))');
    end

    %% Store pre processing
    save(fileFullPath,'params',...
        getVarName(EMUsubpolicies_vec_space),...
        getVarName(P_AkgAkn1_Yck_EMUsubpolicy_idx),...
        getVarName(polyhedralCones),...
        getVarName(polyhedralCone_DR),...
        getVarName(possiblePartitionTransitionFlag),...
        getVarName(hyperPlanes_k_Ae_T),...
        getVarName(decisionRegion_hp_Ae_T),...
        getVarName(M_pi_unique));
    fprintf('\t\tPre processing complete. Data saved in: %s\n',fileFullPath);
end
end

