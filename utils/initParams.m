function [params_FDC, params_RDC, config] = initParams(config,additional_data, getEssParams_flag)
[config] = expandConfig(config);

x_num= config.x_num;
h_num= config.h_num;
z_num= config.z_num;
paramsPrecision= config.paramsPrecision;
l_num = config.l_num;
z_num_per_level = config.z_num_per_level;
appliances_num = config.applianceGroupsNum;
hypothesisStatesPerAppliance = config.hypothesisStatesPerAppliance;
h_vec_space = config.h_vec_space;

if isfield(additional_data, 'P_H0')
    P_HgHn1_p = config.P_HgHn1_p;
    P_HgHn1_q = config.P_HgHn1_q;
    P_XgH_h1_param = config.P_XgH_h1_param;
    P_XgH_h2_param = config.P_XgH_h2_param;

    P_XgH = zeros(x_num,h_num);
    P_XgH(:,1) = roundOffInSimplex(binopdf(0:x_num-1,x_num-1,P_XgH_h1_param),paramsPrecision);
    P_XgH(:,2) = roundOffInSimplex(binopdf(0:x_num-1,x_num-1,P_XgH_h2_param),paramsPrecision);

    P_HgHn1 = zeros(h_num,h_num);
    P_HgHn1(:,1) = roundOffInSimplex([P_HgHn1_p;1-P_HgHn1_p],paramsPrecision);
    P_HgHn1(:,2) = roundOffInSimplex([1-P_HgHn1_q;P_HgHn1_q],paramsPrecision);

    if(any(sum(P_HgHn1,2)==0))
        error('HMM error')
    end
    mc = dtmc(P_HgHn1);
    xFix = asymptotics(mc);
    if(size(xFix,1)>1)
        error('HMM error')
    end

    P_XHgHn1 = zeros(x_num,h_num,h_num);
    for h_kn1_idx=1:h_num
        %         P_XH_ = zeros(x_num,h_num);
        for h_k_idx = 1:h_num
            for x_k_idx=1:x_num
                %                 P_XH_(x_k_idx,h_k_idx) = roundOff(P_HgHn1(h_k_idx,h_kn1_idx)*P_XgH(x_k_idx,h_k_idx),paramsPrecision);
                P_XHgHn1(x_k_idx,h_k_idx,h_kn1_idx) = roundOff(P_HgHn1(h_k_idx,h_kn1_idx)*P_XgH(x_k_idx,h_k_idx),paramsPrecision);
            end
        end

        %         P_XH_vec = P_XH_(:);
        %         if(sum(P_XH_vec)>=paramsPrecision)
        %             P_XHgHn1(:,:,h_kn1_idx) = reshape(roundOffInSimplex(P_XH_vec,paramsPrecision),[x_num,h_num]);
        %         end
    end

    P_H0 = roundOffInSimplex(additional_data.P_H0,paramsPrecision);
    
    config.P_H0 = P_H0;
    config.P_XgH = P_XgH;
    config.P_HgHn1 = P_HgHn1;
    config.P_XHgHn1 = P_XHgHn1;

    genDataParams = struct;
    genDataParams.k_num = config.k_num;
    genDataParams.h_num = config.h_num;
    genDataParams.x_p_pu = config.x_p_pu;
    genDataParams.x_offset = config.x_offset;
    genDataParams.P_XgH = P_XgH;
    genDataParams.P_HgHn1 = P_HgHn1;
    genDataParams.numHorizons = 1000;
    genDataParams.P_H0 = P_H0;

    fileNamePrefix = sprintf('%ssyntheticData_train_%d_%d_',config.cache_folder_path,config.P_HgHn1_p_idx,config.P_HgHn1_q_idx);
    [fileFullPath,fileExists] = findFileName(genDataParams,fileNamePrefix,'genDataParams');
    if(fileExists)
        load(fileFullPath,'gt_data_train');
    else
        rng(config.rng_id_sim,'twister');
        [sm_data_train,gt_data_train,~,h_0_idxs] = generateSyntheticData(genDataParams);
        save(fileFullPath,'sm_data_train','gt_data_train','h_0_idxs','genDataParams')
    end
    gt_data_train_vec = reshape(gt_data_train,[],1);

    numel_trainingData = genDataParams.numHorizons*config.k_num;
    rewardPerApplianceDetection_adv = cell2mat(config.rewardPerApplianceDetection_adv);
    prior_prob_appliance_states = cell(appliances_num,1);
    for app_idx = 1:appliances_num
        ha_num = hypothesisStatesPerAppliance(app_idx);
        prior_probs_ = zeros(ha_num,1);
        h_data_t = h_vec_space(app_idx,gt_data_train_vec);
        for ha_idx = 1:ha_num
            prior_probs_(ha_idx) = sum(h_data_t==ha_idx)/numel_trainingData;
        end
        prior_prob_appliance_states{app_idx} = prior_probs_;
    end
elseif isfield(additional_data, 'training_data')
    training_applianceData = additional_data.training_data.training_applianceData;
    training_noiseData = additional_data.training_data.training_noiseData;
    training_gtData = additional_data.training_data.training_gtData;
    config = getHMMParams(config,training_applianceData,training_noiseData,training_gtData);

    numel_trainingData = numel(training_gtData(:,:,1));
    rewardPerApplianceDetection_adv = cell2mat(config.rewardPerApplianceDetection_adv);
    prior_prob_appliance_states = cell(appliances_num,1);
    for app_idx = 1:appliances_num
        ha_num = hypothesisStatesPerAppliance(app_idx);
        prior_probs_ = zeros(ha_num,1);
        h_data_t = training_gtData(:,:,app_idx);
        for ha_idx = 1:ha_num
            prior_probs_(ha_idx) = sum(h_data_t==ha_idx,'all')/numel_trainingData;
        end
        prior_prob_appliance_states{app_idx} = prior_probs_;
    end
else
    error('not implemented!')
end

if config.homogeneous_reward
    prior_prob_appliance_states = cell(appliances_num,1);
    for app_idx = 1:appliances_num
        ha_num = hypothesisStatesPerAppliance(app_idx);
        prior_probs_ = ones(ha_num,1);
        prior_prob_appliance_states{app_idx} = prior_probs_;
    end
end

C_HgHh_design = zeros(h_num);
for hh_vec_idx = 1:h_num
    hh_vec = h_vec_space(:,hh_vec_idx);
    for h_vec_idx = 1:h_num
        h_vec = h_vec_space(:,h_vec_idx);
        for app_idx = 1:appliances_num
            if hh_vec(app_idx) == h_vec(app_idx)
                prior_probs_ = prior_prob_appliance_states{app_idx};
                C_HgHh_design(h_vec_idx,hh_vec_idx) = C_HgHh_design(h_vec_idx,hh_vec_idx) + ...
                    roundOff(rewardPerApplianceDetection_adv(app_idx)...
                    /prior_probs_(h_vec(app_idx))^(config.reward_amplifying_factor), paramsPrecision);
            end
        end
    end
end

if any(isinf(C_HgHh_design),'all')
    error('any(isinf(C_HgHh_design),''all'')')
end

% Prepare params
params_FDC = struct;
params_FDC.k_num = config.k_num;
params_FDC.h_num = h_num;
params_FDC.x_num = x_num;
params_FDC.x_offset = config.x_offset;
params_FDC.x_p_pu = config.x_p_pu;
params_FDC.y_p_pu = config.y_p_pu;
params_FDC.d_p_pu = config.d_p_pu;
params_FDC.paramsPrecision = paramsPrecision;
params_FDC.beliefSpacePrecision_adv = config.beliefSpacePrecision_adv;
params_FDC.d_num = config.d_num;
params_FDC.y_num = config.y_num;
params_FDC.d_offset = config.d_offset;
params_FDC.y_offset = config.y_offset;
params_FDC.y_control_p_pu = config.y_control_p_pu;
params_FDC.y_control_num = config.y_control_num;
params_FDC.y_control_offset = config.y_control_offset;
params_FDC.z_num = z_num;
params_FDC.z_offset = config.z_offset;
params_FDC.minPowerDemandInW = config.minPowerDemandInW;
params_FDC.minLikelihoodFilter = config.minLikelihoodFilter;
params_FDC.C_HgHh_homogeneous = config.C_HgHh_homogeneous;
params_FDC.C_HgHh_design = C_HgHh_design;
params_FDC.h_vec_space = h_vec_space;
params_FDC.hypothesisStatesPerAppliance = hypothesisStatesPerAppliance;

params_FDC.P_H0 = config.P_H0;
params_FDC.P_HgHn1 = config.P_HgHn1;
params_FDC.P_XgH = config.P_XgH;
params_FDC.P_XHgHn1 = config.P_XHgHn1;
a_num = h_num*z_num;
params_FDC.a_num = a_num;

function_handles = get_vectorizing_function_handles(params_FDC);
HZ2A = function_handles.HZ2A;
A2HZ = function_handles.A2HZ;

P_HgA = zeros(h_num,a_num);
P_ZgA = zeros(z_num,a_num);
for z_idx = 1:z_num
    for Hk_idx = 1:h_num
        a_idx = HZ2A(Hk_idx,z_idx);
        P_HgA(Hk_idx,a_idx) = 1;
        P_ZgA(z_idx,a_idx) = 1;
    end
end

params_FDC.P_HgA = P_HgA;
params_FDC.P_ZgA = P_ZgA;
if getEssParams_flag
    [params_FDC, ~] = getEssParams(config, params_FDC, false);
end

y_control_num = params_FDC.y_control_num;
s_num = x_num*h_num*a_num;
params_FDC.s_num = s_num;
params_FDC.u_num = y_control_num*s_num;

%% RDC params
params_RDC = params_FDC;
ZIdxsgLIdx = zeros(z_num_per_level,l_num);
LIdxgZIdx = zeros(z_num,1);
Zk_idx = 0;
for Lk_idx = 1:l_num
    ZIdxsgLIdx(:,Lk_idx) = Zk_idx + 1: Zk_idx + z_num_per_level;
    LIdxgZIdx(Zk_idx + 1: Zk_idx + z_num_per_level) = Lk_idx;
    Zk_idx = Zk_idx + z_num_per_level ;
end

b_num = h_num*l_num;
params_RDC.b_num = b_num;
params_RDC.l_num = l_num;
params_RDC.z_num_per_level = z_num_per_level;
params_RDC.ZIdxsgLIdx = ZIdxsgLIdx;
params_RDC.LIdxgZIdx = LIdxgZIdx;

t_num = x_num*h_num*l_num;
w_num = y_control_num*t_num;
params_RDC.t_num = t_num;
params_RDC.w_num = w_num;

function_handles = get_vectorizing_function_handles(params_RDC);
HL2B = function_handles.HL2B;

P_HgB = zeros(h_num,b_num);
P_LgB = zeros(l_num,b_num);
for Lk_idx = 1:l_num
    for Hk_idx = 1:h_num
        b_idx = HL2B(Hk_idx,Lk_idx);
        P_HgB(Hk_idx,b_idx) = 1;
        P_LgB(Lk_idx,b_idx) = 1;
    end
end

P_BgA = zeros(b_num,a_num);
for Ak_idx = 1:a_num
    [Hk_idx,Zk_idx] = A2HZ(Ak_idx);

    Lk_idx = LIdxgZIdx(Zk_idx);
    Bk_idx = HL2B(Hk_idx,Lk_idx);

    P_BgA(Bk_idx,Ak_idx) = 1;
end

params_RDC.P_HgB = P_HgB;
params_RDC.P_LgB = P_LgB;
params_RDC.P_BgA = P_BgA;
end