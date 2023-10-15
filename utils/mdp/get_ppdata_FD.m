function [pp_data, ppdata_fileFullPath] = get_ppdata_FD(params_in,fileNamePrefix)
params = struct;
used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
    'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','P_Zp1gZD','paramsPrecision',...
    'P_HgA','P_XHgHn1','u_num'};
for fn = used_fieldnames
    params.(fn{1}) = params_in.(fn{1});
end

initialize_pp_data = true;
[ppdata_fileFullPath,fileExists] = findFileName(params,fileNamePrefix,'params');
if(fileExists)
    saved_data = load(ppdata_fileFullPath,'pp_data');
    if isfield(saved_data,'pp_data')
        pp_data = saved_data.pp_data;
        initialize_pp_data = false;
    end
end

if initialize_pp_data
    x_num = params.x_num;
    z_num = params.z_num;
    h_num = params.h_num;
    u_num = params.u_num;
    a_num = params.a_num;
    d_num = params.d_num;
    y_control_num = params.y_control_num;
    y_control_p_pu = params.y_control_p_pu;
    y_control_offset = params.y_control_offset;
    x_p_pu = params.x_p_pu;
    x_offset = params.x_offset;
    d_p_pu = params.d_p_pu;
    d_offset = params.d_offset;
    P_Zp1gZD = params.P_Zp1gZD;
    paramsPrecision = params.paramsPrecision;
    P_XHgHn1 = params.P_XHgHn1;

    y_control_range = 1:y_control_num;
    x_range = 1:x_num;
    h_range = 1:h_num;
    z_range = 1:z_num;

    function_handles = get_vectorizing_function_handles(params);
    XsHAn1_2S = function_handles.XsHAn1_2S;
    YcSs_2U = function_handles.YcSs_2U;
    HZs2A = function_handles.HZs2A;
    HZ2A = function_handles.HZ2A;

    valid_DgYZn1 = cell(y_control_num, z_num);
    valid_XgYZn1 = cell(y_control_num, z_num);
    valid_YgXZn1 = cell(x_num, z_num);
    P_AgU_YcAkn1 = cell(y_control_num, a_num);
    valid_DgXZn1 = cell(x_num, z_num);

    valid_Xidxs_gYc = cell(y_control_num, 1);
    for Yck_idx = y_control_range
        valid_DIdxs = round(((Yck_idx+y_control_offset)*y_control_p_pu - (x_range+x_offset)*x_p_pu)/d_p_pu) - d_offset;
        valid_XIdxs_flag = valid_DIdxs>=1 & valid_DIdxs<=d_num;
        valid_Xidxs_gYc{Yck_idx} = x_range(valid_XIdxs_flag);
    end

    valid_YgX = cell(x_num, 1);
    for X_idx = x_range
        valid_DIdxs = round(((y_control_range+y_control_offset)*y_control_p_pu - (X_idx+x_offset)*x_p_pu)/d_p_pu) - d_offset;
        valid_YIdxs_flag = valid_DIdxs>=1 & valid_DIdxs<=d_num;
        valid_YgX{X_idx} = y_control_range(valid_YIdxs_flag);
    end

    for z_kn1_idx = 1:z_num
        valid_d_idxs_flag = reshape(sum(P_Zp1gZD(:,z_kn1_idx,:),1)>=paramsPrecision,[],1);
        min_d_idx = find(valid_d_idxs_flag, 1, 'first');
        max_d_idx = find(valid_d_idxs_flag, 1, 'last');

        for Yck_idx = y_control_range
            valid_DIdxs = round(((Yck_idx+y_control_offset)*y_control_p_pu - (x_range+x_offset)*x_p_pu)/d_p_pu) - d_offset;
            valid_XIdxs_flag = valid_DIdxs>=min_d_idx & valid_DIdxs<=max_d_idx;
            valid_XIdxs = x_range(valid_XIdxs_flag);
            valid_DIdxs = valid_DIdxs(valid_XIdxs_flag);
            valid_DgYZn1{Yck_idx, z_kn1_idx} = valid_DIdxs;
            valid_XgYZn1{Yck_idx, z_kn1_idx} = valid_XIdxs;
        end

        for X_idx = x_range
            valid_DIdxs = round(((y_control_range+y_control_offset)*y_control_p_pu - (X_idx+x_offset)*x_p_pu)/d_p_pu) - d_offset;
            valid_YIdxs_flag = valid_DIdxs>=min_d_idx & valid_DIdxs<=max_d_idx;
            valid_YgXZn1{X_idx, z_kn1_idx} = y_control_range(valid_YIdxs_flag);
            valid_DgXZn1{X_idx, z_kn1_idx} = valid_DIdxs;
        end
    end

    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing PP ddata : ');
    incPercent = (1/y_control_num/z_num)*100;
    for Yck_idx = y_control_range
        P_Zp1gZD_t = P_Zp1gZD;
        HZ2A_ = HZ2A;
        YcSs_2U_ = YcSs_2U;
        XsHAn1_2S_ = XsHAn1_2S;
        P_XHgHn1_ = P_XHgHn1;
        HZs2A_ = HZs2A;

        P_AgU_YcAkn1_ = cell(1, a_num);
        for z_kn1_idx = 1:z_num
            valid_XIdxs = valid_XgYZn1{Yck_idx, z_kn1_idx};
            valid_DIdxs = valid_DgYZn1{Yck_idx, z_kn1_idx};
            P_Zp1gZD_ = reshape(P_Zp1gZD_t(:,z_kn1_idx,:),z_num,d_num);
            for hkn1_idx = h_range
                Akn1_idx = HZ2A_(hkn1_idx, z_kn1_idx);
                P_AgU = zeros(a_num,u_num);
                for hk_idx = h_range
                    UIdxs = YcSs_2U_(Yck_idx,XsHAn1_2S_(valid_XIdxs, hk_idx, Akn1_idx));
                    P_AgU(HZs2A_(hk_idx, z_range), UIdxs) = P_Zp1gZD_(:,valid_DIdxs).*repmat(P_XHgHn1_(valid_XIdxs,hk_idx,hkn1_idx)', z_num, 1);
                end
                P_AgU(P_AgU<paramsPrecision)=0;
                P_AgU_YcAkn1_{Akn1_idx} = sparse(P_AgU);
            end
            send(progressDataQueue, incPercent);
        end
        P_AgU_YcAkn1(Yck_idx,:) = P_AgU_YcAkn1_;
    end
    progressData.terminate();

    pp_data.valid_XgYZn1 = valid_XgYZn1;    
    pp_data.valid_YgXZn1 = valid_YgXZn1; 
    pp_data.valid_DgXZn1 = valid_DgXZn1;   
    pp_data.P_AgU_YcAkn1 = P_AgU_YcAkn1; 

    pp_data.params = params;
    save(ppdata_fileFullPath,'pp_data','params','-v7.3')
end
end

