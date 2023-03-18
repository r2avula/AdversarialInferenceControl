function [pp_data, ppdata_fileFullPath] = get_ppdata_RD(params_in,fileNamePrefix)
params = struct;
used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
    'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','P_Zp1gZD','paramsPrecision','minLikelihoodFilter',...
    'beliefSpacePrecision_adv','P_HgHn1','P_XgH','P_HgA','k_num','P_ZgA','P_H0','b_num','P_XHgHn1','ZIdxsgLIdx','l_num','t_num','w_num'};
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
    l_num = params.l_num;
    y_control_num = params.y_control_num;
    y_control_p_pu = params.y_control_p_pu;
    y_control_offset = params.y_control_offset;
    x_p_pu = params.x_p_pu;
    x_offset = params.x_offset;
    d_p_pu = params.d_p_pu;
    d_offset = params.d_offset;
    P_Zp1gZD = params.P_Zp1gZD;
    paramsPrecision = params.paramsPrecision;
    d_num = params.d_num;
    h_num = params.h_num;
    z_num = params.z_num;
    w_num = params.w_num;
    a_num = params.a_num;
    b_num = params.b_num;
    P_XHgHn1 = params.P_XHgHn1;
    ZIdxsgLIdx = params.ZIdxsgLIdx;

    function_handles = get_vectorizing_function_handles(params);
    XsB_2T = function_handles.XsB_2T;
    YTs_2W = function_handles.YTs_2W;
    HZs2A = function_handles.HZs2A;
    HL2B = function_handles.HL2B;

    y_control_range = 1:y_control_num;
    x_range = 1:x_num;
    h_range = 1:h_num;
    l_range = 1:l_num;

    valid_DgYLn1 = cell(y_control_num, l_num);
    valid_XgYLn1 = cell(y_control_num, l_num);
    valid_YgXLn1 = cell(x_num, l_num);
    P_Akn1W_YcBk = repmat({sparse(a_num, w_num)}, [y_control_num, b_num]);
    P_AgW_YcAkn1 = cell(y_control_num, a_num);

    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing PP ddata : ');
    incPercent = (1/l_num)*100;

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

    for l_kn1_idx = 1:l_num
        z_kn1_idxs = ZIdxsgLIdx(:,l_kn1_idx)';
        min_d_idx = 1;
        max_d_idx = d_num;
        for z_kn1_idx = z_kn1_idxs
            valid_d_idxs_flag = reshape(sum(P_Zp1gZD(:,z_kn1_idx,:),1)>=paramsPrecision,[],1);
            min_d_idx = max(min_d_idx, find(valid_d_idxs_flag, 1, 'first'));
            max_d_idx = min(max_d_idx, find(valid_d_idxs_flag, 1, 'last'));
        end

        for Yck_idx = y_control_range
            valid_DIdxs = round(((Yck_idx+y_control_offset)*y_control_p_pu - (x_range+x_offset)*x_p_pu)/d_p_pu) - d_offset;
            valid_XIdxs_flag = valid_DIdxs>=min_d_idx & valid_DIdxs<=max_d_idx;
            valid_XIdxs = x_range(valid_XIdxs_flag);
            valid_DIdxs = valid_DIdxs(valid_XIdxs_flag);
            valid_DgYLn1{Yck_idx, l_kn1_idx} = valid_DIdxs;
            valid_XgYLn1{Yck_idx, l_kn1_idx} = valid_XIdxs;
        end

        for X_idx = x_range
            valid_DIdxs = round(((y_control_range+y_control_offset)*y_control_p_pu - (X_idx+x_offset)*x_p_pu)/d_p_pu) - d_offset;
            valid_YIdxs_flag = valid_DIdxs>=min_d_idx & valid_DIdxs<=max_d_idx;
            valid_YgXLn1{X_idx, l_kn1_idx} = y_control_range(valid_YIdxs_flag);
        end

        for Yck_idx = y_control_range
            valid_XIdxs = valid_XgYLn1{Yck_idx, l_kn1_idx};
            valid_DIdxs = valid_DgYLn1{Yck_idx, l_kn1_idx};
            
            for z_kn1_idx = z_kn1_idxs
                P_Zp1gZD_ = reshape(P_Zp1gZD(:,z_kn1_idx,:),z_num,d_num);
                for hkn1_idx = h_range
                    Akn1_idx = HZs2A(hkn1_idx, z_kn1_idx);
                    P_AgW = zeros(a_num,w_num);
                    for hk_idx = h_range
                        WIdxs = YTs_2W(Yck_idx,XsB_2T(valid_XIdxs, HL2B(hk_idx, l_kn1_idx)));
                        for l_k_idx = l_range
                            P_Akn1W_YcBk{Yck_idx, HL2B(hk_idx, l_k_idx)}(Akn1_idx, WIdxs) = P_Akn1W_YcBk{Yck_idx, HL2B(hk_idx, l_k_idx)}(HZs2A(hkn1_idx, z_kn1_idx), WIdxs) + ...
                                sum(P_Zp1gZD_(ZIdxsgLIdx(:,l_k_idx),valid_DIdxs),1).*P_XHgHn1(valid_XIdxs,hk_idx,hkn1_idx)';
                        end
                        P_AgW(HZs2A(hk_idx, 1:z_num), WIdxs) = P_Zp1gZD_(:,valid_DIdxs).*repmat(P_XHgHn1(valid_XIdxs,hk_idx,hkn1_idx)', z_num, 1);
                    end
                    P_AgW_YcAkn1{Yck_idx, Akn1_idx} = sparse(P_AgW);
                end
            end
        end
        send(progressDataQueue, incPercent);
    end
    progressData.terminate();
    pp_data.valid_DgYLn1 = valid_DgYLn1;   
    pp_data.valid_XgYLn1 = valid_XgYLn1;    
    pp_data.valid_YgXLn1 = valid_YgXLn1;  
    pp_data.valid_YgX = valid_YgX;      
    pp_data.P_AgW_YcAkn1 = P_AgW_YcAkn1;  
    pp_data.P_Akn1W_YcBk = P_Akn1W_YcBk;        
    pp_data.params = params;
    save(ppdata_fileFullPath,'pp_data','params')
end
end

