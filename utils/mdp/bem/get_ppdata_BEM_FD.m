  function [pp_data, ppdata_fileFullPath] = get_ppdata_BEM_FD(params_in,fileNamePrefix, pp_data_FD)
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
    y_control_num = params.y_control_num;

    y_control_range = 1:y_control_num;
    x_range = 1:x_num;
    h_range = 1:h_num;

    function_handles = get_vectorizing_function_handles(params);
    HsZ2A = function_handles.HsZ2A;
    XHAn1s_2S = function_handles.XHAn1s_2S;
    YcSs_2U = function_handles.YcSs_2U;

    valid_XgYZn1 = pp_data_FD.valid_XgYZn1;
    valid_YgXZn1 = pp_data_FD.valid_YgXZn1;
    valid_DgXZn1 = pp_data_FD.valid_DgXZn1;
    P_AgU_YcAkn1 = pp_data_FD.P_AgU_YcAkn1;

    P_UkgYckn1Idx = cell(y_control_num, 1);
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing PP ddata : ');
    incPercent = (1/y_control_num)*100;
    findClosestNumber_ = @findClosestNumber;
    for Yckn1_idx = y_control_range
        HsZ2A_ = HsZ2A;
        valid_YgXZn1_ = valid_YgXZn1;
        XHAn1s_2S_ = XHAn1s_2S;
        YcSs_2U_ = YcSs_2U;

        P_Uk = zeros(u_num,1);
        for X_idx = x_range
            for z_kn1_idx = 1:z_num
                Akn1_idxs = HsZ2A_(h_range, z_kn1_idx);
                valid_YIdxs = valid_YgXZn1_{X_idx, z_kn1_idx};
                Y_idx_BEM = findClosestNumber_(Yckn1_idx, valid_YIdxs);
                for hk_idx = h_range
                    S_idxs = XHAn1s_2S_(X_idx, hk_idx, Akn1_idxs);
                    U_idxs = YcSs_2U_(Y_idx_BEM, S_idxs);
                    P_Uk(U_idxs) = 1;
                end
            end
        end
        P_UkgYckn1Idx{Yckn1_idx} = P_Uk;
        send(progressDataQueue, incPercent);
    end
    progressData.terminate();

    pp_data.valid_XgYZn1 = valid_XgYZn1;    
    pp_data.valid_YgXZn1 = valid_YgXZn1;  
    pp_data.valid_DgXZn1 = valid_DgXZn1;  
    pp_data.P_UkgYckn1Idx = P_UkgYckn1Idx; 
    pp_data.P_AgU_YcAkn1 = P_AgU_YcAkn1; 
    pp_data.params = params;
    save(ppdata_fileFullPath,'pp_data','params','-v7.3')
end

    function closestNumber = findClosestNumber(target, array)
        absoluteDifferences = abs(array - target);
        [~, index] = min(absoluteDifferences);
        closestNumber = array(index);
    end
end

