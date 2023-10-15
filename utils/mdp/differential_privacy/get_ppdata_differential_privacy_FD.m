  function [pp_data, ppdata_fileFullPath] = get_ppdata_differential_privacy_FD(paramsin,fileNamePrefix, pp_data_FD)
params = struct;
used_fieldnames = {'x_num', 'h_num', 'y_control_num','a_num','z_num','d_num','y_control_p_pu','y_control_offset',...
    'x_p_pu','x_offset','d_p_pu','d_offset','P_Zp1gZD','P_Zp1gZD','paramsPrecision',...
    'P_HgA','P_XHgHn1','u_num','differential_privacy_epsilon','minPowerDemandInW'};
for fn = used_fieldnames
    params.(fn{1}) = paramsin.(fn{1});
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
    y_control_offset = params.y_control_offset;
    x_offset  = params.x_offset;
    x_p_pu = params.x_p_pu;
    minPowerDemandInW = params.minPowerDemandInW;

    if minPowerDemandInW~= 0
        error('minPowerDemandInW need to be 0')
    end

    x_range = 1:x_num;
    h_range = 1:h_num;

    function_handles = get_vectorizing_function_handles(params);
    HsZ2A = function_handles.HsZ2A;
    XHAn1s_2S = function_handles.XHAn1s_2S;
    YcSs_2U = function_handles.YcSs_2U;

    consumption_sensitivity = params.minPowerDemandInW + (x_num + params.x_offset)*params.x_p_pu;
    differential_privacy_b = consumption_sensitivity/params.differential_privacy_epsilon;

    valid_XgYZn1 = pp_data_FD.valid_XgYZn1;
    valid_YgXZn1 = pp_data_FD.valid_YgXZn1;
    valid_DgXZn1 = pp_data_FD.valid_DgXZn1;
    P_AgU_YcAkn1 = pp_data_FD.P_AgU_YcAkn1;

    P_Uk_differential_privacy = zeros(u_num,1);
    for X_idx = x_range
        x_k_pu = (X_idx + x_offset);
        for z_kn1_idx = 1:z_num
            Akn1_idxs = HsZ2A(h_range, z_kn1_idx);
            valid_YIdxs = valid_YgXZn1{X_idx, z_kn1_idx};
            prob_y_given_x = zeros(length(valid_YIdxs),1);
            for idx = 1:length(valid_YIdxs)
                y_idx = valid_YIdxs(idx);
                y_k_pu = (y_idx + y_control_offset);
                prob_y_given_x(idx) = probYGivenX(x_k_pu, y_k_pu, x_p_pu, differential_privacy_b);
            end
            if sum(prob_y_given_x)==0
                error('here');
            end
            prob_y_given_x = prob_y_given_x/sum(prob_y_given_x);
            for idx = 1:length(valid_YIdxs)
                y_idx = valid_YIdxs(idx);
                for hk_idx = h_range
                    S_idxs = XHAn1s_2S(X_idx, hk_idx, Akn1_idxs);
                    U_idxs = YcSs_2U(y_idx, S_idxs);
                    P_Uk_differential_privacy(U_idxs) = prob_y_given_x(idx);
                end
            end
        end
    end

    pp_data.valid_XgYZn1 = valid_XgYZn1;
    pp_data.valid_YgXZn1 = valid_YgXZn1;
    pp_data.valid_DgXZn1 = valid_DgXZn1;  
    pp_data.P_Uk_differential_privacy = P_Uk_differential_privacy; 
    pp_data.P_AgU_YcAkn1 = P_AgU_YcAkn1; 
    pp_data.params = params;
    save(ppdata_fileFullPath,'pp_data','params','-v7.3')
end

      function prob_y_given_x = probYGivenX(x, y, c, laplacian_scale)
          prob_y_given_x = cdflaplace( (y + 0.5) * c - x * c, 0, laplacian_scale) - ...
              cdflaplace( (y - 0.5) * c - x * c, 0, laplacian_scale);
      end

      function fx = cdflaplace(x, mu, laplacian_scale)
          fx = 0.5*(1 + sign(x-mu).*(1 - exp( -abs(x-mu)/(laplacian_scale) )));
      end
  end

