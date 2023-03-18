function [config] = getHMMParams(config,appliances_consumption,noise,appliances_state)
x_num = config.x_num;
x_offset = config.x_offset;
x_p_pu = config.x_p_pu;
h_num = config.h_num;
k_num = config.k_num;
paramsPrecision = config.paramsPrecision;
minPowerDemandInW = config.minPowerDemandInW;
h_vec_space = config.h_vec_space;
hypothesisStatesPerAppliance = config.hypothesisStatesPerAppliance;
appliances_num = size(h_vec_space,1);

xa_k_idxs = min(max(1,round((appliances_consumption-minPowerDemandInW)/x_p_pu)-x_offset),x_num);
na_k_idxs = min(max(1,round((noise-minPowerDemandInW)/x_p_pu)-x_offset),x_num);
totalDays = size(appliances_consumption,2);

P_Ha0 = cell(appliances_num,1);
P_WgHa = cell(appliances_num,1);
P_HagHan1 = cell(appliances_num,1);
for app_idx = 1:appliances_num
    ha_num = hypothesisStatesPerAppliance(app_idx);
    xa_k_idxs_t = xa_k_idxs(:,:,app_idx);
    state_data = appliances_state(:,:,app_idx);

    P_H0_t = zeros(ha_num,1);
    for ha_idx = 1:ha_num
        P_H0_t(ha_idx) = sum(state_data==ha_idx,'all');
    end
    P_H0_t = P_H0_t/sum(P_H0_t);
    P_Ha0{app_idx} = P_H0_t;


    P_WgH_t = zeros(x_num,ha_num);
    for ha_idx = 1:ha_num
        for x_k_idx = 1:x_num
            P_WgH_t(x_k_idx,ha_idx) = sum(xa_k_idxs_t(:)==x_k_idx & state_data(:)==ha_idx);
        end

        temp_sum = sum(P_WgH_t(:,ha_idx));
        if(temp_sum>0)
            P_WgH_t(:,ha_idx) = P_WgH_t(:,ha_idx)/temp_sum;
        else
            P_WgH_t(:,ha_idx) = 1/x_num;
        end
    end
    P_WgHa{app_idx} = P_WgH_t;

    prev_state_data = [state_data(k_num,:);state_data(1:end-1,:)];
    P_HagHan1_t = zeros(ha_num,ha_num);
    for hakn1_idx = 1:ha_num
        for ha_idx = 1:ha_num
            P_HagHan1_t(ha_idx,hakn1_idx) = sum(prev_state_data(:)==hakn1_idx & state_data(:)==ha_idx);
        end
        temp_sum = sum(P_HagHan1_t(:,hakn1_idx));
        if(temp_sum>0)
            P_HagHan1_t(:,hakn1_idx) = P_HagHan1_t(:,hakn1_idx)/temp_sum;
        else
            P_HagHan1_t(:,hakn1_idx) = 1/ha_num;
        end
    end
    P_HagHan1{app_idx} = P_HagHan1_t;
end

P_Nk = zeros(x_num,1);
for x_k_idx = 1:x_num
    P_Nk(x_k_idx) = sum(na_k_idxs(:)==x_k_idx);
end

noise_exists = true;

temp_sum = sum(P_Nk);
if(temp_sum>0)
    P_Nk = P_Nk/temp_sum;
else
    noise_exists = false;
end

P_H0 = zeros(h_num,1);
P_XgH = zeros(x_num,h_num);
for h_vec_idx = 1:h_num
    h_vec = h_vec_space(:,h_vec_idx);
    if length(P_Ha0)>1
        P_H0(h_vec_idx) = P_Ha0{1}(h_vec(1))*P_Ha0{2}(h_vec(2));
        for app_idx = 3:appliances_num
            P_H0(h_vec_idx) = P_H0(h_vec_idx)*P_Ha0{app_idx}(h_vec(app_idx));
        end
    else
        P_H0 = P_Ha0{1};
    end

    if length(P_WgHa)>1
        P_XgH_t = conv(P_WgHa{1}(:,h_vec(1)), P_WgHa{2}(:,h_vec(2)));
        P_XgH_t_excess_sum = sum(P_XgH_t(x_num+1:end));
        P_XgH_t = P_XgH_t(1:x_num);
        P_XgH_t(x_num) = P_XgH_t(x_num) + P_XgH_t_excess_sum;
        P_XgH_t = P_XgH_t/sum(P_XgH_t);

        for app_idx = 3:appliances_num
            P_XgH_t = conv(P_XgH_t, P_WgHa{app_idx}(:,h_vec(app_idx)));
            P_XgH_t_excess_sum = sum(P_XgH_t(x_num+1:end));
            P_XgH_t = P_XgH_t(1:x_num);
            P_XgH_t(x_num) = P_XgH_t(x_num) + P_XgH_t_excess_sum;
            P_XgH_t = P_XgH_t/sum(P_XgH_t);
        end
    else
        P_XgH_t = P_WgHa{1}(:,h_vec(1));
    end

    if(noise_exists)
        P_XgH_t = conv(P_XgH_t, P_Nk);
        P_XgH_t_excess_sum = sum(P_XgH_t(x_num+1:end));
        P_XgH_t = P_XgH_t(1:x_num);
        P_XgH_t(x_num) = P_XgH_t(x_num) + P_XgH_t_excess_sum;
        P_XgH_t = P_XgH_t/sum(P_XgH_t);
    end

    P_XgH(:,h_vec_idx) = roundOffInSimplex(P_XgH_t,paramsPrecision);
end

h_vec_data = zeros(k_num,totalDays);
for dayIdx = 1:totalDays
    [~,h_vec_data(:,dayIdx)] = ismember(reshape(appliances_state(:,dayIdx,:),k_num,[]),h_vec_space','rows');
end

P_HgHn1 = zeros(h_num,h_num);
if appliances_num > 1
    possible_transitions_h_vec = false(h_num,h_num);
    for h_vec_kn1_idx = 1:h_num
        h_vec_kn1 = h_vec_space(:,h_vec_kn1_idx);
        P_Hk_vec_gHkn1_vec_t = zeros(h_num,1);
        for app_idx = 1:appliances_num
            h_vec_t = h_vec_kn1;
            h_vec_t(app_idx) = double(~logical(h_vec_t(app_idx) - 1)) + 1;
            [~, h_vec_t_idx] = ismember(h_vec_t',h_vec_space','row');

            possible_transitions_h_vec(h_vec_t_idx,h_vec_kn1_idx) = true;

            temp_t = P_HagHan1{1}(h_vec_t(1),h_vec_kn1(1))*P_HagHan1{2}(h_vec_t(2),h_vec_kn1(2));
            for app_idx_t = 3:appliances_num
                temp_t = temp_t*P_HagHan1{app_idx_t}(h_vec_t(app_idx_t),h_vec_kn1(app_idx_t));
            end
            P_Hk_vec_gHkn1_vec_t(h_vec_t_idx) = temp_t;
        end

        h_vec_t = h_vec_kn1;
        h_vec_t_idx = h_vec_kn1_idx;
        possible_transitions_h_vec(h_vec_t_idx,h_vec_kn1_idx) = true;

        temp_t = P_HagHan1{1}(h_vec_t(1),h_vec_kn1(1))*P_HagHan1{2}(h_vec_t(2),h_vec_kn1(2));
        for app_idx_t = 3:appliances_num
            temp_t = temp_t*P_HagHan1{app_idx_t}(h_vec_t(app_idx_t),h_vec_kn1(app_idx_t));
        end
        P_Hk_vec_gHkn1_vec_t(h_vec_t_idx) = temp_t;


        temp_sum = sum(P_Hk_vec_gHkn1_vec_t);
        if(temp_sum>0)
            P_Hk_vec_gHkn1_vec_t = P_Hk_vec_gHkn1_vec_t/temp_sum;
        else
            P_Hk_vec_gHkn1_vec_t = 1/h_num;
        end

        P_HgHn1(:,h_vec_kn1_idx) = roundOffInSimplex(P_Hk_vec_gHkn1_vec_t,paramsPrecision);
    end
else
    P_HgHn1 = P_HagHan1{1};
end

% possible_h_vec_idxs = cell(h_num,1);
% for h_vec_idx = 1:h_num
%     possible_h_vec_idxs{h_vec_idx} = find(possible_transitions_h_vec(:,h_vec_idx))';
% end

P_XHgHn1 = zeros(x_num,h_num,h_num);
for h_kn1_idx=1:h_num
    %     P_XH_ = zeros(x_num,h_num);
    for h_k_idx = 1:h_num
        for x_k_idx=1:x_num
            P_XHgHn1(x_k_idx,h_k_idx,h_kn1_idx) = roundOff(P_HgHn1(h_k_idx,h_kn1_idx)*P_XgH(x_k_idx,h_k_idx),paramsPrecision);
        end
    end

    %     P_XH_vec = P_XH_(:);
    %     if(sum(P_XH_vec)>=paramsPrecision)
    %         P_XHgHn1(:,:,h_kn1_idx) = reshape(roundOffInSimplex(P_XH_vec,paramsPrecision),[x_num,h_num]);
    %     end
end

% z_num = params.z_num;
% P_HgA = params.P_HgA;
% P_ZgA = params.P_ZgA;
% P_BgA = params.P_BgA;
% a_num = params.a_num;
% A2HZ = params.A2HZ;
% P_Z0 = ones(z_num,1)/z_num;
% P_Z0 = roundOffInSimplex(P_Z0,paramsPrecision);
P_H0 = roundOffInSimplex(P_H0,paramsPrecision);

% P_A0 = zeros(a_num,1);
% for a_idx = 1:a_num
%     h_k_idx = A2HZ{a_idx}(1);
%     z_k_idx = A2HZ{a_idx}(2);
%     P_A0(a_idx) = P_H0(h_k_idx)*P_Z0(z_k_idx);
% end
% P_A0 = P_A0/sum(P_A0);
% P_A0 = roundOffInSimplex(P_A0,paramsPrecision);
% P_H0 = roundOffInSimplex(P_HgA*P_A0,paramsPrecision);
% P_Z0 = roundOffInSimplex(P_ZgA*P_A0,paramsPrecision);
% P_B0 = roundOffInSimplex(P_BgA*P_A0,paramsPrecision);

% params.P_A0 = P_A0;
% params.P_Z0 = P_Z0;
% params.P_B0 = P_B0;

if(~all(sum(P_XgH,1)>1-1e-12,'all'))
    error('~all(sum(P_XgH,1)==1,''all'')');
end

if(~all(sum(P_HgHn1,1)>1-1e-12,'all'))
    error('~all(sum(P_HgHn1,1)==1,''all'')');
end

% if(~all(sum(P_XHgHn1,[1,2])>1-1e-6 | sum(P_XHgHn1,[1,2])<1e-6,'all'))
%     error('~all(sum(P_XHgHn1,1)==1 | sum(P_XHgHn1,1)==0,''all'')');
% end

config.P_H0 = P_H0;
config.P_XgH = P_XgH;
config.P_HgHn1 = P_HgHn1;
config.P_XHgHn1 = P_XHgHn1;
end
