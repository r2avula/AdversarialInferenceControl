function [fscore_per_appliance, tp_rate_recall_per_appliance, tn_rate_per_appliance, precision_per_appliance] = computeFScores(params, gt_data, detected_data)
h_vec_space = params.h_vec_space;
hypothesisStatesPerAppliance = params.hypothesisStatesPerAppliance;
appliances_num = length(hypothesisStatesPerAppliance);
appliance_gt_data_available = false;
if size(gt_data,3)>1
    appliance_gt_data_available = true;
else
    gt_data = gt_data(:);
end
detected_data = detected_data(:);
fscore_per_appliance =  zeros(appliances_num,1);
precision_per_appliance =  zeros(appliances_num,1);
tp_rate_recall_per_appliance =  zeros(appliances_num,1);
tn_rate_per_appliance =  zeros(appliances_num,1);
for app_idx = 1:appliances_num
    ha_num = hypothesisStatesPerAppliance(app_idx);
    if ha_num~= 2
        error('not implemented')
    end
    if appliance_gt_data_available
        h_data_t = reshape(gt_data(:,:,app_idx),[],1);
    else
        h_data_t = h_vec_space(app_idx,gt_data)';
    end
    hh_data_t = h_vec_space(app_idx,detected_data)';

    tp = sum(h_data_t==2 & hh_data_t ==2);
    fp = sum(h_data_t==1 & hh_data_t ==2);
    fn = sum(h_data_t==2 & hh_data_t ==1);
    tn = sum(h_data_t==1 & hh_data_t ==1);
    fscore_per_appliance(app_idx) = 2*tp/(2*tp + fp + fn);
    precision_per_appliance(app_idx) = tp/sum(hh_data_t==2);
    tp_rate_recall_per_appliance(app_idx) = tp/sum(h_data_t==2);
    tn_rate_per_appliance(app_idx) = tn/sum(h_data_t==1);
end
end

