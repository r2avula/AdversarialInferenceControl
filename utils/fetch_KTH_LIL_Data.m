function [smData,applianceData,gtData,dateStrings] = fetch_KTH_LIL_Data(config)
dataset = config.dataset;
path_to_raw_sm_data = config.path_to_raw_sm_data;
rootdir_raw_aggregate_data = strcat(path_to_raw_sm_data,"aggregate", filesep);
rootdir_raw_appliance_data = strcat(path_to_raw_sm_data,"appliance", filesep);

path_to_processed_sm_data = strcat("resources",filesep,"smart_meter_real_data", filesep);
metaDataFilename = strcat(path_to_processed_sm_data,"meta_data.mat");
if ~exist(metaDataFilename, 'file')
    cache_data = load(strcat(rootdir_raw_appliance_data,"metadata.mat"), 'applianceDataPath');
    applianceDataPath = cache_data.applianceDataPath;
    applianceCount = length(applianceDataPath);

    startDate = '2020_08_28';
    endDate = '2021_05_31';
    totalDays = days(datetime(endDate,'InputFormat', 'yyyy_MM_dd')-datetime(startDate,'InputFormat','yyyy_MM_dd')) + 1;
    dateStrings = strings(totalDays,1);
    for fileIdx = 1:totalDays
    	dateStrings(fileIdx) = string(caldays(fileIdx-1) + datetime(startDate,'InputFormat','yyyy_MM_dd'),'yyyy_MM_dd');
    end

    applianceRecordedDateMatrix = false(applianceCount,totalDays);
    for applianceIdx = 1:applianceCount
        filelist = dir(strcat(rootdir_raw_appliance_data,applianceDataPath(applianceIdx)));
    	fileFlags = ~[filelist.isdir];
    	filelist = filelist(fileFlags);
    	[~,recordedDayIdxs,~] = intersect(dateStrings,erase(string({filelist.name})','.mat'));
        applianceRecordedDateMatrix(applianceIdx,recordedDayIdxs) = true;
    end
    save(metaDataFilename,'applianceDataPath','dateStrings','applianceRecordedDateMatrix')
end
meta_data = load(metaDataFilename);
applianceDataPath = meta_data.applianceDataPath;
dateStrings = meta_data.dateStrings;
applianceRecordedDateMatrix = meta_data.applianceRecordedDateMatrix;

dataType = config.dataType;
applianceGroupsNum = config.applianceGroupsNum;
hypothesisStatesPerAppliance = cell2mat(config.hypothesisStatesPerAppliance)';
applianceGroups = cell(applianceGroupsNum,1);
applianceGroupThresholdBoundaries = cell(applianceGroupsNum,1);
for applianceGroupIdx = 1:applianceGroupsNum
    applianceGroupKeyWord = config.(['applianceGroup',num2str(applianceGroupIdx)]);
    applianceGroupNum = numel(applianceGroupKeyWord);
    applianceGroupNames_t = [];
    for t_idx = 1:applianceGroupNum
        TF = contains(applianceDataPath,applianceGroupKeyWord{t_idx},'IgnoreCase',true);
        applianceGroupNames_t = [applianceGroupNames_t;find(TF)]; %#ok<AGROW> 
    end   
    applianceGroups{applianceGroupIdx} = applianceGroupNames_t;

    boundaries = cell2mat(config.(['applianceGroup',num2str(applianceGroupIdx),'Threshold']));
    if(numel(boundaries)+1~= hypothesisStatesPerAppliance(applianceGroupIdx))
        error('applianceGroupThresholdBoundaries');
    end
    applianceGroupThresholdBoundaries{applianceGroupIdx} = [0 boundaries inf];
end

slotIntervalInSeconds = config.slotIntervalInSeconds;
slotIntervalInHours = slotIntervalInSeconds/3600; %in hours
slot_num_in_day = 24/slotIntervalInHours;

fileNamePrefix = strcat(path_to_processed_sm_data,'smartMeterData_');
dataParams = struct;
dataParams.dataset = dataset;
dataParams.slotIntervalInSeconds = slotIntervalInSeconds;
dataParams.applianceGroups = applianceGroups;
dataParams.dataType = dataType;

[filename,fileExists] = findFileName(dataParams,fileNamePrefix,'dataParams');
if(fileExists)
    load(filename,'smData','applianceData','gtData');
else
    totalDays = length(dateStrings);
    [progressData, progressDataQueue] = ProgressData('\t\t\tProcessing real energy consumption data : ');
    incPercent = (1/totalDays)*100;
    gtData = ones(slot_num_in_day,totalDays,applianceGroupsNum);
    if(strcmp(dataType,'real_reactive'))
        smData = zeros(2,slot_num_in_day,totalDays);
        applianceData = zeros(2,slot_num_in_day,totalDays,applianceGroupsNum);

        for recordedDateIdx = 1:totalDays
            dateString = dateStrings(recordedDateIdx);
            sm_data_fileFullPath = strcat(rootdir_raw_aggregate_data,dateString,".mat");
            varname = strcat("aggregate_",dateString);
            recorded_data = load(sm_data_fileFullPath,varname);
            recorded_data = recorded_data.(varname);
            
            real_smData_t = recorded_data.real_power;
            real_smData_t(isnan(real_smData_t)) = 0;
            real_smData_t = max(real_smData_t,0);

            reactive_smData_t = recorded_data.reactive_power;
            reactive_smData_t(isnan(reactive_smData_t)) = 0;
            reactive_smData_t = max(reactive_smData_t,0);

            if (slotIntervalInSeconds > 1)
                real_smData_t = reshape(real_smData_t,slotIntervalInSeconds,[]);
                reactive_smData_t = reshape(reactive_smData_t,slotIntervalInSeconds,[]);
                smData(1,:,recordedDateIdx) = mean(real_smData_t, 1);
                smData(2,:,recordedDateIdx) = mean(reactive_smData_t, 1);
            else
                smData(1,:,recordedDateIdx) = real_smData_t;
                smData(2,:,recordedDateIdx) = reactive_smData_t;
            end

            for applianceGroupIdx = 1:applianceGroupsNum
                applianceIdxs_t = applianceGroups{applianceGroupIdx}(applianceRecordedDateMatrix(applianceGroups{applianceGroupIdx},recordedDateIdx));
                if(~isempty(applianceIdxs_t))
                    real_applianceData_t = zeros(slot_num_in_day,1);
                    reactive_applianceData_t = zeros(slot_num_in_day,1);
                    for applianceIdx = applianceIdxs_t'
                        applianceName = applianceDataPath(applianceIdx);
                        appliance_data_fileFullPath = strcat(rootdir_raw_appliance_data,applianceName,filesep,dateString,".mat");
                        varname = strcat(strrep(applianceName,'\','_'),'_',dateString);
                        recorded_data = load(appliance_data_fileFullPath,varname);
                        recorded_data = recorded_data.(varname);

                        real_applianceData_tt = recorded_data.real_power;
                        real_applianceData_tt(isnan(real_applianceData_tt)) = 0;
                        real_applianceData_tt = max(real_applianceData_tt,0);

                        reactive_applianceData_tt = recorded_data.reactive_power;
                        reactive_applianceData_tt(isnan(reactive_applianceData_tt)) = 0;
                        reactive_applianceData_tt = max(reactive_applianceData_tt,0);
                        if (slotIntervalInSeconds > 1)
                            real_applianceData_tt = reshape(real_applianceData_tt,slotIntervalInSeconds,[]);
                            real_applianceData_t = real_applianceData_t + mean(real_applianceData_tt, 1)';

                            reactive_applianceData_tt = reshape(reactive_applianceData_tt,slotIntervalInSeconds,[]);
                            reactive_applianceData_t = reactive_applianceData_t + mean(reactive_applianceData_tt, 1)';
                        else
                            real_applianceData_t = real_applianceData_t + real_applianceData_tt;
                            reactive_applianceData_t = reactive_applianceData_t + reactive_applianceData_tt;
                        end
                    end

                    applianceData(1,:,recordedDateIdx,applianceGroupIdx) = real_applianceData_t;
                    applianceData(2,:,recordedDateIdx,applianceGroupIdx) = reactive_applianceData_t;

                    hypothesisStates_t = hypothesisStatesPerAppliance(applianceGroupIdx);
                    boundaries = applianceGroupThresholdBoundaries{applianceGroupIdx};

                    gtData_t = zeros(slot_num_in_day,1);
                    for temp_h_idx = 1:hypothesisStates_t
                        gtData_t(real_applianceData_t>=boundaries(temp_h_idx) & real_applianceData_t<boundaries(temp_h_idx+1)) = temp_h_idx;
                    end
                    gtData(:,recordedDateIdx,applianceGroupIdx) = gtData_t;
                end
            end

            send(progressDataQueue, incPercent);
        end
    elseif(strcmp(dataType,'real'))
        smData = zeros(slot_num_in_day,totalDays);
        applianceData = zeros(slot_num_in_day,totalDays,applianceGroupsNum);

        for recordedDateIdx = 1:totalDays
            dateString = dateStrings(recordedDateIdx);
            sm_data_fileFullPath = strcat(rootdir_raw_aggregate_data,dateString,".mat");
            varname = strcat("aggregate_",dateString);
            recorded_data = load(sm_data_fileFullPath,varname);
            recorded_data = recorded_data.(varname);
            real_smData_t = recorded_data.real_power;
            real_smData_t(isnan(real_smData_t)) = 0;
            real_smData_t = max(real_smData_t,0);

            if (slotIntervalInSeconds > 1)
                real_smData_t = reshape(real_smData_t,slotIntervalInSeconds,[]);
                smData(:,recordedDateIdx) = mean(real_smData_t, 1);
            else
                smData(:,recordedDateIdx) = real_smData_t;
            end

            for applianceGroupIdx = 1:applianceGroupsNum
                applianceIdxs_t = applianceGroups{applianceGroupIdx}(applianceRecordedDateMatrix(applianceGroups{applianceGroupIdx},recordedDateIdx));
                if(~isempty(applianceIdxs_t))
                    real_applianceData_t = zeros(slot_num_in_day,1);
                    for applianceIdx = applianceIdxs_t'
                        applianceName = applianceDataPath(applianceIdx);
                        appliance_data_fileFullPath = strcat(rootdir_raw_appliance_data,applianceName,filesep,dateString,".mat");
                        varname = strcat(strrep(applianceName,'\','_'),'_',dateString);
                        recorded_data = load(appliance_data_fileFullPath,varname);
                        recorded_data = recorded_data.(varname);

                        real_applianceData_tt = recorded_data.real_power;
                        real_applianceData_tt(isnan(real_applianceData_tt)) = 0;
                        real_applianceData_tt = max(real_applianceData_tt,0);
                        if (slotIntervalInSeconds > 1)
                            real_applianceData_tt = reshape(real_applianceData_tt,slotIntervalInSeconds,[]);
                            real_applianceData_t = real_applianceData_t + mean(real_applianceData_tt, 1)';
                        else
                            real_applianceData_t = real_applianceData_t + real_applianceData_tt;
                        end
                    end

                    applianceData(:,recordedDateIdx,applianceGroupIdx) = real_applianceData_t;

                    hypothesisStates_t = hypothesisStatesPerAppliance(applianceGroupIdx);
                    boundaries = applianceGroupThresholdBoundaries{applianceGroupIdx};

                    gtData_t = zeros(slot_num_in_day,1);
                    for temp_h_idx = 1:hypothesisStates_t
                        gtData_t(real_applianceData_t>=boundaries(temp_h_idx) & real_applianceData_t<boundaries(temp_h_idx+1)) = temp_h_idx;
                    end
                    gtData(:,recordedDateIdx,applianceGroupIdx) = gtData_t;
                end
            end

            send(progressDataQueue, incPercent);
        end
    else
        error('not implemented');
    end
    progressData.terminate();

    save(filename,'smData','applianceData','gtData','dataParams');
end

controlStartHourIndex = config.controlStartTime+1;
controlEndHourIndex = config.controlEndTime;

slot_num_in_hour = slot_num_in_day/24;
controlStartSlotIndex = (controlStartHourIndex-1)*slot_num_in_hour + 1;
controlEndSlotIndex = (controlEndHourIndex)*slot_num_in_hour;

if(strcmp(dataType,'real_reactive'))
    smData = smData(:, controlStartSlotIndex:controlEndSlotIndex,:);
    applianceData = applianceData(:, controlStartSlotIndex:controlEndSlotIndex,:,:);
elseif(strcmp(dataType,'real'))
    smData = smData(controlStartSlotIndex:controlEndSlotIndex,:);
    applianceData = applianceData(controlStartSlotIndex:controlEndSlotIndex,:,:);
else
    error('not implemented');
end
gtData = gtData(controlStartSlotIndex:controlEndSlotIndex,:,:);
end

