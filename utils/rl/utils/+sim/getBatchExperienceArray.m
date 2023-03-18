function Experiences = getBatchExperienceArray(ExpStructArray,ObservationDimension,ActionDimension)
% Based on BATCHEXPERIENCEARRAY

% Revised: 8-9-2021
% Copyright 2021 The MathWorks, Inc.

%% NOTE SAMPLE WILL NOT COPY CUSTOM EXTRA FIELDS

if isempty(ExpStructArray)
    Experiences = struct.empty;
else
    ExpStructNames = string(fieldnames(ExpStructArray));
    if any(ExpStructNames.matches("NextObservation"))
        NextObservationArray = cell(numel(ObservationDimension),1);
        for ct = 1:numel(ObservationDimension)
            BatchDim = numel(ObservationDimension{ct})+1;
            % NextObservation
            NextObservation = arrayfun(@(x) (x.NextObservation{ct}), ExpStructArray, 'UniformOutput', false);
            NextObservationArray{ct} = cat(BatchDim, NextObservation{:});
        end
        Experiences.NextObservation = NextObservationArray;
    end
    if any(ExpStructNames.matches("Action"))
        ActionArray = cell(numel(ActionDimension),1);
        Action = [ExpStructArray.Action];
        for ct = 1:numel(ActionDimension)
            BatchDim = numel(ActionDimension{ct})+1;
            ActionArray{ct} = cat(BatchDim,Action{ct,:});
        end
        Experiences.Action = ActionArray;    
    end
    if any(ExpStructNames.matches("Observation"))
        ObservationArray = cell(numel(ObservationDimension),1);
        for ct = 1:numel(ObservationDimension)
            BatchDim = numel(ObservationDimension{ct})+1;
            % Observation
            Observation = arrayfun(@(x) (x.Observation{ct}), ExpStructArray, 'UniformOutput', false);
            ObservationArray{ct} = cat(BatchDim, Observation{:});
        end
        Experiences.Observation = ObservationArray;    
    end
    if any(ExpStructNames.matches("Reward"))
        Experiences.Reward = [ExpStructArray.Reward];    
    end
    if any(ExpStructNames.matches("IsDone"))
        Experiences.IsDone = [ExpStructArray.IsDone];    
    end
    if any(ExpStructNames.matches("P_Aks"))
        P_Aks = arrayfun(@(x) (x.P_Aks), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Aks =  cat(2, P_Aks{:});
    end
    if any(ExpStructNames.matches("P_Bks"))
        P_Bks = arrayfun(@(x) (x.P_Bks), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Bks =  cat(2, P_Bks{:});
    end
    if any(ExpStructNames.matches("P_Hks"))
        P_Hks = arrayfun(@(x) (x.P_Hks), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Hks =  cat(2, P_Hks{:});
    end
    if any(ExpStructNames.matches("P_YksgY12kn1"))
        P_YksgY12kn1 = arrayfun(@(x) (x.P_YksgY12kn1), ExpStructArray, 'UniformOutput', false);
        Experiences.P_YksgY12kn1 =  cat(2, P_YksgY12kn1{:});
    end
    if any(ExpStructNames.matches("P_Uk_adv"))
        P_Uk_adv = arrayfun(@(x) (x.P_Uk_adv), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Uk_adv =  cat(2, P_Uk_adv{:});
    end
    if any(ExpStructNames.matches("P_Uk_control"))
        P_Uk_control = arrayfun(@(x) (x.P_Uk_control), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Uk_control =  cat(2, P_Uk_control{:});
    end
    if any(ExpStructNames.matches("P_Wk"))
        P_Wk = arrayfun(@(x) (x.P_Wk), ExpStructArray, 'UniformOutput', false);
        Experiences.P_Wk =  cat(2, P_Wk{:});
    end
    if any(ExpStructNames.matches("AdversarialRewardEstimate"))
        Experiences.AdversarialRewardEstimate = [ExpStructArray.AdversarialRewardEstimate];
    end
    if any(ExpStructNames.matches("penalty"))
        Experiences.penalty = [ExpStructArray.penalty];
    end
    if any(ExpStructNames.matches("MeanAdversarialRewardEstimate"))
        Experiences.MeanAdversarialRewardEstimate = [ExpStructArray.MeanAdversarialRewardEstimate];
    end
end