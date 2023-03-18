classdef RLTrainingResult
    % rlTrainingResult Training statistics result.
    properties (SetAccess = private)
        %         ActorLoss
        %         CriticLoss
        Loss
        % Training episode indicies
        EpisodeIndex
        % Cumulative reward per episode
        EpisodeReward
        % Cumulative adversarial reward per episode
        AdversarialEpisodeRewardEstimate
        % Number of steps in an episode
        EpisodeSteps
        % Window average of EpisodeReward
        AverageEpisodeReward
        % Window average of AdversarialEpisodeRewardEstimate
        AverageAdversarialEpisodeRewardEstimate
        % Total number of agent steps
        TotalAgentSteps
        % Window average of TotalAgentSteps
        AverageSteps
        % Episode Q0 value per episode
        EpisodeQ0
        % Simulation information
        SimulationInfo        
    end

    properties
        % Training options
        TrainingOptions
        LossNum
    end

    % Hidden data
    properties (Hidden, SetAccess = private)
        % Time stamp of logged episode info
        TimeStamp
        % Episode Manager session id
        SessionId
        % Information struct containing meta data
        Information
        % RL App specific meta data
        AppData
    end

    %% Public api
    methods
        function this = RLTrainingResult(trainOpts,sessionId, loss_num)
            % RESULT = rlTrainingResult(TRAINOPTS) creates a default
            % training result object.
            arguments
                trainOpts (1,1)
                sessionId = []
                loss_num = 2
            end
            this.TrainingOptions = trainOpts;
            this.SessionId = sessionId;
            this.LossNum = loss_num;
            this = reset(this);
        end

        function this = set.TrainingOptions(this,trainOpts)
            arguments
                this
                trainOpts (1,1)
            end
            this.TrainingOptions = trainOpts;
        end

        function this = setTrainingOptions(this,trainOpts)
            % RESULTARRAY = setTrainingOptions(RESULTARRAY,TRAINOPTS) sets
            % the training options in each element of RESULTARRAY.
            % RESULTARRAY is an array of rlTrainingResult objects.
            for idx = 1:numel(this)
                this(idx).TrainingOptions = trainOpts;
            end
        end

        function episodemanager = plot(this)
            % plot(RESULT) plots the training result in the Episode Manager.
            episodemanager = episodeManager.createEpisodeManager("reconstruct",Checkpoint=this);
            show(episodemanager);
        end
    end

    %% Hidden methods
    methods (Hidden)
        function this = setSimulationInfo(this,simInfo)
            % Set the SimulationInfo property
            arguments
                this
                simInfo
            end
            this.SimulationInfo = simInfo;
        end

        function this = setInformation(this,info)
            % Set Information property
            arguments
                this
                info struct
            end
            mustBeMember(fields(info), { 'EnvironmentName', ...
                'AgentName', ...
                'BlockPath', ...
                'TrainingOpts', ...
                'HasCritic', ...
                'HardwareResource', ...
                'LearningRate', ...
                'TrainingStartTime', ...
                'ElapsedTime', ...
                'TimeStamp', ...
                'StopTrainingCriteria', ...
                'StopTrainingValue', ...
                'FinalResult',...
                'TotalAgentLearnSteps'});
            this.Information = info;
        end

        function this = setAppData(this,appData)
            % Set app specific data.
            arguments
                this
                appData struct
            end
            mustBeMember({ 'PreviewTimeStamp', 'AgentName', ...
                'EnvironmentName' }, fields(appData));
            this.AppData = appData;
        end

        function this = update(this,data,index)
            % Update the episode info at the specified index from data
            % DATA may be a struct (episode update case) or a training
            % result object (load checkpoint case)
            % INDEX may be a scalar (episode update case) or an array
            % (load checkpoint case)
            arguments
                this
                data  (1,1) {mustBeA(data,{'struct','train.RLTrainingResult'})}
                index (1,:) double
            end
            this.TimeStamp      (index,:) = data.TimeStamp;
            this.EpisodeIndex   (index,:) = data.EpisodeIndex;
            %             this.ActorLoss   (index,:) = data.ActorLoss;
            %             this.CriticLoss   (index,:) = data.CriticLoss;
            if ~isempty(data.Loss)
                this.Loss   (:,index,:) = data.Loss;
            end
            this.EpisodeReward  (index,:) = data.EpisodeReward;
            this.AdversarialEpisodeRewardEstimate  (index,:) = data.AdversarialEpisodeRewardEstimate;
            this.EpisodeSteps   (index,:) = data.EpisodeSteps;
            this.AverageEpisodeReward  (index,:) = data.AverageEpisodeReward;
            this.AverageAdversarialEpisodeRewardEstimate  (index,:) = data.AverageAdversarialEpisodeRewardEstimate;
            this.TotalAgentSteps(index,:) = data.TotalAgentSteps;
            this.AverageSteps   (index,:) = data.AverageSteps;
            if ~isempty(data.EpisodeQ0)
                this.EpisodeQ0  (index,:) = data.EpisodeQ0;
            end
        end

        function this = remove(this,rmidx)
            % Remove entries with index rmidx
            arguments
                this
                rmidx double {mustBePositive}
            end
            this.EpisodeIndex   (rmidx,:) = [];
            this.EpisodeReward  (rmidx,:) = [];
            %             this.ActorLoss   (rmidx,:) = [];
            %             this.CriticLoss   (rmidx,:) = [];
            this.Loss   (:,rmidx,:) = [];
            this.AdversarialEpisodeRewardEstimate  (rmidx,:) = [];
            this.EpisodeSteps   (rmidx,:) = [];
            this.AverageEpisodeReward  (rmidx,:) = [];
            this.AverageAdversarialEpisodeRewardEstimate  (rmidx,:) = [];
            this.TotalAgentSteps(rmidx,:) = [];
            this.AverageSteps   (rmidx,:) = [];
            this.TimeStamp(rmidx,:) = [];
        end

        function flag = checkManualStopTrainOrMaxEpisodes(this)
            % Check if training was terminated by the stop train button or
            % by reaching max episodes
            flag = ismember(this.Information.StopTrainingCriteria, ["Stop Training button","MaxEpisodes"]);
        end

        function flag = checkStopTrainOptionChanged(this,agentIdx)
            % check if the StopTrainingCriteria or StopTrainingValue
            % options were modified.
            flag = this.TrainingOptions.StopTrainingCriteria ~= this.Information.StopTrainingCriteria || ...
                this.TrainingOptions.StopTrainingValue(agentIdx) ~= str2double(this.Information.StopTrainingValue);
        end

    end

    methods (Hidden, Static)
        function obj = struct2class(resultStruct)
            requiredProps = [...
                "EpisodeIndex";
                "EpisodeReward";
                "Loss";
                "AdversarialEpisodeRewardEstimate";
                "EpisodeSteps";
                "AverageEpisodeReward";
                "AverageAdversarialEpisodeRewardEstimate";
                "TotalAgentSteps";
                "AverageSteps";
                "EpisodeQ0";
                "SimulationInfo";
                "TimeStamp";
                "Information" ];

            % rebuild the info struct
            infoStruct = train.RLTrainingResult.rebuildInfoStruct([resultStruct.Information]);

            % Saved agent result case (19a-21b)
            if isstruct(resultStruct) && isscalar(resultStruct) && ...
                    isequal(fields(resultStruct),{'TrainingStats';'Information'})
                resultStruct = resultStruct.TrainingStats;
            end

            % Check if the struct has session id information. If the
            % session id is not available then a new Episode Manager
            % will be spawned.
            trainOpts = infoStruct(1).TrainingOpts;
            if isfield(resultStruct(1),'SessionId')
                sessionId = resultStruct(1).SessionId;
            else
                sessionId = [];
            end

            % Set the properties from the result struct
            for agentIdx = 1:numel(resultStruct)
                obj(agentIdx) = train.RLTrainingResult(trainOpts,sessionId); %#ok<AGROW>
                for ct = 1:numel(requiredProps)
                    name = requiredProps(ct);
                    if name == "Information"
                        % assign the reconstructed info struct
                        resultStruct(agentIdx).Information = infoStruct(agentIdx);
                    elseif name == "EpisodeQ0" && ~isfield(resultStruct(agentIdx),"EpisodeQ0")
                        % If EpisodeQ0 field is not present (e.g. old
                        % PG agent struct) then assign empty
                        resultStruct(agentIdx).EpisodeQ0 = [];
                    elseif name == "TimeStamp"
                        if ~isfield(resultStruct(agentIdx),"TimeStamp") || isempty(resultStruct(agentIdx).TimeStamp)
                            % If time stamp info is not present in
                            % resultStruct then check in infoStruct
                            if isempty(infoStruct(agentIdx).TimeStamp) || strcmp(infoStruct(agentIdx).TimeStamp,"")
                                % If time stamp is not present in
                                % infoStruct then add dummy timestamp
                                % values
                                tStamp = repmat("00:00:00",1,resultStruct(agentIdx).EpisodeIndex(end));
                                resultStruct(agentIdx).TimeStamp = tStamp;
                            else
                                % get time stamp info from infoStruct
                                resultStruct(agentIdx).TimeStamp = infoStruct(agentIdx).TimeStamp(1:resultStruct(agentIdx).EpisodeIndex(end));
                            end
                        else
                            % get time stamp info from resultStruct
                            resultStruct(agentIdx).TimeStamp = resultStruct(agentIdx).TimeStamp(1:resultStruct(agentIdx).EpisodeIndex(end));
                        end
                    end
                    obj(agentIdx).(name) = resultStruct(agentIdx).(name);
                end
            end
        end

        function info = rebuildInfoStruct(info)
            % Load missing info in case of old result struct (19a-20b)
            % verify information struct
            if isstruct(info) && isfield(info,'TraningStartTime')
                % g2722589 compensate for typo in field name
                info.TrainingStartTime = info.TraningStartTime;
                info = rmfield(info, 'TraningStartTime');
            end
            if ~isstruct(info) || ~all(ismember({'TrainingOpts','HardwareResource',...
                    'LearningRate','TrainingStartTime','ElapsedTime'},fields(info)))
                error(message('rl:general:InvalidResultsStruct'));
            end
            for idx = 1:numel(info)
                if ~isfield(info(idx),'EnvironmentName') || isempty(info(idx).EnvironmentName)
                    info(idx).EnvironmentName = "Environment";
                end
                if ~isfield(info(idx),'AgentName') || isempty(info(idx).AgentName)
                    info(idx).AgentName = "Agent_"+idx;
                end
                if ~isfield(info(idx),'BlockPath') || isempty(info(idx).BlockPath)
                    info(idx).BlockPath = [];
                end
                if ~isfield(info(idx),'StopTrainingCriteria') || isempty(info(idx).StopTrainingCriteria)
                    info(idx).StopTrainingCriteria = "--";
                end
                if ~isfield(info(idx),'StopTrainingValue') || isempty(info(idx).StopTrainingValue)
                    info(idx).StopTrainingValue = "--";
                end
                if ~isfield(info(idx),'FinalResult') || isempty(info(idx).FinalResult)
                    info(idx).FinalResult = "--";
                end
                if ~isfield(info(idx),'TimeStamp') || isempty(info(idx).TimeStamp) || all(strcmp(info(idx).TimeStamp,""))
                    info(idx).TimeStamp = "";
                end
                if isfield(info(idx),'ElapsedTime') && ~isempty(info(idx).ElapsedTime)
                    elapsedTime = duration(0,0,str2double(regexprep(info(idx).ElapsedTime,' sec','')));
                    if isnan(elapsedTime)
                        elapsedTime = duration(info(idx).ElapsedTime);
                    end
                    info(idx).ElapsedTime = string(elapsedTime);
                end
            end
        end
    end

    methods (Access = private)
        function this = reset(this)
            % Reset the training result
            maxEpisodes = this.TrainingOptions.MaxEpisodes;
            this.TimeStamp        = repmat("",maxEpisodes,1);
            this.EpisodeIndex     = zeros(maxEpisodes,1);
            %             this.ActorLoss     = zeros(maxEpisodes,1);
            %             this.CriticLoss     = zeros(maxEpisodes,1);
            this.Loss     = zeros(this.LossNum, maxEpisodes,1);
            this.EpisodeReward    = zeros(maxEpisodes,1);
            this.AdversarialEpisodeRewardEstimate    = zeros(maxEpisodes,1);
            this.EpisodeSteps     = zeros(maxEpisodes,1);
            this.AverageEpisodeReward    = zeros(maxEpisodes,1);
            this.AverageAdversarialEpisodeRewardEstimate    = zeros(maxEpisodes,1);
            this.AverageSteps     = zeros(maxEpisodes,1);
            this.TotalAgentSteps  = zeros(maxEpisodes,1);
            this.Information      = [];
            this.EpisodeQ0        = [];
            this.AppData          = struct;
        end
    end
end