classdef EpisodeManagerData < handle
    % EpisodeManagerData
    
    % Copyright 2021, The MathWorks Inc.
    
    properties (SetAccess = private)
        % Last episode data with fields:
        % StartTime
        % Duration
        % FinalResult
        % TextStatus
        % StopReason
        % StopValue
        % TrainingStatus
        % EpisodeCount
        % ShowEpisodeQ0
        % ShowLastN
        % ShowLastNValue
        % SelectedAgentIndex
        LastEpisodeInfo

        % Episode Manager options
        Options
    end
    
    methods
        function this = EpisodeManagerData(options)
            % DATA = EpisodeManagerData(OPTIONS) creates a default
            % EpisodeManagerData object from OPTIONS.
            %
            % OPTIONS is a EpisodeManagerOptions object.
            
            this.Options = options;
            reset(this)
        end
        
        function reset(this)
            % Reset the Episode manager data to default initial values
            %
            % reset(DATA) resets the fields in LastEpisodeInfo property to 
            % default values.
            
            na = numel(this.Options.AgentName);
            this.LastEpisodeInfo = struct( ...
                'StartTime',[],...
                'Duration',[],...
                'FinalResult',"Training in progress",...
                'TextStatus',repmat("Training",1,na),...
                'StopReason',repmat("",1,na),...
                'StopValue',repmat("",1,na),...
                'TrainingStatus',ones(1,na),...
                'EpisodeCount',1,...
                'ShowEpisodeQ0',1,...
                'ShowLastN',0,...
                'ShowLastNValue',this.Options.TrainingOptions.MaxEpisodes,...
                'SelectedAgentIndex',1 );
            for idx = 1:numel(this.Options.DataName)
                this.LastEpisodeInfo.(this.Options.DataName(idx)) = repmat("",1,na); %zeros(1,na);
            end
            %             this.LastEpisodeInfo.ActorLoss = nan;
            %             this.LastEpisodeInfo.CriticLoss = nan;
            this.LastEpisodeInfo.Loss = [];
        end
        
        function update(this,name,value,idx)
            % Update the EPisode Manager data
            %
            % update(DATA,NAME,VALUE) updates the LastEpisodeInfo property
            % with information from NAME and VALUE. NAME must be a field
            % name in LastEPisodeInfo and VALUE is the value for that
            % field.
            %
            % update(DATA,NAME,VALUE,IDX) updates the LastEpisodeInfo property
            % with information from NAME and VALUE. IDX specifies the index
            % to update in the field name specified by NAME.
            
            arguments
                this
                name {mustBeTextScalar}
                value
                idx double = []
            end
            this.LastEpisodeInfo.(name) = value;
        end
    end

    methods (Hidden)
        function setOptions(this,options)
            arguments
                this
                options (1,1) episodeManager.EpisodeManagerOptions
            end
            this.Options = options;
        end
    end

end