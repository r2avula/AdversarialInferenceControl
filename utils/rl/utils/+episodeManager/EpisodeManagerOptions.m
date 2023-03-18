classdef EpisodeManagerOptions
    % EpisodeManagerOptions creates an options set for the Episode Manager.

    % Copyright 2021, The MathWorks Inc.

    properties (SetAccess = private)
        % Name of agents to be displayed in EpisodeManager
        AgentName
        % Name of data points to be shown in EpisodeManager
        DataName
    end
    properties
        % Environment name
        EnvName
        % Agent block path
        AgentBlock
        % Training options struct
        TrainingOptions
        % Flag for specifying agent has critic
        HasCritic
        % Display text for DataName
        DisplayName
        % Flag for specifying which DataName elements are shown in plot
        ShowOnFigure
        % Color of lines
        Color
        % Line style
        LineStyle
        % Marker style
        Marker
        % Size of marker
        MarkerSize

        ShowGUI
        LossNames
    end

    methods
        function this = EpisodeManagerOptions(agentname,dataname,args)
            % EpisodeManagerOptions(agentname,dataname) creates an options
            % object for constructing the EpisodeManager.
            %
            % Optional arguments:
            % - EnvName - Environment name
            % - AgentBlock - Block path of RL Agent blocks
            % - TrainingOptions - rlTrainingOptions object
            % - HasCritic - logical flag to specify if agent has a critic
            % - DisplayName - Display text for the DataName elements
            % - ShowOnFigure - Logical flag to specify which DataName
            %       elements are shown in the Episode Manager plots
            % - Color - Color for plots
            % - LineStyle - Line style for plots
            % - Marker - Marker for plots
            % - MarkerSize - Size of markers

            arguments
                agentname (:,1) string {mustBeText}
                dataname  (:,1) string {mustBeText}
                args.EnvName = string(message('rl:general:TextEnvironment'))
                args.AgentBlock      = {}
                args.TrainingOptions = rlTrainingOptions()
                args.HasCritic       = true
                args.DisplayName     = dataname
                args.ShowOnFigure    = true
                args.Color           = "b"
                args.LineStyle       = "-"
                args.Marker          = "."
                args.MarkerSize      = 7
                args.ShowGUI      = false
                args.LossNames = [];
            end

            % set properties
            this.AgentName       = agentname;
            this.DataName        = strrep(dataname," ","");
            this.EnvName         = args.EnvName;
            this.AgentBlock      = args.AgentBlock;
            this.TrainingOptions = args.TrainingOptions;
            this.HasCritic       = args.HasCritic;
            this.DisplayName     = args.DisplayName;
            this.ShowOnFigure    = args.ShowOnFigure;
            this.Color           = args.Color;
            this.LineStyle       = args.LineStyle;
            this.Marker          = args.Marker;
            this.MarkerSize      = args.MarkerSize;
            this.ShowGUI      = args.ShowGUI;
            this.LossNames = args.LossNames;

            if all(this.HasCritic==false)
                this.ShowOnFigure(this.DataName=="EpisodeQ0") = false;
            end
        end

        function value = getNumAgents(this)
            value = numel(this.AgentName);
        end

        function value = getNumData(this)
            value = numel(this.DataName);
        end

        function this = set.EnvName(this,value)
            arguments
                this
                value (1,1) string {mustBeNonzeroLengthText}
            end
            this.EnvName = value;
        end

        function this = set.AgentBlock(this,value)
            arguments
                this
                value (:,1) string {mustBeText}
            end
            if isempty(value)
                this.AgentBlock = {};
            elseif numel(value) == getNumAgents(this)
                this.AgentBlock = value;
            else
                error(getString(message('rl:general:errSetOpts','AgentBlock','AgentName')));
            end
        end

        function this = set.TrainingOptions(this,value)
            arguments
                this
                value (1,1)
            end
            this.TrainingOptions = value;
        end

        function this = set.HasCritic(this,value)
            arguments
                this
                value (:,1) logical {mustBeVector}
            end
            if isscalar(value)
                this.HasCritic = repmat(value,getNumAgents(this),1);
            elseif numel(value) == getNumAgents(this)
                this.HasCritic = value;
            else
                error(getString(message('rl:general:errSetOpts','HasCritic','AgentName')));
            end
        end

        function this = set.DisplayName(this,value)
            arguments
                this
                value (:,1) string {mustBeText}
            end
            if numel(value) == getNumData(this)
                this.DisplayName = value;
            else
                error(getString(message('rl:general:errSetOpts','Displayname','DataName')));
            end
        end

        function this = set.ShowOnFigure(this,value)
            arguments
                this
                value (:,1) logical {mustBeVector}
            end
            if isscalar(value)
                this.ShowOnFigure = repmat(value,getNumData(this),1);
            elseif numel(value) == getNumData(this)
                this.ShowOnFigure = value;
            else
                error(getString(message('rl:general:errSetOpts','ShowOnFigure','DataName')));
            end
        end

        function this = set.Color(this,value)
            arguments
                this
                value (:,1) string {mustBeText}
            end
            if isscalar(value)
                this.Color = repmat(value,getNumData(this),1);
            elseif numel(value) == getNumData(this)
                this.Color = value;
            else
                error(getString(message('rl:general:errSetOpts','Color','DataName')));
            end
        end

        function this = set.LineStyle(this,value)
            arguments
                this
                value (:,1) string {mustBeText}
            end
            if isscalar(value)
                this.LineStyle = repmat(value,getNumData(this),1);
            elseif numel(value) == getNumData(this)
                this.LineStyle = value;
            else
                error(getString(message('rl:general:errSetOpts','LineStyle','DataName')));
            end
        end

        function this = set.Marker(this,value)
            arguments
                this
                value (:,1) string {mustBeText}
            end
            if isscalar(value)
                this.Marker = repmat(value,getNumData(this),1);
            elseif numel(value) == getNumData(this)
                this.Marker = value;
            else
                error(getString(message('rl:general:errSetOpts','Marker','DataName')));
            end
        end

        function this = set.MarkerSize(this,value)
            arguments
                this
                value (:,1) double {mustBeVector}
            end
            if isscalar(value)
                this.MarkerSize = repmat(value,getNumData(this),1);
            elseif numel(value) == getNumData(this)
                this.MarkerSize = value;
            else
                error(getString(message('rl:general:errSetOpts','MarkerSize','DataName')));
            end
        end
    end

    methods (Static)
        function obj = createDefault(args)
            % Create default Episode Manager options.
            arguments
                args.AgentName {mustBeText} = "Agent"
                args.EnvironmentName {mustBeTextScalar} = "Environment"
                args.AgentBlock = {}
                args.HasCritic logical = true
                args.TrainingOptions = rlTrainingOptions()
                args.ShowGUI = false
                args.LossNames = ["Actor Loss", "Critic Loss"]
            end
            dataName = ["EpisodeReward";
                "AdversarialEpisodeRewardEstimate";
                "EpisodeSteps";
                "GlobalStepCount";
                "AverageEpisodeReward";
                "AverageAdversarialEpisodeRewardEstimate";
                "AverageSteps";
                "EpisodeQ0"];
            displayName = [...
                "Episode Bayes risk";
                "Episode Bayes reward";
                string(message('rl:general:TextEpisodeSteps'));
                string(message('rl:general:TextTotalNumSteps'));
                "Moving avg. (100) Bayes risk";
                "Moving avg. (100) Bayes reward";
                string(message('rl:general:TextAverageSteps'));
                "Critic initial estimate"];
            obj = episodeManager.EpisodeManagerOptions( ...
                args.AgentName, dataName, ...
                'EnvName', args.EnvironmentName, ...
                'DisplayName', displayName, ...
                'TrainingOptions', args.TrainingOptions, ...
                'HasCritic',args.HasCritic, ...
                'ShowOnFigure', [1,1,0,0,1,1,0,1], ...
                'Color',["#B0E2FF","#EDB120","k","k","#0072BD","#FF0000","k","#000000"],...
                'ShowGUI', args.ShowGUI,...
                'LossNames',args.LossNames);
            obj.AgentBlock = args.AgentBlock;
        end
    end
end