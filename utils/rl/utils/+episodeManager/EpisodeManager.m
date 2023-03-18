classdef EpisodeManager < handle
    % EpisodeManager

    % Copyright 2021, The MathWorks Inc.

    %% Properties
    properties 
        % Episode Manager view
        View

        % Episode Manager data
        Data

        % Episode Manager options
        Options

        % Flag for standalone mode
        IsStandAlone

        % Flag for visibility
        IsClosed

        ShowGUI

        Duration

        %         LossNum
    end

    properties(Access = private, Transient)
        % Timer object
        Timer
    end

    properties (SetObservable, AbortSet)
        % Flag for stop training
        RequestToStopTraining (1,1) logical = false
    end

    properties (SetAccess = private)
        Id
    end

    %% Public API
    methods
        function this = EpisodeManager(options,view)
            % E = CustomEpisodeManager(OPTIONS)
            % creates an GUI for tracking RL training progress. OPTIONS is
            % an instance of CustomEpisodeManagerOptions.
            %
            % E = CustomEpisodeManager(OPTIONS,VIEW)
            % creates an episode manager gui from components specified in
            % VIEW. VIEW is an instance of episodeManager.view.StandaloneView
            % or episodeManager.view.EmbeddedView.

            arguments
                options (1,1) episodeManager.EpisodeManagerOptions
                view = []
                %                 lossNum = 0
            end

            this.ShowGUI = options.ShowGUI;
            %             this.LossNum = lossNum;

            if this.ShowGUI
                % If view was not specified then create a standalone view
                if isempty(view)
                    ndocs = numel(options.AgentName);
                    view = episodeManager.view.StandaloneView('NumDocuments',ndocs,'LossNames',options.LossNames);
                    this.IsStandAlone = 1;

                elseif isa(view,'episodeManager.view.StandaloneView')
                    this.IsStandAlone = 1;

                elseif isa(view,'episodeManager.view.EmbeddedView')
                    this.IsStandAlone = 0;

                else
                    error(message('rl:general:InvalidView'));
                end
            end

            % Set properties
            this.View = view;
            this.Data = episodeManager.data.EpisodeManagerData(options);
            this.Options = options;
            this.IsClosed = false;

            % Set a unique id for the Episode Manager. This must be
            % different from the app container tag (which is also unique) 
            % in order to avoid inconsistencies in the app.
            % e.g. There is only one app container for the app but an
            % episode manager for each training result.
            this.Id = string(matlab.lang.internal.uuid);
            
            %             show(this)

            if this.ShowGUI
                % Build episode manager UI
                build(this)
            end

            % Reset episode manager UI
            reset(this)

            if this.ShowGUI
                % Connect callbacks
                connect(this)

                % Show the Episode Manager
                show(this)

                cbShowEpisodeQ0(this, struct('Value',0))
            end
        end

        function delete(this)
            % delete the episode manager view
            if ~isempty(this.Timer) && isvalid(this.Timer)
                stop(this.Timer)
                delete(this.Timer)
            end

            if this.IsStandAlone
                delete(this.View)
            end
        end

        function show(this)
            % Show the Episode Manager figure

            container = getContainer(this.View);
            container.Visible = true;
            this.IsClosed = false;
            bringToFront(container)
            container.RightWidth = 480;
            container.WindowMaximized = 1;
        end

        function data = getData(this)
            % Get the episode manager data

            data = this.Data;
        end

        function setStartTime(this,timestr)
            % Set the start time label in the episode manager

            arguments
                this
                timestr {mustBeTextScalar}
            end
            %             if this.ShowGUI
            %                 panel = getPanel(this.View);
            %                 startTimeLabel = getWidgets(panel,'StartTimeLabel');
            %                 startTimeLabel.Text = timestr;
            %             end
            update(this.Data,'StartTime',timestr);
        end

        function setActionMessage(this,msg)
            % Set the status bar message in the episode manager

            arguments
                this
                msg {mustBeTextScalar}
            end

            statusbar = getStatusBar(this.View);
            setStatusEpisodeManager(statusbar,msg)
        end

        function stepEpisode(this,episodeInfo)
            % Step the episode manager with information from episodeInfo
            %
            % episodeInfo is a struct with the following fields:
            %  - AverageSteps   : Running average of number of steps per episode
            %  - AverageReward  : Running average of reward per episode
            %  - EpisodeReward  : Reward for current episode
            %  - GlobalStepCount: Total times the agent was invoked
            %  - EpisodeCount   : Total number of episodes the agent has trained for
            %  - TrainingStatus : Training status of agents (0 or 1)
            %
            % Each field has the size 1 x numAgents

            arguments
                this
                episodeInfo struct
            end

            % update the data
            datafields = fields(episodeInfo);
            for idx = 1:numel(datafields)
                update(this.Data,datafields{idx},episodeInfo.(datafields{idx}))
            end
            %             update(this.Data, 'ActorLoss' ,episodeInfo.('ActorLoss' ))
            %             update(this.Data, 'CriticLoss' ,episodeInfo.('CriticLoss' ))
            update(this.Data, 'Loss' ,episodeInfo.('Loss' ))
            % update UI
            if ~this.IsClosed && this.ShowGUI
                update(this)
            end
        end

        function stopTraining(this,finalResultText)
            % update episode manager to reflect training stopped state
            % after training is terminated or finished

            arguments
                this
                finalResultText string {mustBeNonzeroLengthText}
            end

            % stop the timer
            stop(this.Timer);
            delete(this.Timer);

            if this.ShowGUI
                panel = getPanel(this.View);
                panelWidgets = getWidgets(panel);

                % update data
                update(this.Data,'Duration',panelWidgets.DurationLabel.Text);

                % Update UI
                update(panel)
                panelWidgets.ProgressGrid.RowHeight{end} = 'fit';

                setActionMessage(this,'')
            end
            update(this.Data,'FinalResult',finalResultText);
        end

        function stopTrainingAgent(this,agentIdx,stopReason,stopValue)
            % update episode manager to reflect training stopped state for
            % the agent specified by agentIdx.

            arguments
                this
                agentIdx   (1,1) double {mustBePositive}
                stopReason string {mustBeNonzeroLengthText}
                stopValue  string {mustBeNonzeroLengthText}
            end

            panel = getPanel(this.View);
            dialog = getDialog(this.View);

            % Update data
            status = localTrainingStatusText(stopReason);
            update(this.Data,'TrainingStatus',0,agentIdx);
            update(this.Data,'TextStatus',status,agentIdx);
            update(this.Data,'StopReason',stopReason,agentIdx);
            update(this.Data,'StopValue',stopValue,agentIdx);

            % Update UI (status table and more details dialog)
            update(panel)
            update(dialog)
        end

        function resumeTraining(this,checkpoint)

            % update the training options in case they were modified
            this.Options.TrainingOptions = checkpoint(1).TrainingOptions;

            % update episode manager data
            for idx = numel(this.Options.AgentName):-1:1
                % Resume training if:
                % 1. training was terminated by stop training button or
                %    max episodes.
                % 2. StopTrainingCriteria or StopTrainingValue was modified
                if checkManualStopTrainOrMaxEpisodes(checkpoint(idx)) || checkStopTrainOptionChanged(checkpoint(idx),idx)
                    trainingStatus(idx) = "Training";
                    stopReason(idx) = "";
                    stopValue(idx) = "";
                else
                    trainingStatus(idx) = "Training finished";
                    stopReason(idx) = checkpoint(idx).Information.StopTrainingCriteria;
                    stopValue(idx) = checkpoint(idx).Information.StopTrainingValue;
                end
            end
            update(this.Data,'TextStatus',trainingStatus)
            update(this.Data,'FinalResult','Training in progress');
            update(this.Data,'StopReason',stopReason);
            update(this.Data,'StopValue',stopValue);
            setOptions(this.Data, this.Options);

            if this.ShowGUI
                % Update UI
                panel = getPanel(this.View);
                update(panel)
                document = getDocument(this.View);
                rebuild(document)

                % Update stop button
                panelWidgets = getWidgets(panel);
                panelWidgets.StopButton.Text = getString(message('rl:general:TextStopTraining'));
                panelWidgets.StopButton.Enable = true;
                
                % Update the status bar
                setActionMessage(this,'')
                setActionMessage(this,'Resumed training from checkpoint')
            end

            this.RequestToStopTraining = false;

            % Reset the timer
            elapsedTime = duration(checkpoint(1).Information(1).ElapsedTime);
            startTime = datetime('now') - elapsedTime;
            resetTimer(this,startTime)

            if this.ShowGUI
                % bring the episode manager to front
                show(this)
            end
        end

        function update(this)
            % update the episode manager UI from the stored data

            document = getDocument(this.View);
            dialog = getDialog(this.View);
            panel = getPanel(this.View);

            % update all view components
            update(document)
            update(dialog)
            update(panel)
        end

        function updateWithTrainingResult(this,result)
            % Update the episode manager with information from a training
            % result object.

            arguments
                this
                result train.RLTrainingResult
            end

            % stop the timer
            stop(this.Timer);
            delete(this.Timer);

            % update episode manager data
            info = [result.Information];
            updateDataFromResult(this,result,info);

            if this.ShowGUI
                document = getDocument(this.View);
                panel    = getPanel(this.View);
                dialog   = getDialog(this.View);

                docWidgets   = getWidgets(document);
                panelWidgets = getWidgets(panel);

                % update timestamps in data tips
                updateTimeStamps(document, result(1).TimeStamp(:)')

                % update plot for each agent
                for agentIdx = 1:numel(this.Options.AgentName)
                    numEpisodes = result(agentIdx).EpisodeIndex(end);
                    for dataIdx = 1:numel(this.Options.DataName)
                        dataname = this.Options.DataName(dataIdx);
                        % GlobalStepCount is saved as TotalAgentSteps in the
                        % training statistics
                        if dataname == "GlobalStepCount"
                            episodeData = result(agentIdx).("TotalAgentSteps");
                        else
                            episodeData = abs(result(agentIdx).(dataname));
                        end
                        % Update plot
                        if this.Options.ShowOnFigure(dataIdx)
                            line = docWidgets.(dataname+"Line")(agentIdx);
                            line.YData(1:numEpisodes) = episodeData';
                        end
                    end

                    LossNum = size(result(agentIdx).Loss,1);
                    for loss_idx = 1:LossNum
                        dataname = strcat('Loss_',num2str(loss_idx),'_');
                        episodeData = abs(result(agentIdx).('Loss')(loss_idx,:));
                        line = docWidgets.(dataname+"Line")(agentIdx);
                        line.YData(1:numEpisodes) = episodeData';
                    end
                end

                % update the dialog and panel
                update(dialog)
                update(panel)

                panelWidgets.DurationLabel.Text = info(1).ElapsedTime;
            end
        end

        function connect(this)
            % Specify callback behavior for widgets

            panel = getPanel(this.View);
            panelWidgets = getWidgets(panel);

            panelWidgets.StopButton.ButtonPushedFcn            = @(~,ed) cbStopButton(this,ed);
            panelWidgets.StatusTable.CellSelectionCallback     = @(~,ed) cbStatusTable(this,ed);
            panelWidgets.MoreDetailsButton.ButtonPushedFcn     = @(~,~) cbMoreDetails(this);
            panelWidgets.ShowEpisodeQ0CheckBox.ValueChangedFcn = @(~,ed) cbShowEpisodeQ0(this,ed);
            panelWidgets.ShowLastNCheckBox.ValueChangedFcn     = @(~,ed) cbShowLastNCheckbox(this,ed);
            panelWidgets.ShowLastNEditField.ValueChangedFcn    = @(~,ed) cbShowLastNEditField(this,ed);

        end
        
        function cbShowEpisodeQ0(this,ed)
            document = getDocument(this.View);
            panel = getPanel(this.View);
            ShowEpisodeQ0CheckBox = getWidgets(panel,'ShowEpisodeQ0CheckBox');
            episodeQ0Line = getWidgets(document,'EpisodeQ0Line');
            episodeQ0LegendLine = getWidgets(panel,'EpisodeQ0LegendLine');
            % set the episode Q0 line
            for idx = 1:numel(episodeQ0Line)
                if ~isempty(episodeQ0Line(idx)) && ed.Value == 1
                    episodeQ0Line(idx).Visible = 'on';
                    update(this.Data,'ShowEpisodeQ0',1);
                    ShowEpisodeQ0CheckBox.Value = 1;
                else
                    episodeQ0Line(idx).Visible = 'off';
                    update(this.Data,'ShowEpisodeQ0',0);
                    ShowEpisodeQ0CheckBox.Value = 0;
                end
            end
            % set the legend
            if ed.Value == 1
                episodeQ0LegendLine.Visible = 'on';
            else
                episodeQ0LegendLine.Visible = 'off';
            end
        end
    end

    %% Private methods
    methods (Access = private)

        function updateDataFromResult(this,stats,info)
            % Update the Data property from a training result
            % structure

            na = numel(this.Options.AgentName);

            update(this.Data, 'EpisodeCount',   stats(1).EpisodeIndex(end));
            update(this.Data, 'TrainingStatus', zeros(1,na));
            update(this.Data, 'StartTime',      info(1).TrainingStartTime);
            update(this.Data, 'Duration',       info(1).ElapsedTime);

            for agentIdx = 1:na
                % update metrics
                for dataIdx = 1:numel(this.Options.DataName)
                    dataname = this.Options.DataName(dataIdx);
                    if dataname == "EpisodeQ0" && ...
                            (~isprop(stats(agentIdx),'EpisodeQ0') || ...
                            isempty(stats(agentIdx).EpisodeQ0))
                        episodeData = "--";
                    elseif dataname == "GlobalStepCount"
                        episodeData = stats(agentIdx).("TotalAgentSteps");
                    else
                        episodeData = abs(stats(agentIdx).(dataname));
                    end
                    update(this.Data,dataname,episodeData(end),agentIdx);
                end
                %                 update(this.Data,'ActorLoss',stats.ActorLoss(end),agentIdx);
                %                 update(this.Data,'CriticLoss',stats.CriticLoss(end),agentIdx);
                update(this.Data,'Loss',stats.Loss(:,end),agentIdx);

                % update stop training reason, value and training status
                reason = info(agentIdx).StopTrainingCriteria;
                value = info(agentIdx).StopTrainingValue;
                update(this.Data,'StopReason',reason,agentIdx);
                update(this.Data,'TextStatus',localTrainingStatusText(reason),agentIdx);
                update(this.Data,'StopValue',string(value),agentIdx);
            end

            % update final result text
            update(this.Data,'FinalResult',info(agentIdx).FinalResult);
        end

        function build(this)
            % build the UI for episode manager
            container = getContainer(this.View);

            % Set a figure title
            if this.IsStandAlone
                container.Title = getString(message('rl:general:FigureName'));
            end

            document = getDocument(this.View);
            dialog = getDialog(this.View);
            panel = getPanel(this.View);

            % build
            if this.IsStandAlone
                build(document,this.Options.AgentName)
            else
                build(document)
            end
            build(panel,this.Options)
            build(dialog)
            %             update(this.Data,'ShowEpisodeQ0',0);

            % Set the data in the view components
            setData(document,this.Data)
            setData(dialog,this.Data)
            setData(panel,this.Data)

            % Store episode manager
            master = this.shelve(this);

            % Specify app close behavior
            if this.IsStandAlone
                container.CanCloseFcn = @(es) cbCanClose(this,master);
                addlistener(container,'StateChanged',@(~,~) cbEMStateChanged(this));
            end
        end

        function reset(this)

            % reset the timer
            resetTimer(this)


            if this.ShowGUI
                document = getDocument(this.View);
                dialog = getDialog(this.View);
                panel = getPanel(this.View);
            end

            % reset the data
            reset(this.Data)

            if this.ShowGUI
                % reset UI
                reset(document)
                reset(dialog)
                reset(panel)
                setActionMessage(this,'')
            end
            %             episodeQ0Line = getWidgets(document,'EpisodeQ0Line');
            %             ShowEpisodeQ0CheckBox = getWidgets(panel,'ShowEpisodeQ0CheckBox');
            %             episodeQ0LegendLine = getWidgets(panel,'EpisodeQ0LegendLine');
            %             % set the episode Q0 line
            %             for idx = 1:numel(episodeQ0Line)
            %                 episodeQ0Line(idx).Visible = 'off';
            %             end
            %             update(this.Data,'ShowEpisodeQ0',0);
            %             ShowEpisodeQ0CheckBox.Value = 0;
            %             episodeQ0LegendLine.Visible = 'off';
        end

        function resetTimer(this,startTime)
            arguments
                this
                startTime = []
            end
            % reset the timer
            this.Timer = timer('Tag','RLEpisodeManagerTimer');
            this.Timer.Period = 1;
            this.Timer.ExecutionMode = 'fixedRate';

            if isempty(startTime)
                startTime = datetime('now');
            end
            if this.ShowGUI
                panel = getPanel(this.View);
                durationLabel = getWidgets(panel,'DurationLabel');
                this.Timer.TimerFcn = @(src,ed) cbUpdateDuration(this,startTime,durationLabel);
            else
                this.Timer.TimerFcn = @(src,ed) cbUpdateDuration(this,startTime,this.Duration);
            end
            start(this.Timer);
        end

        % ==================== CALLBACKS =======================

        function flag = cbCanClose(this,master)
            this.discard(master,this.Id);
            this.IsClosed = true;
            flag = true;
        end

        function cbEMStateChanged(this)
            % delete Episode Manager when container is closed
            container = getContainer(this.View);
            if container.State == matlab.ui.container.internal.appcontainer.AppState.TERMINATED
                delete(this);
            end
        end

        function cbUpdateDuration(this,startTime,durationLabel)
            % update the duration label
            if ~this.IsClosed
                currentTime = datetime('now');
                elapsedTime = currentTime - startTime;
                durationLabel.Text = string(elapsedTime);
            end
        end

        function cbStopButton(this,ed)
            stopButton = ed.Source;
            stopButton.Text = getString(message('rl:general:TextTrainingStopped'));
            stopButton.Tooltip = "";
            stopButton.Enable = false;
            this.RequestToStopTraining = true;
        end

        function cbStatusTable(this,ed,episodeInfoLabel)
            choice = ed.Indices(1);

            % REVISIT Turn on selection color override once selection focus
            % can be disabled
            s1 = uistyle('BackgroundColor',	[0, 114, 189]/255);
            s1.FontColor = 'white';
            s2 = uistyle('BackgroundColor',[1 1 1]);
            s2.FontColor = 'black';
            addStyle(ed.Source,s2,'row',1:numel(this.Options.AgentName))
            addStyle(ed.Source,s1,'row',choice);

            episodeInfoLabel.Text = string(message('rl:general:TitleEpisodeInformation',...
                char(this.Options.AgentName(choice))))+":";
            update(this.Data,'SelectedAgentIndex',choice);

            panel = getPanel(this.View);
            update(panel);
        end

        function cbMoreDetails(this)
            container = getContainer(this.View);
            dialog = getDialog(this.View);
            show(dialog,container);
        end        

        function cbShowLastNCheckbox(this,ed)
            panel = getPanel(this.View);
            showLastNEditField = getWidgets(panel,'ShowLastNEditField');
            N = showLastNEditField.Value;
            data = this.Data.LastEpisodeInfo;
            episodeCt = data.EpisodeCount;
            if ed.Value == 1
                showLastNEditField.Enable = 'on';
                update(this.Data,'ShowLastN',1);
                update(this.Data,'ShowLastNValue',N);
                xmin = max(0,episodeCt-N+1);
            else
                showLastNEditField.Enable = 'off';
                update(this.Data,'ShowLastN',0);
                update(this.Data,'ShowLastNValue',episodeCt);
                xmin = 0;
            end
            % update the x-axis limit
            xlim = [xmin,episodeCt];
            document = getDocument(this.View);
            updatePlotXLim(document,xlim);
        end

        function cbShowLastNEditField(this,ed)
            N = ed.Value;
            if N>0
                data = this.Data.LastEpisodeInfo;
                episodeCt = data.EpisodeCount;
                update(this.Data,'ShowLastNValue',N);
                % update the x-axis limit
                xmin = max(0,episodeCt-N+1);
                xlim = [xmin,episodeCt];
                document = getDocument(this.View);
                updatePlotXLim(document,xlim);
            end
        end
    end

    %% Static methods
    methods (Hidden, Static)
        function episodeMgr = getHandleById(id)
            % Get the handle to an existing, open episode manager gui by
            % its session id.
            try
                master = episodeManager.EpisodeManager.shelve();
                episodeMgr = master(id);
            catch
                episodeMgr = [];
            end
        end
    end

    methods (Access = private, Static)
        function obj = shelve(this)
            % Store the episode manager object in a persistent variable
            arguments
                this = []
            end
            persistent master
            % If the episode manager is provided as an argument then
            % append to master, else just return master
            if ~isempty(this) && isvalid(this)
                id = this.Id;
                if isempty(master)
                    master = containers.Map(id,this);
                else
                    master(id) = this;
                end
            end
            obj = master;
        end
        function discard(master,id)
            % Discard the episode manager with id from the master
            if ~isempty(master)
                remove(master,char(id));
            end
        end
    end

    %% Hidden QE methods
    methods (Hidden)
        function prop = qeGet(this,name)
            prop = this.(name);
        end

        function widgets = qeGetWidgets(this)
            widgets.AppContainer = getContainer(this.View);
            widgets.Statusbar = getStatusBar(this.View);
            widgets.DocumentWidgets = getWidgets(getDocument(this.View));
            widgets.PanelWidgets = getWidgets(getPanel(this.View));
            widgets.DialogWidgets = getWidgets(getDialog(this.View));
        end
    end
end

%% Local functions
function status = localTrainingStatusText(reason)
if reason == "user clicked stop training button"
    status = "TrainingStopped";
else
    status = "TrainingFinished";
end
end
