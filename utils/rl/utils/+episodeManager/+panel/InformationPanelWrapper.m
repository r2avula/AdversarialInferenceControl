classdef InformationPanelWrapper < episodeManager.view.AbstractViewComponent
    % InformationPanelWrapper
    
    % Copyright 2021, The MathWorks Inc.
    
    properties (Access = private)
        % FigurePanel
        FigurePanel
        
        % Episode Manager data
        %Data
        
        % Panel widgets
        %   - StatusLabel
        %   - ProgressBar
        %   - StopButton
        %   - EpisodeNumberLabel
        %   - DurationLabel
        %   - FinalResultLabel
        %   - StatusTable
        %   - EpisodeInfoLabel
        %   - EpisodeRewardLabel
        %   - AverageRewardLabel
        %   - EpisodeQ0Label
        %   - ShowDetailsButton
        %   - ShowEpisodeQ0CheckBox
        %   - ShowLastNCheckBox
        %   - ShowLastNEditField
        Widgets
    end
    
    properties (Hidden)
        BuildFlag
    end
    
    %% Public api
    methods
        function this = InformationPanelWrapper(container,contextual)
            arguments
                container 
                contextual (1,1) logical = false
            end
            
            % Create FigurePanel
            %             panelOpts.Title = "InfoPanel";
            panelOpts.Title = "Agent Training";
            panelOpts.Tag = "infopanel";
            panelOpts.Region = "right";
            panelOpts.PermissibleRegions = {'right'};
            panelOpts.Contextual = contextual;
            panel = matlab.ui.internal.FigurePanel(panelOpts);
            add(container, panel);
            
            this@episodeManager.view.AbstractViewComponent(panel);
            
            % Set the properties
            this.FigurePanel = panel;
            this.Data = [];
            this.Widgets = struct;
            this.BuildFlag = true;
        end
        
        function widgets = getWidgets(this,name)
            arguments
                this
                name {mustBeTextScalar} = ""
            end
            if name == ""
                widgets = this.Widgets;
            else
                widgets = this.Widgets.(name);
            end
        end
                
        function build(this,options)
            
            panel = this.FigurePanel;
            
            % build labels, buttons, table
            % skip if the panel was previously built (in case of RL app)
            if this.BuildFlag
                
                % set the panel title
                panel.Title ="Agent Training";
                this.Widgets.FigurePanel = panel;

                % create main grid
                fig = panel.Figure;
                infogrid = uigridlayout(fig,[5,1]);
                infogrid.RowHeight = {'fit','fit','fit','fit','1x'};
                infogrid.Scrollable = 'on';
                this.Widgets.InfoGrid = infogrid;

                % --- Training progress section ---

                % panel
                progressPanel = uipanel(infogrid,'BorderType','none');

                % local grid
                progressGrid = uigridlayout(progressPanel,[5 2]);
                progressGrid.ColumnWidth = {'4x','5x'};
                progressGrid.RowHeight = {'fit',15,15,15,15};
                progressGrid.Padding = 0;
                this.Widgets.ProgressGrid = progressGrid;

                % progress bar grid
                grid = uigridlayout(progressGrid,[1 2]);
                grid.Layout.Column = [1 2];
                %grid.ColumnWidth = {'2x','1.25x'};
                grid.ColumnWidth = {'1x',110};
                grid.RowHeight = {25};
                grid.Padding = 0;

                % progress bar on left
                g = uigridlayout(grid,[1,1]);
                g.Padding = [0 5 0 5];
                this.Widgets.ProgressBar = uigridlayout(g,[1 2]);
                this.Widgets.ProgressBar.Padding = 0;
                this.Widgets.ProgressBar.ColumnWidth = {'0.1x','9.9x'};
                this.Widgets.ProgressBar.RowHeight = {20};
                this.Widgets.ProgressBar.BackgroundColor = 'w';
                uipanel(this.Widgets.ProgressBar,'BackgroundColor',[0, 114, 189]/256,'BorderType','none')
                uipanel(this.Widgets.ProgressBar,'BackgroundColor','w','BorderType','none')

                % stop button on the right
                this.Widgets.StopButton = uibutton(grid,'push');
                this.Widgets.StopButton.Text = "StopTraining";

                % episode number
                this.Widgets.EpisodeNumberTextLabel = uilabel(progressGrid,...
                    'Text',"EpisodeNumber"+":");
                this.Widgets.EpisodeNumberLabel = uilabel(progressGrid,'Text','');

                % start time
                %                 this.Widgets.StartTimeTextLabel = uilabel(progressGrid,...
                %                     'Text',"StartTime"+":");
                %                 this.Widgets.StartTimeLabel = uilabel(progressGrid,'Text','');

                % duration
                this.Widgets.DurationTextLabel = uilabel(progressGrid,...
                    'Text',"Duration"+":");
                this.Widgets.DurationLabel = uilabel(progressGrid,'Text','');

                % final result label
                this.Widgets.FinalResultTextLabel = uilabel(progressGrid,...
                    'Text',"FinalResult"+":",...
                    'VerticalAlignment','top');
                this.Widgets.FinalResultLabel = uilabel(progressGrid,...
                    'Text','','WordWrap','on','VerticalAlignment','top');


                % --- Training info section ---

                % panel
                trainingPanel = uipanel(infogrid,'Title',...
                    "Training Info",'FontWeight','bold');
                this.Widgets.TrainingPanel = trainingPanel;

                % local grid
                numrows = 2 + sum(options.ShowOnFigure);
                trainingGrid = uigridlayout(trainingPanel,[numrows 2]);
                trainingGrid.ColumnWidth = {'4x','5x'};
                trainingGrid.RowHeight = repmat({15},1,numrows);
                trainingGrid.RowHeight{end} = 'fit';
                %                 trainingGrid.RowHeight{1} = 'fit';

                %                 this.Widgets.StatusTable = uitable(trainingGrid);
                %                 this.Widgets.StatusTable.RowName = {};
                %                 this.Widgets.StatusTable.ColumnName = [...
                %                     "Agent";
                %                     "Status" ];
                %                 this.Widgets.StatusTable.Data = repmat("",numel(options.AgentName),2);
                %                 this.Widgets.StatusTable.RowStriping = 'off';
                %                 this.Widgets.StatusTable.ColumnEditable = false;
                %                 this.Widgets.StatusTable.ColumnSortable = false;
                %                 this.Widgets.StatusTable.SelectionType = 'row';
                %                 this.Widgets.StatusTable.Multiselect = matlab.lang.OnOffSwitchState.off;
                %                 this.Widgets.StatusTable.Layout.Column = [1 2];

                this.Widgets.EpisodeInfoLabel = uilabel(trainingGrid,...
                    'Text',"EpisodeInformation','",...
                    'FontWeight','bold');
                this.Widgets.EpisodeInfoLabel.Layout.Column = [1 2];

                for logIdx = 1:numel(options.DataName)
                    if options.ShowOnFigure(logIdx)
                        widgetText = options.DisplayName(logIdx);
                        widgetName = options.DataName(logIdx) + "Label";
                        this.Widgets.(options.DataName(logIdx)+"TextLabel") = ...
                            uilabel(trainingGrid,'Text',widgetText+":");
                        this.Widgets.(widgetName) = uilabel(trainingGrid,'Text','');
                    end
                end

                g = uigridlayout(trainingGrid,[1 2]);
                g.Padding = 0;
                g.Layout.Column = [1 2];
                g.ColumnWidth = {'1x',110};
                g.RowHeight = {'fit'};
                detailsBtn = uibutton(g,'Text',...
                    "MoreDetails"+"...");
                detailsBtn.Layout.Column = 2;
                this.Widgets.MoreDetailsButton = detailsBtn;


                % --- Plot options section ---

                % panel
                plotOptionsPanel = uipanel(infogrid,...
                    'Title',"Plot Options",...
                    'FontWeight','bold');
                this.Widgets.PlotOptionsPanel = plotOptionsPanel;

                % local grid
                plotGrid = uigridlayout(plotOptionsPanel,[2 2]);
                plotGrid.ColumnWidth = {'fit','1x'};
                plotGrid.RowHeight = {15};

                cb = uicheckbox(plotGrid,'Text',...
                    "Show Critic Initital Estimate");
                cb.Layout.Column = [1 2];
                this.Widgets.ShowEpisodeQ0CheckBox = cb;

                cb = uicheckbox(plotGrid,'Text',...
                    "Show Last N Episodes");
                this.Widgets.ShowLastNCheckBox = cb;

                ef = uieditfield(plotGrid,'numeric');
                this.Widgets.ShowLastNEditField = ef;


                % ---  Legend section ---

                % panel
                legendPanel = uipanel(infogrid);

                % local grid
                legendGrid = uigridlayout(legendPanel,[1 1]);
                legendGrid.RowHeight = 55;
                legendGrid.Padding = 0;

                % Legend
                legendAxes = uiaxes(legendGrid);
                hold(legendAxes,'on');
                showIdx = find(options.ShowOnFigure==1);
                for idx = 1:numel(showIdx)
                    line = plot(legendAxes,nan,nan,'Color',options.Color(showIdx(idx)),...
                        'LineStyle',options.LineStyle(showIdx(idx)),...
                        'Marker',options.Marker(showIdx(idx)),...
                        'MarkerSize',options.MarkerSize(showIdx(idx)));
                    if options.DataName(showIdx(idx)) == "EpisodeQ0"
                        this.Widgets.EpisodeQ0LegendLine = line;
                    end
                end
                legendAxes.XTick = [];
                legendAxes.YTick = [];
                legendAxes.Interactions = [];
                legendAxes.Color = 'w';
                legendAxes.XColor = 'w';
                legendAxes.YColor = 'w';
                legendAxes.Box = 'off';
                legendAxes.Toolbar.Visible = 'off';
                this.Widgets.Legend = legend(legendAxes,options.DisplayName(options.ShowOnFigure==1));
                this.Widgets.Legend.Location = 'north';
                this.Widgets.Legend.Color = 'w';
                this.Widgets.Legend.HitTest = 'off';
                this.Widgets.Legend.Box = 'off';
                this.Widgets.Legend.NumColumns = 2;
                this.Widgets.Legend.FontSize = 9;

                % ---  Blank space ---

                uilabel(infogrid,'Text'," ");
                
                this.BuildFlag = false;
            end
        end
        
        function reset(this)
            % initialize the training info panel
            
            options = this.Data.Options;
            
            % progress
            episodeNum = 1;
            maxEpisodes = options.TrainingOptions.MaxEpisodes;
            progressRatio = max(0.01,10*episodeNum/maxEpisodes);
            this.Widgets.ProgressBar.ColumnWidth = {...
                [num2str(progressRatio),'x'],[num2str(10-progressRatio),'x']};
            
            % update episode number
            this.Widgets.EpisodeNumberLabel.Text = ...
                episodeNum + "/" + options.TrainingOptions.MaxEpisodes;
            
            % enable stop training button
            this.Widgets.StopButton.Enable = 'on';
            this.Widgets.StopButton.Text = "StopTraining";
            
            % final result
            this.Widgets.FinalResultLabel.Text = ...
                "TrainingInProgress";
            
            % init status table data
%             agentName = options.AgentName(:);
%             agentBlock = options.AgentBlock(:);
%             agentStatus = repmat("Training",...
%                 numel(agentName),1);
%             this.Widgets.StatusTable.Data = [agentName,agentStatus];
            
            % select the first row in status table
            %this.Widgets.StatusTable.Selection = 1;
%             addStyle(this.Widgets.StatusTable, uistyle('BackgroundColor', ...
%                 [0, 114, 189]/255, 'FontColor', 'white'), 'row', 1);
%             %addStyle(this.Widgets.StatusTable, uistyle('BackgroundColor', ...
%             %    [135 206 250]/255), 'row', 1);
%             
%             % set tooltip in status table
%             if ~isempty(agentBlock)
%                 this.Widgets.StatusTable.Tooltip = ...
%                     localBlockPathTooltipText(agentName,agentBlock);
%             end
            
            % update the episode info header label
            this.Widgets.EpisodeInfoLabel.Text = ...
                " ";
            
            % episode Q0 settings
            if any(options.HasCritic)
                this.Widgets.ShowEpisodeQ0CheckBox.Value = 1;
                this.Widgets.ShowEpisodeQ0CheckBox.Enable = 'on';
                this.Widgets.EpisodeQ0LegendLine.Visible = 'on';
            else
                this.Widgets.ShowEpisodeQ0CheckBox.Value = 0;
                this.Widgets.ShowEpisodeQ0CheckBox.Enable = 'off';
                this.Widgets.EpisodeQ0LegendLine.Visible = 'off';
            end
            
            % disable show last N episodes checkbox and edit field
            this.Widgets.ShowLastNCheckBox.Value = 0;
            this.Widgets.ShowLastNEditField.Enable = 'off';
            this.Widgets.ShowLastNEditField.RoundFractionalValues = 'on';
            this.Widgets.ShowLastNEditField.Limits = [0 options.TrainingOptions.MaxEpisodes];
            this.Widgets.ShowLastNEditField.LowerLimitInclusive = 'off';
            this.Widgets.ShowLastNEditField.Value = options.TrainingOptions.MaxEpisodes;
        end
        
        function update(this)
            % update the panel UI
            
            data = this.Data.LastEpisodeInfo;
            options = this.Data.Options;
            
            % stop button
            if all(data.TrainingStatus==0)
                this.Widgets.StopButton.Text = ...
                    "TrainingStopped";
                this.Widgets.StopButton.Enable = false;
            end
            
            % progress bar
            episodeNum = data.EpisodeCount;
            maxEpisodes = options.TrainingOptions.MaxEpisodes;
            progressRatio = max(0.01,10*episodeNum/maxEpisodes);
            this.Widgets.ProgressBar.ColumnWidth = {...
                [num2str(progressRatio),'x'],[num2str(10-progressRatio),'x']};
            
            % progress labels
            this.Widgets.EpisodeNumberLabel.Text = episodeNum + "/" + maxEpisodes;
            this.Widgets.StartTimeLabel.Text = string(data.StartTime);
            this.Widgets.FinalResultLabel.Text = data.FinalResult;
            
%             % status table
%             agentName = options.AgentName;
%             agentStatus = data.TextStatus;
%             this.Widgets.StatusTable.Data = [agentName(:),agentStatus(:)];
            
            % select a row in status table
            %this.Widgets.StatusTable.Selection = data.SelectedAgentIndex;
            
            % update the episode info header label
            this.Widgets.EpisodeInfoLabel.Text = ...
                "";
            
            % episode information
            for agentIdx = 1:numel(options.AgentName)
                for dataIdx = 1:numel(options.DataName)
                    if options.ShowOnFigure(dataIdx)
                        name = options.DataName(dataIdx);
                        widgetdata = data.(name)(data.SelectedAgentIndex);
                        this.Widgets.(name+"Label").Text = string(widgetdata);
                    end
                end
            end
        end
    end
end