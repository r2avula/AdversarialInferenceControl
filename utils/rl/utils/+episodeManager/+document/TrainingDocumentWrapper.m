classdef TrainingDocumentWrapper < episodeManager.view.AbstractViewComponent
    % TrainingDocumentWrapper
    
    % Copyright 2021, The MathWorks Inc.
    
    properties %(Access = private)
        % Figure documents for plots
        Document
        
        % Graphics components:
        % - Ax
        % - Timestamps
        % - EpisodeRewardLine
        % - AverageRewardLine
        % - EpisodeQ0Line
        Widgets
        LossNames
    end
    
    %% Public api
    methods
        function this = TrainingDocumentWrapper(container,args)
            arguments
                container
                args.NumDocuments (1,1) double = 1
                args.Document = []
                args.LossNames = []
            end
            
            % if document is not passed as input then create new
            if isempty(args.Document)
                % create document group
%                 group = ui.internal.FigureDocumentGroup();
                group = matlab.ui.internal.FigureDocumentGroup();
                group.Title ="Training Results";
                group.Tag = "documentgroup";
                container.add(group);

                % create figure documents
                for idx = 1:args.NumDocuments
                    documentOptions.Title = "document" + idx;
                    documentOptions.Tag = "document" + idx;
                    documentOptions.DocumentGroupTag = group.Tag;
%                     doc = ui.internal.FigureDocument(documentOptions);
                    doc = matlab.ui.internal.FigureDocument(documentOptions);
                    doc.Closable = 0;
                    container.add(doc);
                    documents(idx) = doc; %#ok<AGROW>
                end
                Document = documents;
            else
                Document = args.Document;
            end
            
            this@episodeManager.view.AbstractViewComponent(Document);
            
            % Set document layout on app
            tag = Document(1).DocumentGroupTag;
            container.DocumentLayout = localDocumentLayout(args.NumDocuments,tag);
            
            % set properties
            this.Document = Document;
            this.Widgets = struct;
            this.LossNames = args.LossNames;
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
        
        function build(this,doctitle)
            % Set document group and document properties
            
            arguments
                this
                doctitle = []
            end
            
            % Build documents
            na = numel(this.Document);
            if na >1
                error('not implemented')
            end

            if ~isempty(doctitle)
                this.Document(1).Title = doctitle(1);
            end
            % create axes

            loss_num = length(this.LossNames);
            g = uigridlayout(this.Document(1).Figure,[2 loss_num]);

            g.Padding = 20;
            this.Widgets.Ax(1) = uiaxes(g);
            this.Widgets.Ax(1).Layout.Row = 1;
            this.Widgets.Ax(1).Layout.Column  = [1,loss_num];
            for loss_idx = 1:loss_num
                this.Widgets.Ax(loss_idx+1) = uiaxes(g);
                this.Widgets.Ax(loss_idx+1).Layout.Row = 2;
                this.Widgets.Ax(loss_idx+1).Layout.Column  = loss_idx;
            end
        end
        
        function reset(this)
            % Initialize the training plots
            %   - Set data points to NaN for all episodes
            %   - Set plot title, legend, labels, line style
            %   - Create custom data tip for time stamps
            
            options = this.Data.Options;
            maxEpisodes = options.TrainingOptions.MaxEpisodes;
            
            xdata = 1:maxEpisodes;
            ydata = nan(1,maxEpisodes);
            
            this.Widgets.TimeStamps = string(zeros(size(xdata)));
            plotIdx = 1;
            ax = this.Widgets.Ax(plotIdx);
            cla(ax);

            for dataIdx = 1:numel(options.DataName)
                if options.ShowOnFigure(dataIdx)
                    % plot line
                    linename = options.DataName(dataIdx);
                    line = plot(ax,xdata,ydata,'Tag',linename);

                    % line properties
                    line.Color = options.Color(dataIdx);
                    line.LineStyle = options.LineStyle(dataIdx);
                    line.Marker = options.Marker(dataIdx);
                    line.MarkerSize = options.MarkerSize(dataIdx);
                    line.DataTipTemplate.DataTipRows(1).Label = "EpisodeNumber";
                    line.DataTipTemplate.DataTipRows(2).Label = options.DisplayName(options.DataName==linename);

                    % time stamps
                    tStampRow = dataTipTextRow("PlotElapsedTime",...
                        @(x) this.Widgets.TimeStamps(x));
                    line.DataTipTemplate.DataTipRows(3) = tStampRow;
                    line.DataTipTemplate.FontSize = 8;

                    hold(ax,'on');

                    this.Widgets.(linename+"Line")(plotIdx) = line;
                end
            end

            ax.XLabel.String = "Episode Number";
            ax.YLabel.String = "Episode Reward";
            axtoolbar(ax,{'export','pan','zoomin','zoomout','restoreview'});

            for loss_idx = 1:length(this.LossNames)
                plotIdx = loss_idx+1;
                ax = this.Widgets.Ax(plotIdx);
                cla(ax);
                % plot line
                linename = strcat('Loss_',num2str(loss_idx),'_');
                line = plot(ax,xdata,ydata,'Tag',linename);
                this.Widgets.(linename+"Line") = line;

                ax.XLabel.String = "Episode Number";
                ax.YLabel.String = this.LossNames(loss_idx);
                axtoolbar(ax,{'export','pan','zoomin','zoomout','restoreview'});
            end
        end

        function rebuild(this)
            % rebuild the plots when the MaxEpisodes property is changed.
            % The plot XData and YData cannot be dynamically adjusted 
            % without throwing warnings. As workaround, clear the plot and
            % rebuild it. The alternative is to use animatedline but it
            % does not have data tip support.
            % rebuild is used before resuming training.

            options = this.Data.Options;
            oldMaxEpisodes = numel(this.Widgets.TimeStamps);
            newMaxEpisodes = options.TrainingOptions.MaxEpisodes;
            
            if oldMaxEpisodes < newMaxEpisodes
                % extract the old data from the plots
                for plotIdx = 1:numel(options.AgentName)
                    for dataIdx = 1:numel(options.DataName)
                        if options.ShowOnFigure(dataIdx)
                            linename = options.DataName(dataIdx);
                            oldData.(linename+"Line")(plotIdx).XData = this.Widgets.(linename+"Line")(plotIdx).XData;
                            oldData.(linename+"Line")(plotIdx).YData = this.Widgets.(linename+"Line")(plotIdx).YData;
                        end
                    end
                end

                for loss_idx = 1:length(this.LossNames)
                    linename = strcat('Loss_',num2str(loss_idx),'_');
                    oldData.(linename+"Line").XData = this.Widgets.(linename+"Line").XData;
                    oldData.(linename+"Line").YData = this.Widgets.(linename+"Line").YData;
                end

                oldData.TimeStamps = this.Widgets.TimeStamps;

                % reset the plots
                reset(this)

                % rebuild the data
                this.Widgets.TimeStamps(1:oldMaxEpisodes) = oldData.TimeStamps;
                for plotIdx = 1:numel(options.AgentName)
                    for dataIdx = 1:numel(options.DataName)
                        if options.ShowOnFigure(dataIdx)
                            linename = options.DataName(dataIdx);
                            this.Widgets.(linename+"Line")(plotIdx).XData(1:oldMaxEpisodes) = oldData.(linename+"Line")(plotIdx).XData;
                            this.Widgets.(linename+"Line")(plotIdx).YData(1:oldMaxEpisodes) = oldData.(linename+"Line")(plotIdx).YData;
                            this.Widgets.(linename+"Line")(plotIdx).DataTipTemplate.DataTipRows(3).Value = @(x) this.Widgets.TimeStamps(x);
                        end
                    end
                end

                for loss_idx = 1:length(this.LossNames)
                    linename = strcat('Loss_',num2str(loss_idx),'_');
                    this.Widgets.(linename+"Line").XData(1:oldMaxEpisodes) = oldData.(linename+"Line").XData;
                    this.Widgets.(linename+"Line").YData(1:oldMaxEpisodes) = oldData.(linename+"Line").YData;
                end
            end
        end

        function update(this)
            % Update the UI
            
            options = this.Data.Options;
            data = this.Data.LastEpisodeInfo;

            % get user locale, this prevents issues during mismatch with
            % system locale (g2534350)
            locale = matlab.internal.datetime.getDatetimeSettings('locale');
            startTime = datetime(data.StartTime,'Locale',locale);

            episodeCt = data.EpisodeCount;
            
            % update each plot
            for plotIdx = 1:numel(options.AgentName)
                % for each logged metric, update the training plot
                for lineIdx = 1:numel(options.DataName)
                    if options.ShowOnFigure(lineIdx)
                        dataname = options.DataName(lineIdx);
                        line = this.Widgets.(dataname+"Line")(plotIdx);
                        episodeData = data.(dataname);
                        line.YData(episodeCt) = episodeData(plotIdx);
                        this.Widgets.TimeStamps(episodeCt) = string(datetime('now')-startTime);
                    end
                end
            end


            for loss_idx = 1:length(this.LossNames)
                dataname = strcat('Loss_',num2str(loss_idx),'_');
                line = this.Widgets.(dataname+"Line");
                episodeData = data.('Loss')(loss_idx);
                line.YData(episodeCt) = episodeData(plotIdx);
                this.Widgets.TimeStamps(episodeCt) = string(datetime('now')-startTime);
            end
            
            % update the x-axis limit
            if data.ShowLastN
                xmin = max(0,episodeCt-data.ShowLastNValue+1);
            else
                xmin = 0;
            end
            xlim = [xmin,episodeCt];
            updatePlotXLim(this,xlim);
        end
        
        function updatePlotXLim(this,lim)
            % update plot x limits
            
            for idx = 1:numel(this.Widgets.Ax)
                ax = this.Widgets.Ax(idx);
                ax.XLim = lim  + [0,1];  % allow padding of 1
            end
        end

        function updateTimeStamps(this,timeStamps)
            options = this.Data.Options;
            this.Widgets.TimeStamps(1:numel(timeStamps)) = timeStamps(:)';
            % update each plot
            for plotIdx = 1:numel(options.AgentName)
                for dataIdx = 1:numel(options.DataName)
                    if options.ShowOnFigure(dataIdx)
                        linename = options.DataName(dataIdx);
                        this.Widgets.(linename+"Line")(plotIdx).DataTipTemplate.DataTipRows(3).Value = @(x) this.Widgets.TimeStamps(x);
                    end
                end
            end
        end
    end
end

%% Local functions

function layout = localDocumentLayout(numDocuments,groupTag)
% create a default layout for the view

% document layout
layout = struct;

% set document grid
% For numDocuments < 3, make a single column layout
% otherwise, make a two column layout
if numDocuments<4
    layout.gridDimensions.w = 1;
    layout.gridDimensions.h = numDocuments;
else
    layout.gridDimensions.w = 2;
    layout.gridDimensions.h = ceil(numDocuments/2);
end

% set tile count
layout.tileCount = numDocuments;

% set column and row weights
layout.ColumnWeights = 0.95;

% Define which documents appear in which tile
% Id of a document is defined as <GroupTag>_<DocumentTag>
for idx = 1:numDocuments
    documentState.id = groupTag + "_document" + idx;
    tileChildren = documentState;
    tileOccupancy(idx).children = tileChildren; %#ok<AGROW>
end
layout.tileOccupancy = tileOccupancy;
end