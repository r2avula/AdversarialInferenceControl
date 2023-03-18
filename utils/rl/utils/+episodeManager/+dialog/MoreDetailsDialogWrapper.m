classdef MoreDetailsDialogWrapper < controllib.ui.internal.dialog.AbstractDialog & ...
        episodeManager.view.AbstractViewComponent
    % MoreDetailsDialogWrapper Dialog for view agent details.
    
    % Copyright 2021 The MathWorks Inc.
    
    properties (Access = private)
        % Dialog
        Dialog
        
        % Table row names
        RowName
        
        % Table column names
        ColumnName
        
        % Dialog widgets (uitable and close button)
        Widgets
    end
    
    %% Public Methods
    methods
        function this = MoreDetailsDialogWrapper()
            % MoreDetailsDialogWrapper creates a dialog with a uitable.
            
            this@episodeManager.view.AbstractViewComponent([]);
            
            this.Name = 'MoreDetailsDialog';
            this.Title = "More Details";
            this.Data = [];
            this.RowName = [];
            this.ColumnName = [];
            this.Widgets = struct;
        end
        
        function delete(this)
            % delete the dialog
            
            close(this)
            delete@controllib.ui.internal.dialog.AbstractDialog(this)
        end
        
        function widgets = getWidgets(this,name)
            % get the dialog widgets
            
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
        
        function build(this)
            % construct the uifigure, this will also call buildUI under 
            % the hood
            
            getWidget(this);
        end
        
        function reset(this)
            % reset the dialog
            
            options = this.Data.Options;
            this.RowName = [...
                "Training";...
                "EpisodeNumber"; ...
                options.DisplayName(:); ...
                "AverageWindow";...
               "TrainingStoppedBy";...
                "TrainingStoppedAtValue" ];
            this.ColumnName = options.AgentName;
            tbl = this.Widgets.InfoTable;
            tbl.RowName = this.RowName(:);
            tbl.ColumnName = this.ColumnName(:);
            tbl.Data = repmat("",numel(this.RowName),numel(this.ColumnName));
            tbl.RowStriping = 'off';
        end
        
        function update(this)
            % update the UI if visible
            
            if this.IsVisible
                updateUI(this)
            end
        end
        
        function updateUI(this)
            data = this.Data.LastEpisodeInfo;
            options = this.Data.Options;
            tbl = this.Widgets.InfoTable;
            tabledata = repmat("",numel(string(tbl.RowName)),...
                numel(string(tbl.ColumnName)));
            tabledata(1,:) = data.TextStatus;
            tabledata(2,:) = data.EpisodeCount;
            for dataIdx = 1:numel(options.DataName)
                tabledata(2+dataIdx,:) = data.(options.DataName(dataIdx));
            end
            tabledata(end-2,:) = options.TrainingOptions.ScoreAveragingWindowLength;
            tabledata(end-1,:) = data.StopReason;
            tabledata(end,:) = data.StopValue;
            this.Widgets.InfoTable.Data = tabledata;
            this.Widgets.InfoTable.RowName = this.RowName(:);
            this.Widgets.InfoTable.ColumnName = this.ColumnName(:);
        end
    end
    
    %% Implementation of public abstract or overloaded methods
    methods (Access = protected)
        function buildUI(this)
            % buildUI overload to build dialog ui inside obj.UIFigure
            
            % Create layout grid
            grid = uigridlayout(this.UIFigure,[2,1]);
            grid.RowHeight = {'1x','fit'};
            grid.ColumnWidth = {'1x'};
            
            % create uitable
            tbl = uitable(grid);
            
            % close button
            g = uigridlayout(grid,[1 2]);
            g.ColumnWidth = {'1x',60};
            g.Padding = 0;
            CloseButton = uibutton(g,'push','Text',...
                "Close");
            CloseButton.Layout.Column = 2;
            
            this.Widgets.InfoTable = tbl;
            this.Widgets.CloseButton = CloseButton;
            
            this.UIFigure.Position(3:4) = [400,350];
        end
        
        function connectUI(this)
            % Set callbacks for UI
            this.Widgets.CloseButton.ButtonPushedFcn = @(~,~) closeCb(this);
        end
    end
    
    %% Private Methods
    methods (Access = private)
        function closeCb(this)
            close(this);
        end
    end
end