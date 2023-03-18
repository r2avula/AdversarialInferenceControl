classdef (Abstract) AbstractAppContainerView < episodeManager.view.AbstractView
    % AbstractAppContainerView
    
    % Copyright 2021, The MathWorks, Inc.
    
    properties
        Tab
        Document
        Panel
        Dialog
        StatusBar
    end
    
    %% Public api
    methods
        function this = AbstractAppContainerView(container,tab,document,panel,dialog,statusbar)
            % VIEW = AbstractAppContainerView(CONTAINER,TAB,DOCUMENT,PANEL,DIALOG,STATUSBAR)
            % creates an abstract VIEW from the components TAB, DOCUMENT, 
            % PANEL, DIALOG and STATUSBAR.
            %
            % CONTAINER is a ui.container.internal.AppContainer.
            %
            % The VIEW object's purpose is to provide an api for packaging
            % the components efficiently.
            
            arguments
                container (1,1) 
                tab {mustBeScalarOrEmpty}
                document {mustBeScalarOrEmpty}
                panel {mustBeScalarOrEmpty}
                dialog {mustBeScalarOrEmpty}
                statusbar {mustBeScalarOrEmpty}
            end
            
            this@episodeManager.view.AbstractView(container);
            
            this.Tab = tab;
            this.Document = document;
            this.Panel = panel;
            this.Dialog = dialog;
            this.StatusBar = statusbar;
        end
        
        function tab = getTab(this)
            % TAB = getTab(VIEW) returns the tab associated with VIEW.
            
            tab = this.Tab;
        end
        
        function document = getDocument(this)
            % DOCUMENT = getDocument(VIEW) returns the document associated 
            % with VIEW.
            
            document = this.Document;
        end
        
        function panel = getPanel(this)
            % PANEL = getPanel(VIEW) returns the panel associated with VIEW.
            
            panel = this.Panel;
        end
        
        function dialog = getDialog(this)
            % DIALOG = getDialog(VIEW) returns the dialog associated 
            % with VIEW.
            
            dialog = this.Dialog;
        end
        
        function statusbar = getStatusBar(this)
            % STATUSBAR = getStatusBar(VIEW) returns the status bar 
            % associated with VIEW.
            
            statusbar = this.StatusBar;
        end
    end
end