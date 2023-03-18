classdef StandaloneView < episodeManager.view.AbstractAppContainerView
    % StandaloneView
    
    % Copyright 2021, The MathWorks, Inc.
    
    methods
        function this = StandaloneView(opts)
            % VIEW = StandaloneView(OPTIONS) creates a composite view with
            % the components: TAB, DOCUMENT, PANEL, DIALOG and STATUSBAR. 
            %
            % Each component must be subclassed from AbstractViewComponent 
            % and have individual implementations of build, reset and 
            % update methods.
            %
            % The VIEW object's purpose is to provide an api for packaging
            % the components efficiently.
            
            arguments
                opts.Tag {mustBeTextScalar} = char(matlab.lang.internal.uuid)
                opts.NumDocuments (1,1) double {mustBeInteger,mustBePositive} = 1
                opts.LossNames = []
            end
            
            % create container
            appOptions.Tag = opts.Tag;
            container = matlab.ui.container.internal.AppContainer(appOptions);
            
            % create status bar
            statusbar = rl.internal.app.StatusBarManager(container,'statusbar');
            
            % create info panel
            panel = episodeManager.panel.InformationPanelWrapper(container);
            
            % create training document
            document = episodeManager.document.TrainingDocumentWrapper(container,'NumDocuments',opts.NumDocuments,'LossNames',opts.LossNames);
            
            % create dialog
            dialog = episodeManager.dialog.MoreDetailsDialogWrapper();
            
            % empty tab
            tab = [];
            
            % Disable toolstrip
            container.ToolstripEnabled = false;
            
            this@episodeManager.view.AbstractAppContainerView(container,tab,document,panel,dialog,statusbar);
        end
    end
    
    methods (Access = protected)
        function delete_(this)
            % Delete the dialog
            delete(this.Dialog)
            % Delete the container, also deletes other components
            delete(this.Container)
        end
    end
end