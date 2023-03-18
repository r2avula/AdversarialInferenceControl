classdef EmbeddedView < rl.internal.episodeManager.view.AbstractAppContainerView
    % EmbeddedView
    
    % Copyright 2021, The MathWorks, Inc.
    
    methods
        function this = EmbeddedView(container,tab,document,panel,dialog,statusbar)
            % VIEW = EmbeddedView(CONTAINER,TAB,DOCUMENT,PANEL,DIALOG,STATUSBAR)
            % creates an composite view from the components TAB, DOCUMENT, 
            % PANEL, DIALOG and STATUSBAR.
            %
            % CONTAINER is a matlab.ui.container.internal.AppContainer.
            %
            % The VIEW object's purpose is to provide an api to package the 
            % components efficiently.
            
            this@rl.internal.episodeManager.view.AbstractAppContainerView(...
                container,tab,document,panel,dialog,statusbar);
        end
    end
    
    methods (Access = protected)
        function delete_(~)
            
            % no op, delete will be handled by parent app
        end
    end
end