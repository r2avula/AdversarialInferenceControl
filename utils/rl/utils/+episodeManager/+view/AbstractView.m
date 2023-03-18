classdef (Abstract) AbstractView < handle
    % AbstractView
    
    % Copyright 2021, The MathWorks, Inc.
    
    properties 
        % Container for the view
        Container
    end
    
    %% Public api
    methods
        function this = AbstractView(container)
            % VIEW = AbstractView(CONTAINER) creates a abstract view
            % template for packaging composite views.
            %
            % CONTAINER may be:
            %   1. matlab.ui.container.internal.AppContainer
            %   2. uifigure handle
            %   3. figure handle
            
            arguments
                container (1,1)
            end
            
            this.Container = container;
        end
        
        function delete(this)
            % delete(VIEW) deletes the view and its underlying components.
            delete_(this)
        end
        
        function container = getContainer(this)
            % CONTAINER = getContainer(VIEW) returns the container for
            % VIEW.
            container = this.Container;
        end
    end
    
    %% Protected methods
    methods (Abstract, Access = protected)
        delete_(this)
    end
end