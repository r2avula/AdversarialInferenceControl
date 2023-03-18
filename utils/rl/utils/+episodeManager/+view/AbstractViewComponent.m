classdef (Abstract) AbstractViewComponent < handle
    % AbstractViewComponent
    
    % Copyright 2021, The MathWorks, Inc.
    
    properties (Access = protected)
        Component
        Data
    end
    
    %% Public api
    methods
        function this = AbstractViewComponent(component)
            this.Component = component;
            this.Data = [];
        end
        
        function setData(this,data)
            this.Data = data;
        end
    end
    
    methods (Abstract)
        build(this)
        reset(this)
        update(this)
    end
end