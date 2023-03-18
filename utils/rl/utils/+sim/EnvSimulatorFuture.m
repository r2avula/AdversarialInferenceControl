classdef EnvSimulatorFuture < rl.env.internal.AbstractSimulatorFuture
% Based on MATLABSIMULATORFUTURE

% Revised: 7-26-2021
% Copyright 2021 The MathWorks, Inc.

    properties (Access = private)
        % wrapped future object
        F_
    end

    methods 
        function this = EnvSimulatorFuture(F,id)
            arguments
                F (1,1) 
                id (1,1) {mustBeInteger}
            end
            this.F_ = F;
            this.ID = id;
        end
    end

    methods (Access = protected)
        function [outs,errs] = fetchOutputs_(this)
            F = [this.F_];
            errs = cell(size(F));
            outs = [];
            try
                outs = fetchOutputs(F);
            catch ex %#ok<NASGU> 
                for i = 1:numel(F)
                    err = F(i).Error;
                    if ~isempty(err)
                        errs{i} = err.remotecause{1};
                    end
                end
            end
        end
        function [idx,out,err] = fetchNext_(this,varargin)
            F = [this.F_];
            err = [];
            idx = [];
            out = [];
            try
                [idx,out] = fetchNext(F,varargin{:});
            catch ex
                if isempty(ex.cause)
                    err = ex;
                else
                    err = ex.cause{1};
                end
            end
        end
        function cancel_(this)
            F = [this.F_];
            cancel(F);
        end
        function wait_(this,varargin)
            F = [this.F_];
            wait(F,varargin{:});
        end

        function val = getRead_(this)
            val = this.F_.Read;
        end
        function val = getState_(this)
            val = this.F_.State;
        end
        function val = getDiary_(this)
            val = this.F_.Diary;
        end
    end
end