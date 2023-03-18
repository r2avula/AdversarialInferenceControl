classdef ADAMOptimizer < rl.optimizer.AbstractOptimizer
%classdef customADAMOptimizer < customAbstractOptimizer
    % ADAMOPTIMIZER   Adaptive Moment estimation (ADAM) solver for cells
    
    %   Copyright 2021 The MathWorks, Inc.
    
    properties
        % GradientDecayFactor   Decay factor for moving average of gradients.
        %   A real scalar in [0,1) specifying the exponential decay rate
        %   for the gradient moving average. This parameter is denoted by
        %   the symbol Beta1 in the ADAM paper.
        GradientDecayFactor
        
        % SquaredGradientDecayFactor   Decay factor for moving average of squared gradients.
        %   A real scalar in [0,1) specifying the exponential decay rate
        %   for the squared gradient moving average. This parameter is
        %   denoted by the symbol Beta2 in the ADAM paper.
        SquaredGradientDecayFactor
        
        % Epsilon   Offset for the denominator in the ADAM update.
        %   A positive real scalar specifying the offset to use in the
        %   denominator for the ADAM update to prevent divide-by-zero
        %   problems.
        Epsilon
    end
    
    properties (Access = private)
        % GradientMovingAverage   Moving average of gradients.
        %   A cell array of length NumLearnableParameters. Each element of
        %   the cell array contains the moving average of gradient for that
        %   learnable parameter.
        GradientMovingAverage
        
        % SquaredGradientMovingAverage   Moving average of squared gradients.
        %   A cell array of length NumLearnableParameters. Each element of
        %   the cell array contains the moving average of the squared
        %   gradient for that learnable parameter.
        SquaredGradientMovingAverage
        
        % NumUpdates   Number of updates so far.
        %   A non-negative integer indicating the number of update steps
        %   that have been computed so far.
        NumUpdates
    end
    
    methods
        function this = ADAMOptimizer(Options)
            % Constructor
            this = this@rl.optimizer.AbstractOptimizer(Options);
            Options = Options.OptimizerParameters;
            this.GradientDecayFactor = Options.GradientDecayFactor;
            this.SquaredGradientDecayFactor = Options.SquaredGradientDecayFactor;
            this.Epsilon = Options.Epsilon;
            this.NumUpdates = 0;
        end
    end

    methods (Access = protected)
        function [LearnableParameters, this] = update_(this,LearnableParameters,Gradients,GlobalLearnRate)
            
            LocalLearnRates = this.LocalLearnRates;
            NumLearnableParameters = this.NumLearnableParameters;
            
            Beta1 = this.GradientDecayFactor;
            Beta2 = this.SquaredGradientDecayFactor;
            Epsilon = this.Epsilon; %#ok<*PROPLC> 
            
            this.NumUpdates = this.NumUpdates + 1;
            LearnRateShrinkFactor = iCalculateLearnRateShrinkFactor(Beta1,Beta2,this.NumUpdates);
            
            processedGradients = cell(size(Gradients));
            for i = 1:NumLearnableParameters
                % No update needed for parameters that are not learning
                if any(LocalLearnRates{i},'all') && ~isempty(Gradients{i})
                    if any(isnan(Gradients{i}))
                        error('any(isnan(Gradients{i}))')
                    end
                    EffectiveLearningRate = LearnRateShrinkFactor.*GlobalLearnRate.*LocalLearnRates{i};
                    [processedGradients{i}, this.GradientMovingAverage{i}, this.SquaredGradientMovingAverage{i}] = nnet.internal.cnn.solver.adamstep(...
                        Gradients{i}, this.GradientMovingAverage{i}, this.SquaredGradientMovingAverage{i}, EffectiveLearningRate, Beta1, Beta2, Epsilon);
                    
                    % add gradients to get updated LearnableParameters
                    % similar to internal Layer API updateLearnableParameters()
                    LearnableParameters{i} = LearnableParameters{i} + processedGradients{i};
                    if any(isnan(LearnableParameters{i}))
                        error('any(isnan(LearnableParameters{i}))')
                    end
                end
            end
        end

        function this = internalInitialize_(this)
            this.GradientMovingAverage = iInitializeMovingAverage(this.NumLearnableParameters);
            this.SquaredGradientMovingAverage = iInitializeMovingAverage(this.NumLearnableParameters);
        end
    end
    
    %======================================================================
    % Save/Load
    %======================================================================
    methods
        function this = saveobj(this)
            % Save statistics as cpu array so users can load from non-gpu
            % machines
            if ~isempty(this.GradientMovingAverage)
                UseGPU = rl.internal.optimizer.rlAbstractSolver.useGPU(this.GradientMovingAverage);
                if UseGPU
                    this.GradientMovingAverage = dlupdate(@gather,this.GradientMovingAverage);
                    this.SquaredGradientMovingAverage = dlupdate(@gather,this.SquaredGradientMovingAverage);
                end
            end
        end
    end
end

%% Local functions
function MovingAverage = iInitializeMovingAverage(NumLearnableParameters)
MovingAverage = cell(1,NumLearnableParameters);
for i = 1:NumLearnableParameters
    MovingAverage{i} = single(zeros(1));
end
end

function LearnRateShrinkFactor = iCalculateLearnRateShrinkFactor(Beta1,Beta2,t)
LearnRateShrinkFactor = sqrt(1-Beta2^t)/(1-Beta1^t);
end