classdef ContinuousDeterministicActor < rl.function.AbstractActorFunction

    properties (Access = private)
        ModelInputMap
    end
    
    methods
        function this = ContinuousDeterministicActor(model_, observationInfo, actionInfo, nameValueArgs)
            arguments
                model_
                observationInfo (:,1) rl.util.RLDataSpec
                actionInfo (1,1) rl.util.rlNumericSpec
                nameValueArgs.ObservationInputNames string = string.empty
                nameValueArgs.UseDevice (1,1) string {mustBeMember(nameValueArgs.UseDevice,["cpu","gpu"])} = "cpu"
            end

            rl.internal.validate.validateModelType(model_);
            inputSize = [];
            outputSize = [];
            model_ = rl.internal.model.createInternalModel(model_, nameValueArgs.UseDevice, inputSize, outputSize);

            % mapping validation
            modelInputMap = rl.internal.validate.mapFunctionObservationInput(model_,observationInfo,nameValueArgs.ObservationInputNames);

            % Constructor            
            this = this@rl.function.AbstractActorFunction(model_, observationInfo, actionInfo, modelInputMap);
            
            this.ModelInputMap = modelInputMap;

            % number of action outputs = number of actions
            numAction = prod(actionInfo.Dimension);
            modelOutputSize = getSize(this,'output');
            if (prod(modelOutputSize{1}) ~= numAction)
                error(message('rl:function:errDeterministicActorOutputSizeIncompatible'));
            end
        end 

        function [grad, gradInfo] = customGradient(this, gradientFcn, inputData, varargin)
            [inputData,batchSize,sequenceLength] = processInputData_(this,inputData);
            [grad, gradInfo] = customGradient(this.Model_, gradientFcn, inputData, batchSize, sequenceLength, varargin{:});
        end


        function [inputData,batchSize,sequenceLength] = processInputData_(this,inputData)
            [batchSize,sequenceLength] = inferDataDimension(this.InputInfo_(1), inputData);
            inputData = rl.util.cellify(inputData);
        end

        
        function [action, actionInfo] = getAction(this, observation)
            arguments
                this
                observation (:,1) cell
            end

            if numel(observation) ~= numel(this.ObservationInfo)
                error(message('rl:function:errActorFcnGetActionIncorrectNumInputChannel'))
            end

            % evaluate model to get action and next state
            [action, state] = getAction_(this, observation);
            % reshape action according to action spec
            [batchSize,sequenceLength] = inferDataDimension(this.ObservationInfo(1), observation{1});
            action = reshapeData(this.ActionInfo,action,batchSize,sequenceLength);
            action = rl.util.cellify(action);
            action = {double(action{1})};
            actionInfo.state = state;
        end
    end

    methods (Access = protected)
        function [action, state] = getAction_(this, observation)
            [action, state] = evaluate(this, observation);
        end

        function this = setModel_(this, model)
            % reconstruct the function from new model and existing info
            useDevice = this.UseDevice;
            if isempty(this.ModelInputMap.Observation)
                this = rlContinuousDeterministicActor(model,this.ObservationInfo,this.ActionInfo);
            else
                this = rlContinuousDeterministicActor(model,this.ObservationInfo,this.ActionInfo,...
                    "ObservationInputNames",this.ModelInputMap.Observation);
            end
            this.UseDevice = useDevice;
        end
    end
end