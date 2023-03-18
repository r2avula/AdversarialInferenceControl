classdef (Abstract) AbstractActorCriticAgent < matlab.mixin.Heterogeneous ...
        & rl.mixin.Learnable ...
        & matlab.mixin.Copyable
    
    properties 
        % Actor        
        Actor
        ActorOptimizer   

        % Critic
        Critic
        CriticOptimizer        

        % Agent options
        AgentOptions
    end

    properties (Dependent) 
        ObservationInfo
        ActionInfo
    end

    %======================================================================
    % Public API
    %======================================================================
    methods
        function agent = AbstractActorCriticAgent(actor, critic, options)
            agent.AgentOptions = options;
            agent = setActor(agent,actor);
            agent = setCritic(agent,critic);            
        end

        function [agent, Data] = learn(agent,experience, Data)
            [agent, Data] = learn_(agent,experience, Data);
        end

        function agent = setActor(agent, actor)
            % enable gradient acceleration
            if rl.util.rlfeature('BuiltinGradientAcceleration')
                actor = accelerateBuiltin(actor,true);
            else
                actor = accelerate(actor,true);
            end

            agent.Actor = actor;
            agent.ActorOptimizer = opt.ADAMOptimizer(agent.AgentOptions.ActorOptimizerOptions);
        end

        function agent = setCritic(agent,critic)
            % enable gradient acceleration
            if rl.util.rlfeature('BuiltinGradientAcceleration')
                critic = accelerateBuiltin(critic,true);
            else
                critic = accelerate(critic,true);
            end

            agent.Critic = critic;
            agent.CriticOptimizer = opt.ADAMOptimizer(agent.AgentOptions.CriticOptimizerOptions);
        end
            
        function observationInfo = get.ObservationInfo(agent)
            observationInfo = agent.Actor.ObservationInfo;
        end

        function actionInfo = get.ActionInfo(agent)
            actionInfo = agent.Actor.ActionInfo;
        end
            
        function actor = getActor(agent)
            actor = agent.Actor;
        end

        function critic = getCritic(agent)
            critic = agent.Critic;
        end

        function [action, actionInfo] = getAction(agent,observation)
            observation = rl.util.cellify(observation);
            [action, actionInfo] = getAction(agent.Actor,observation);
            action = rl.util.cellify(action);
        end
    end


    %======================================================================
    % Abstract methods
    %======================================================================
    methods (Abstract, Access = protected)
        [agent,Data] = learn_(agent,exp,Data);
    end


    methods (Static)
        function [agent,Data] = SeriesTrainerProcessExperienceFcn(agent,Exp,Data,~)
            [agent,Data] = learn(agent,Exp, Data);
        end
    end

    methods (Access = protected)
        function observationInfo = getObservationInfo(agent)
            observationInfo = agent.Actor.ObservationInfo;
        end
        
        function actionInfo = getActionInfo(agent)
            actionInfo = agent.Actor.ActionInfo;
        end

        function agent = setLearnableParameters_(agent,params)
            agent.Critic = setLearnableParameters(agent.Critic,params.Critic);
            agent.Actor = setLearnableParameters(agent.Actor,params.Actor);
        end

        function params = getLearnableParameters_(agent)
            params.Critic = getLearnableParameters(agent.Critic);
            params.Actor = getLearnableParameters(agent.Actor);
        end
    end
    

    %======================================================================
    % Helpers
    %======================================================================
    methods (Hidden, Sealed)
        function trainMgr = buildTrainingManager(agent,env,trainingOptions)
            % attach the training options to the agent and construct a
            % training manager.
            trainMgr = train.TrainingManager(env,agent,trainingOptions);
        end
    end

    methods (Sealed)
        % sealed public methods
        varargout = sim(agent,env,varargin)
        trainingStatistics = train(agent,env,varargin)
    end
end