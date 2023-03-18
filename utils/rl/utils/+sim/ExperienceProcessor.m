classdef ExperienceProcessor < handle
    %Based on rl.env.internal.ExperienceProcessorInterface, rl.env.internal.PolicyExperienceProcessorInterface

    % Revised: 7-6-2021
    % Copyright 2021 The MathWorks, Inc.

    properties
        % how many steps to delay before calling processExperience_
        %   A delay = 1 allows the last_observation and last_action buffers
        %   to be filled, yielding a more meaningful experience
        StepDelay (1,1) {mustBeInteger,mustBeNonnegative} = 0
        Agent = []
    end
    properties (SetAccess = private)
        % process experience function error, caught for better reporting
        % with Simulink environments
        ProcessExperienceError = []
    end
    properties (Access = private)
        EpisodeInfo_
        MaxSteps_ = inf
        ExperienceLogger_ = []
        StepsTaken_ = 0
    end
    properties (Access = private)
        Fcn_ = []
        Data_ = []
    end
    
    methods
        function this = ExperienceProcessor()
            this.EpisodeInfo_ = sim.EpisodeInfo();
        end
        function data = getData(this)
            data = this.Data_;
        end
        function setData(this,data)
            this.Data_ = data;
        end
        function setFcn(this,fcn)
            this.Fcn_ = fcn;
        end
        function fcn = getFcn(this)
            fcn = this.Fcn_;
        end
        function agent = getAgent(this)
            agent = this.Agent;
        end
        function setAgent(this,agent)
            this.Agent = agent;
        end
        function ts = getSampleTime(this)
            % get the sample time from the agent
            agent = getAgent(this);
            if isempty(agent)
                ts = -1;
            else
                ts = getSampleTime(agent);
            end
        end
    end
    methods (Access = protected)
        function [action, actionInfo] = evaluateAction_(this,observation)
            observation = cellify(observation);
            [action, actionInfo] = getAction(this.Agent,observation{1});
        end
    end
    methods (Access = protected)
        function processExperience_(this,experience,info)
            if ~isempty(this.Fcn_)
                [this.Agent,this.Data_] = feval(this.Fcn_,...
                    this.Agent,experience,this.Data_,info);
            end
        end
    end

    methods
        function infoData  = getEpisodeInfoData(this)
            info = this.EpisodeInfo_;
            infoData.CumulativeReward   = info.CumulativeReward;
            infoData.CumulativeAdversarialRewardEstimate   = info.CumulativeAdversarialRewardEstimate;
            infoData.StepsTaken         = info.StepsTaken;
            infoData.InitialObservation = info.InitialObservation;
            if this.Data_.LearntEventsCount>0
                %                 infoData.AverageActorLoss = this.Data_.CumulativeActorLoss/this.Data_.LearntEventsCount;
                %                 infoData.AverageCriticLoss = this.Data_.CumulativeCriticLoss/this.Data_.LearntEventsCount;
                infoData.AverageLoss = this.Data_.CumulativeLoss/this.Data_.LearntEventsCount;
            else
                %                 infoData.AverageActorLoss = nan;
                %                 infoData.AverageCriticLoss = nan;
                infoData.AverageLoss = [];
            end
        end
    end

    methods
        function setMaxSteps(this,maxSteps,logExperiences)
            arguments
                this (1,1)
                maxSteps (1,1) {mustBePositive,mustBeInteger}
                logExperiences (1,1) logical
            end
            this.MaxSteps_ = maxSteps;
            if logExperiences
                this.ExperienceLogger_ = rl.util.ExperienceLogger(this.MaxSteps_);
            else
                this.ExperienceLogger_ = [];
            end
            reset(this);
        end
        function reset(this)
            this.ProcessExperienceError = [];
            reset(this.EpisodeInfo_);
            if ~isempty(this.ExperienceLogger_)
                reset(this.ExperienceLogger_);
            end
            this.StepsTaken_ = 0;
            %             this.Data_.CumulativeActorLoss = 0;
            %             this.Data_.CumulativeCriticLoss = 0;
            this.Data_.CumulativeLoss = [];
            this.Data_.LearntEventsCount = 0;
        end
        function val = getMaxSteps(this)
            val = this.MaxSteps_;
        end
        function val = isLogExperiences(this)
            val = ~isempty(this.ExperienceLogger_);
        end
        function [action, actionInfo] = evaluateAction(this,observation)
            % execute an action
%             try
                [action, actionInfo] = evaluateAction_(this,observation);
%             catch ex
%                 % store the error
%                 this.ProcessExperienceError = ex;
%                 % throw the error
%                 throw(ex);
%             end
        end
        function stopsim = processExperience(this,experience,simTime)
            arguments
                this (1,1)
                experience (1,1) struct
                simTime double {mustBeScalarOrEmpty} = []
            end
            % process and experience
%             try
                stopsim = processExperienceInternal_(this,experience,simTime);
%             catch ex
%                 % store the error
%                 this.ProcessExperienceError = ex;
%                 % throw the error
%                 throw(ex);
%             end
        end
        function info = getEpisodeInfo(this)
            info = this.EpisodeInfo_;
        end
        function exp = getExperienceLogger(this)
            exp = this.ExperienceLogger_;
        end
        function [exp,t] = getLoggedExperiences(this)
            t = []; exp = [];
            if isLogExperiences(this)
                t = getTimeVector(this.ExperienceLogger_);
                expcell = getExperiences(this.ExperienceLogger_);
                exp = vertcat(expcell{:});
            end
        end
    end
    methods (Access = private)
        function stopsim = processExperienceInternal_(this,experience,simTime)

            info = this.EpisodeInfo_;
            if this.StepsTaken_ >= this.StepDelay
                if isempty(experience.Reward)||...
                        ~isscalar(experience.Reward)||...
                        isinf(experience.Reward)||...
                        isnan(experience.Reward)||...
                        ~isreal(experience.Reward)

                    error(message("rl:env:ErrorRewardNotFiniteRealScalar"))
                end

                % update the episode info
                update(info,experience.Reward, experience.AdversarialRewardEstimate);
                experiencesProcessed = info.StepsTaken;

                % record the initial observation
                if experiencesProcessed == 1
                    info.InitialObservation = experience.Observation;
                end

                % if max steps reached, modify the IsDone signal
                maxStepsReached = experiencesProcessed >= this.MaxSteps_;
                if maxStepsReached && ~experience.IsDone
                    experience.IsDone = uint8(2);
                end

                % "step" against the experience
                infoData = getEpisodeInfoData(this);
                infoData.SimulationTime = simTime;
                processExperience_(this,experience,infoData);

                % update the logged experiences (check for isfull in case
                % the user sets sample time to -1, causing the exp block to
                % execute at every minor time step)
                if isLogExperiences(this) && ~isFull(this.ExperienceLogger_)
                    addExperience2Buffer(this.ExperienceLogger_,experience,simTime);
                end
            else
                % update the initial time
                if isLogExperiences(this)
                    setInitialTime(this.ExperienceLogger_,simTime);
                end
            end

            % check for termination either IsDone > 0
            stopsim = experience.IsDone > 0;

            % update the internal step count
            this.StepsTaken_ = this.StepsTaken_ + 1;
        end

    end
end