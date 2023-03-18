classdef SimData < handle
% SIMDATA

% Revised: 7-7-2021
% Copyright 2021 The MathWorks, Inc.
    
    % data that should be trasnferred to the workers
    properties
        MaxSteps (1,1) {mustBePositive,mustBeInteger} = 500
        LogExperiences (1,1) logical = true
        NumAgents (1,1) {mustBePositive,mustBeInteger} = 1
        StopOnError (1,1) logical = true
        UseParallel (1,1) logical = false
        % parallel functions to execute on workers
        SetupFcn   = []
        CleanupFcn = []
    end

    % data that should NOT be transferred to the workers
    properties (Transient)
        % since the processors have to be installed on the block before
        % simulink compilation starts, we do NOT xfer the processors to the
        % workers
        ExperienceProcessors cell = {}
        AttachedFiles = {}
        TransferBaseWorkspaceVariables (1,1) logical = false
        WorkerRandomSeeds = -1;
    end

    methods
        function setupExperienceProcessors(this)
            % build the experience processors based on the number of
            % policies
            numAgents = this.NumAgents;
            % create the experience processors
            for i = numAgents:-1:1
                ep = sim.ExperienceProcessor();
                expProcessors{i} = ep;
            end
            this.ExperienceProcessors = expProcessors;
        end

        function setupExperienceProcessorsIfEmpty(this)
            % setup the experience processors if not already
            if isempty(this.ExperienceProcessors)
                setupExperienceProcessors(this);
            end
        end

        function updateData(this,agent,processExpFcn,processExpData)
            % update the data for a new round of simulations

            assert(~isempty(this.ExperienceProcessors),...
                "ExperienceProcessors cannot be empty when updateData is called");
            for i = 1:this.NumAgents
                ep = this.ExperienceProcessors{i};

                % set max steps
                setMaxSteps(ep,this.MaxSteps,this.LogExperiences);
                % install the new policy
                setAgent(ep,agent(i));
                % install the processExperienceData
                setData(ep,processExpData{i});
                % installl the processExperienceFcn
                setFcn(ep,processExpFcn{i});
            end
        end

        function [agentData,processExpErrs] = createAgentData(this)
            expProcessors = this.ExperienceProcessors;
            for i = this.NumAgents:-1:1
                ep = expProcessors{i};

                % check for any errors executing the processExpFcn
                processExpErrs{i} = ep.ProcessExperienceError;

                exp = []; t = [];
                if this.LogExperiences
                    [exp,t] = getLoggedExperiences(ep);
                end
                agentData(i).Experiences               = exp;
                agentData(i).Time                      = t;
                agentData(i).EpisodeInfo               = getEpisodeInfoData(ep);
                agentData(i).ProcessExperienceData     = getData(ep);
                agentData(i).Agent                     = getAgent(ep);
            end
        end
        
        function [out,processExpErrs] = createSimOut(this,simInfo)
            out.SimulationInfo   = simInfo;
            [out.AgentData,processExpErrs] = createAgentData(this);
        end
    end
end