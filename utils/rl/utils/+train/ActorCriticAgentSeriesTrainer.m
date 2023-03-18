classdef ActorCriticAgentSeriesTrainer < rl.train.Trainer
% SERIESTRAINER
%
% Train an agent against an environment on a single thread

% Revised: 7-2-2019
% Copyright 2019-2022 The MathWorks, Inc.

    properties (Hidden)
        FinishedStepFcn         = [] 
    end
    methods
        function this = ActorCriticAgentSeriesTrainer(env,agent,trainOpts)
            this = this@rl.train.Trainer(env,agent,trainOpts);
        end

        function trainedAgent = run(this)
            
            simOpts = getSimulationOptions(this.TrainOpts);
            N = simOpts.NumSimulations;
            maxSteps = simOpts.MaxSteps;

            % cleanup post train
            cln1 = onCleanup(@() cleanupPostRun_(this));
            logger = rl.logging.internal.DataManager.getData("Logger");

            if ~isempty(logger)
                rl.logging.internal.util.setupForBuiltInTraining(logger);
                processFcn = @rl.logging.internal.util.processExperienceWithLogging;
                logExp = true;
            else
                processFcn = @agents.AbstractActorCriticAgent.SeriesTrainerProcessExperienceFcn;
                logExp = false;
            end

            % create process experience data
            % storing the logger object in the process data improves
            % performance as opposed to accessing the logger from the
            % static workspace in rl.logging.internal.DataManager.
            processData.Logger = logger;

            for simCount = 1:N
                % book keeping train data for logging
                if ~isempty(logger)
                    trainData = rl.logging.internal.DataManager.getData("TrainData");
                    step(trainData, "EpisodeIndex");
                end
               
                out = sim.RunEpisode(...
                    this.Env,this.Agent,...
                    MaxSteps=maxSteps,...
                    ProcessExperienceFcn=processFcn,...
                    processExperienceData=processData,...
                    CleanupPostSim=false,...
                    LogExperiences=logExp,...
                    FinishedStepFcn=this.FinishedStepFcn);

                ad = out.AgentData;
                simInfo = out.SimulationInfo;
                episodeInfo = [ad.EpisodeInfo];
                this.Agent = ad.Agent;

                % package and log episode data
                if ~isempty(logger)
                    % iteration must be set to trainData.EpisodeIndex 
                    % (and not simCount) to preserve integrity of episodes 
                    % when resuming training from a previous checkpoint.
                    iter = trainData.EpisodeIndex;
                    if ~isempty(logger.EpisodeFinishedFcn)
                        logdata.EpisodeCount   = iter;
                        logdata.Environment    = this.Env;
                        logdata.Agent          = this.Agent;
                        logdata.Experience     = [ad.Experiences];
                        logdata.EpisodeInfo    = episodeInfo;
                        logdata.SimulationInfo = simInfo;
                        store(logger, "rl_episode", logdata, iter);
                    end

                    % write data periodically, this will write data
                    % for all logging contexts.
                    if mod(iter, logger.LoggingOptions.DataWriteFrequency)==0
                        write(logger);
                    end
                end

                % tell the world an episode has finished (don't notify
                % if the episode was manually terminated)
                info.EpisodeInfo    = episodeInfo;
                info.EpisodeCount   = simCount;
                info.WorkerID       = 0;
                info.SimulationInfo = simInfo;
                stopTraining = notifyEpisodeFinishedAndCheckStopTrain(this,info);
                drawnow();
                
                % exit early during simulations
                if stopTraining
                    break;
                end
            end
        
            trainedAgent = this.Agent;
        end

    end
        
    methods (Access = private)
        function cleanupPostRun_(this)
            cleanup(this.Env);
            % cleanup logger
            logger = rl.logging.internal.DataManager.getData("Logger");
            if ~isempty(logger)
                cleanup(logger);
            end
        end
    end
end