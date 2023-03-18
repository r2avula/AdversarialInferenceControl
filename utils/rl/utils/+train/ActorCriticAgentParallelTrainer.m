classdef ActorCriticAgentParallelTrainer < rl.train.Trainer
    % Based on A2CPARALLELTRAINER, SyncParallelTrainer, AbstractParallelTrainer

    % Revised: 8-4-2021
    % Copyright 2021 The MathWorks, Inc.

    properties (Access = private)
        ConstWorkerData_ = []
    end
    properties (Access = protected)
        Pool_
    end

    methods
        function this = ActorCriticAgentParallelTrainer(env,agent,trainOpts)
            arguments
                env (1,1) 
                agent 
                trainOpts rl.option.rlTrainingOptions
            end
            this = this@rl.train.Trainer(env,agent,trainOpts);

            % get the pool
            this.Pool_ = gcp();
        end

        function trainedAgent = run(this)

            % cleanup post train
            cln1 = onCleanup(@() cleanupPostRun_(this));
            % setup
            setupPreRun_(this);

            % get data from the training options
            trainOpts = this.TrainOpts;
            simOpts = getSimulationOptions(trainOpts);
            maxEpisodes = trainOpts.MaxEpisodes;
            maxSteps = simOpts.MaxSteps;

            % setup some "state" data for training
            finishedSimCount = 0;
            startedSimCount = 0;
            activeSims = 0;
            stopTraining = false;
            maxActiveSims = getMaxActiveSims_(this);

            logger = rl.logging.internal.DataManager.getData("Logger");

            % store the starting episode count (useful when resuming
            % training from a checkpoint)
            if ~isempty(logger)
                trainData = rl.logging.internal.DataManager.getData("TrainData");
                startEpisodeCt = trainData.EpisodeIndex;
            else
                startEpisodeCt = 0;
            end

            % run the parallel training loop
            F = sim.EnvSimulatorFuture.empty(1,0);
            while finishedSimCount < maxEpisodes

                % spawn the simulations - up to maxActiveSims active sims at a time
                remainingSims = maxEpisodes - startedSimCount;
                N = min(maxActiveSims,remainingSims);
                while activeSims < N
                    % update the started sim count
                    startedSimCount = startedSimCount + 1;
                    F(end+1) = spawnSim_(this,maxSteps); %#ok<AGROW>
                    activeSims = activeSims + 1;
                end

                % process the futures
                [F, outs, taskIDs] = processFutures_(this,F);
                numFinishedSims = numel(outs);
                for i = 1:numFinishedSims
                    out = outs(i);
                    id = taskIDs(i);

                    % increase finishedSimCount
                    finishedSimCount = finishedSimCount + 1;
                    % decrease activeSims
                    activeSims = activeSims - 1;

                    ad = out.AgentData;
                    simInfo = out.SimulationInfo;
                    episodeInfo = [ad.EpisodeInfo];

                    % log episode data
                    if ~isempty(logger)
                        % iteration must be set to finishedSimCount + startEpisodeCt
                        % (and not finishedSimCount) to preserve integrity
                        % of episodes when resuming training from a
                        % previous checkpoint.
                        iter = finishedSimCount + startEpisodeCt;
                        if ~isempty(logger.EpisodeFinishedFcn)
                            logdata.EpisodeCount   = iter;
                            logdata.Environment    = this.Env;
                            logdata.Agent          = this.Agent;
                            logdata.Experience     = [ad.Experiences];
                            logdata.EpisodeInfo    = episodeInfo;
                            logdata.SimulationInfo = simInfo;
                            store(logger, "rl_episode", logdata, iter);
                        end
                    end

                    % tell the world an episode has finished
                    info.EpisodeInfo    = episodeInfo;
                    info.EpisodeCount   = finishedSimCount;
                    info.WorkerID       = id;
                    info.SimulationInfo = simInfo;
                    stopTraining = notifyEpisodeFinishedAndCheckStopTrain(this,info);
                    drawnow();

                    % exit early during simulations
                    if stopTraining
                        break;
                    end
                end

                % stop training
                if stopTraining
                    % cancel any remaining futures
                    if ~isempty(F)
                        cancel(F);
                    end
                    break;
                end

                % write data periodically, this will write data for all
                % logging contexts.
                if ~isempty(logger) && mod(iter, logger.LoggingOptions.DataWriteFrequency)==0
                    write(logger);
                end
            end
        
            trainedAgent= this.Agent;
        end
    end
    methods (Access = protected)
        function learnFromWorkerData_(this) %#ok<MANU>
            % overloadable - default null op
        end
        function setupWorkers_(this)

            agent = this.Agent;
            maxsteps = this.TrainOpts.MaxStepsPerEpisode;

            % setup constant hyper params
            workerData.paramsPrecision = agent.Params.paramsPrecision;
            workerData.discountFactor_actor               = agent.Params.discountFactor_actor;
            workerData.discountFactor_critic               = agent.Params.discountFactor_critic;            
            workerData.ExperienceBuffer             = replay.RLReplayMemory(agent.ObservationInfo,agent.ActionInfo,maxsteps);
            workerData.valid_YcgS                     = this.Env.valid_YcgS;
            workerData.U2paramIdx                     = this.Env.U2paramIdx;
            workerData.YcsS_2U                     = this.Env.Function_handles.YcsS_2U;

            % distribute the data to all workers
            this.ConstWorkerData_ = parallel.pool.Constant(workerData);
        end
        function cleanupWorkers_(this)
            delete(this.ConstWorkerData_);
            this.ConstWorkerData_ = [];
        end
        function processSimOutputs_(this,outs)
            % average the gradients
            gbuffer = rl.util.GradientBuffer();

            trainData = rl.logging.internal.DataManager.getData("TrainData");
            %             actorLoss = single([]);
            %             criticLoss = single([]);

            for out = outs(:)'
                ad = out.AgentData;
                data = ad.ProcessExperienceData;
                g = data.Gradients;
                % append the gradient to the buffer
                append(gbuffer,g);
                % update train data counters
                if ~isempty(trainData)
                    trainData.EpisodeIndex   = trainData.EpisodeIndex + 1;
                    trainData.AgentStepIndex = trainData.AgentStepIndex + ad.EpisodeInfo.StepsTaken;
                    %                     actorLoss  = [actorLoss; g.ActorLoss];
                    %                     criticLoss = [criticLoss; g.CriticLoss];
                end
            end
            gavg = average(gbuffer);
            % optimize the agent from the gradients
            this.Agent = learnFromGradients(this.Agent, gavg);

            % log data
            logger = rl.logging.internal.DataManager.getData("Logger");
            if ~isempty(logger)
                % learnFromGradients does not update the agent learn counter,
                % so update it here.
                trainData.AgentLearnIndex = trainData.AgentLearnIndex + 1;

                % take the mean losses because we average grads
                %                 learnData.ActorLoss  = mean(actorLoss);
                %                 learnData.CriticLoss = mean(criticLoss);
                learnData.Agent      = this.Agent;
                rl.logging.internal.util.logAgentLearnData(learnData);
            end
        end

        function F = spawnSim_(this,maxSteps)
            % spawn an async simulation
            processExpFcn = @(agent, exp,processExpData, epInfo)train.ActorCriticAgentParallelTrainer.workerProcessExpFcn(agent, exp, processExpData, this.ConstWorkerData_, epInfo);
            processExpData.Gradients = [];

            F = sim.RunEpisode(this.Env, this.Agent,...
                ProcessExperienceFcn=processExpFcn,...
                ProcessExperienceData=processExpData,...
                MaxSteps=maxSteps,...
                LogExperiences=false,...
                CleanupPostSim=false);
        end

        function n = getNumWorkers_(this)
            n = this.Pool_.NumWorkers;
        end

        function n = getMaxActiveSims_(this)
            % overloadable - default = numWorkers
            n = getNumWorkers_(this);
        end
    end
    methods (Access = protected)
        function [F,outs,taskIDs] = processFutures_(this,F)
            outs    = fetchOutputs(F);
            taskIDs = [F.ID];
            processSimOutputs_(this,outs);
            % remove all futures
            F = sim.EnvSimulatorFuture.empty(1,0);
        end
    end

    methods (Access = private)
        function setupPreRun_(this)
            % setup the workers
            setupWorkers_(this);

            % set up data logging (this calls the setup methods on the
            % logger targets to create the log directories).
            logger = rl.logging.internal.DataManager.getData("Logger");
            if ~isempty(logger)
                rl.logging.internal.util.setupForBuiltInTraining(logger);
            end

            % setup the env - this will setup worker random seeds as well
            popts = this.TrainOpts.ParallelizationOptions;
            setup(this.Env,...
                StopOnError                         =this.TrainOpts.StopOnError             ,...
                UseParallel                         =true                                   ,...
                TransferBaseWorkspaceVariables      =popts.TransferBaseWorkspaceVariables   ,...
                AttachedFiles                       =popts.AttachedFiles                    ,...
                WorkerRandomSeeds                   =popts.WorkerRandomSeeds                ,...
                SetupFcn                            =popts.SetupFcn                         ,...
                CleanupFcn                          =popts.CleanupFcn                      );

            % notify tasks are running on workers
            notifyTasksRunning(this);
        end
        
        function cleanupPostRun_(this)
            % cleanup the logger
            logger = rl.logging.internal.DataManager.getData("Logger");
            if ~isempty(logger)
                cleanup(logger);
            end

            % cleanup the env
            cleanup(this.Env);

            % cleanup the workers
            cleanupWorkers_(this);

            % notify tasks are cleaned on workers
            notifyTasksCleanedUp(this);
        end
    end

    methods (Static)
        % Worker processExperienceFcn written as a static function for
        % testability
        function [agent,processExpData] = workerProcessExpFcn(agent, exp,  processExpData, constWorkerData, ~)
            workerData = constWorkerData.Value;

            % get the replay and append the experience
            replay = workerData.ExperienceBuffer;
            appendWithoutSampleValidation(replay,exp);

            % compute the gradients in batch at the end of episode
            if exp.IsDone > 0
                % compute gradients
                experiences = allExperiences(replay,ConcatenateMode="batch");
                processExpData.Gradients = agent.computeGradients(agent.Critic,agent.Actor,...
                    experiences, workerData);
                reset(replay);
            end
        end
    end
end