classdef TrainingManager < handle
    properties
        Agent
        Environment
        TrainingOptions

        % episode manager
        EpisodeMgr
        ShowGUI

        % command line display utils
        percentDone
        session_timer_value
        historical_run_time_seconds = 0
        isActive
        incPercent
        printBuffer
    end

    properties (Hidden, SetAccess = private)
        % episode count "states"
        EpisodeCount = 0
        TotalEpisodeStepCount = 0

        % window "states"
        RewardsInAveragingWindow
        AdversarialRewardEstimatesInAveragingWindow
        StepsInAveragingWindow

        % training meta data
        TrainingStartTime
        TrainingElapsedTime
        Watch
        Devices
        LearnRates

        % simulation info
        SimulationInfo = {}

        % Training result
        TrainingStatistics

        % Training status flag to keep track of training and stop training
        % in Episode Manager.
        % 0 : Agent has met a stop training criteria
        % 1 : Agent is training
        TrainingStatus

        % Stop training reason and value
        TrainingStoppedReason
        TrainingStoppedValue

        % Previous training session info
        LastSessionEpisodeCount = 0
        LastSessionElapsedTime = seconds(0)
        
    end

    properties (Access = private,Transient)
        % listener to determine if stop training has been requested
        StopTrainingListener

        % listener for data rcv on worker
        DataReceivedOnWorkerListener


        % listeners for tasks on workers
        TaskListeners
    end

    events
        DataReceivedFromWorker
        TrainingManagerUpdated
        ManuallyTrainingFinished
    end

    methods
        function this = TrainingManager(env,agent,opt, ShowGUI)
            this.Environment     = env;
            this.Agent          = agent;
            this.TrainingOptions = opt;
            this.TrainingStatus  = ones(1,numel(this.Agent));
            this.TrainingStoppedReason = repmat("",1,numel(this.Agent));
            this.TrainingStoppedValue  = repmat("",1,numel(this.Agent));

            % watch
            this.Watch = nnet.internal.cnn.ui.adapter.Stopwatch();
            reset(this.Watch);

            % training start time
            this.TrainingStartTime = localDateTimeAsString(datetime('now'));
            this.ShowGUI = ShowGUI;

            this.initializeProgress();
        end

        function delete(this)
            delete(this.StopTrainingListener);
            delete(this.DataReceivedOnWorkerListener);
            delete(this.TaskListeners);
            terminateProgress(this);
        end

        function setActionMessage(this,msg)
            arguments
                this
                msg string {mustBeTextScalar}
            end
            % set an action message for the episode manager
            if isvalid(this)
                episodeMgr = this.EpisodeMgr;
                if ~isempty(episodeMgr) && isvalid(episodeMgr)
                    setActionMessage(episodeMgr,msg);
                end
            end
        end

        function msg = getActionMessage(this,id)
            arguments
                this
                id double = 1
            end
            % get an action message from the episode manager
            msg = '';
            if isvalid(this)
                episodeMgr = this.EpisodeMgr;
                if ~isempty(episodeMgr)
                    msg = getActionMessage(episodeMgr,id);
                end
            end
        end

        function reqSimulink = requiresSimulink(this)
            % is simulink needed to run the training
            reqSimulink = isa(this.Environment,'rl.env.SimulinkEnvWithAgent');
        end



        function stopTraining = update(this,episodeFinishedInfo)
            % update the manager once an episode finishes

            epinfo       = episodeFinishedInfo.EpisodeInfo   ;
            episodeCount = episodeFinishedInfo.EpisodeCount + this.LastSessionEpisodeCount;
            workerID     = episodeFinishedInfo.WorkerID      ;
            simInfo      = episodeFinishedInfo.SimulationInfo;

            numAgents = getNumAgents(this);

            % evaluate q0 on the agent
            for idx = 1:numAgents
                if epinfo(idx).StepsTaken > 0
                    q0 = evaluateQ0(this.Agent(idx),epinfo(idx).InitialObservation);
                else
                    q0 = 0;
                end
                epinfo(idx).Q0 = q0;
            end

            % attach the episode info
            this.SimulationInfo{episodeCount} = simInfo;

            % update "states"
            this.EpisodeCount          = episodeCount;
            this.TotalEpisodeStepCount = this.TotalEpisodeStepCount + [epinfo.StepsTaken];
            %             this.percentDone =  this.percentDone + epinfo.StepsTaken*this.incPercent;

            % compute info and update displays
            info = computeMetrics(this,epinfo);
            updateDisplaysFromTrainingInfo(this,info);

            % log data in experiment manager
            logger = rl.logging.internal.DataManager.getData("Logger");
            if ~isempty(logger) && isa(logger, "rl.logging.MonitorLogger")
                logdata.EpisodeReward = info.EpisodeReward;
                logdata.EpisodeMeanReward = info.EpisodeMeanReward;
                logdata.AverageReward = info.AverageReward;
                store(logger, "rl_experiment", logdata, episodeCount);
            end

            % update training stats
            updateTrainingStats(this,info);

            % save agent to disk if requested
            saveAgentToDisk(this,info);

            % stop training
            localStopTraining = checkLocalStopTraining(this,info);
            stopTraining = localStopTraining | checkGlobalStopTraining(this);

            % Update agent stop training in Episode Manager
            stopCriteria  = this.TrainingOptions.StopTrainingCriteria;
            currentStatus = this.TrainingStatus;
            for idx = 1:numAgents
                if currentStatus(idx) && localStopTraining(idx)
                    % update training status
                    this.TrainingStatus(idx) = 0;

                    % log reason and value
                    if isa(stopCriteria,'function_handle')
                        % for custom stop criteria
                        reason = ['@' func2str(stopCriteria)];
                        value = localStopTraining(idx);
                    else
                        reason = stopCriteria;
                        value  = this.TrainingOptions.StopTrainingValue(idx);
                    end
                    this.TrainingStoppedReason(idx) = reason;
                    this.TrainingStoppedValue(idx)  = string(value);

                    % update episode manager
                    if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                        stopTrainingAgent(this.EpisodeMgr,idx,reason,value);
                    end
                end
            end

            % tell the world that the training manager has been updated
            s                     = struct;
            s.EpisodeInfo         = epinfo;
            s.ComputedEpisodeInfo = info;
            s.EpisodeCount        = episodeCount;
            s.WorkerID            = workerID;
            ed                    = rl.util.RLEventData(s);
            notify(this,'TrainingManagerUpdated',ed);
        end

        function [stats, trainedAgent] = run(this,checkpoint)
            % run the training routine with setup and cleanup

            arguments
                this
                checkpoint {localMustBeCheckpointOrEmpty} = []
            end
            preTrain(this,checkpoint);
            train(this);
            stats = postTrain(this);
            trainedAgent = this.Agent;
        end

        function preTrain(this,checkpoint)
            % run before training occurs
            arguments
                this
                checkpoint {localMustBeCheckpointOrEmpty} = []
            end

            % reset the stopTrain flag
            this.stopTrain(false);

            numAgents = getNumAgents(this);

            % make sure the parallel configuration is supported
            if isa(this.TrainingOptions,'rl.option.rlTrainingOptions') && this.TrainingOptions.UseParallel

                % make sure a pool is available/startable. Note, the call to gcp here
                % will start the pool if settings permit
                if isempty(gcp)
                    error(message("rl:general:ParallelTrainNoPool"));
                end

                % make sure the pool is not a ThreadPool (not currently supported)
                if isa(gcp,"parallel.ThreadPool")
                    error(message("rl:general:ParallelTrainThreadPoolNotSupported"));
                end
            end

            % initialize the window "states"
            this.RewardsInAveragingWindow = cell(1,numAgents);
            this.AdversarialRewardEstimatesInAveragingWindow = cell(1,numAgents);
            this.StepsInAveragingWindow = cell(1,numAgents);
            for agentIdx = 1:numAgents
                numScoresToAverage = this.TrainingOptions.ScoreAveragingWindowLength(agentIdx);
                this.RewardsInAveragingWindow{agentIdx} = zeros(numScoresToAverage,1,'single');
                this.AdversarialRewardEstimatesInAveragingWindow{agentIdx} = zeros(numScoresToAverage,1,'single');
                this.StepsInAveragingWindow{agentIdx}   = zeros(numScoresToAverage,1,'single');
            end

            % initialize simulation info
            maxEpisodes = this.TrainingOptions.MaxEpisodes;
            this.SimulationInfo = cell(1,maxEpisodes);

            % build the saved agents directory
            if isempty(dir(this.TrainingOptions.SaveAgentDirectory))
                if ~strcmpi(this.TrainingOptions.SaveAgentCriteria,"none")
                    try
                        mkdir(this.TrainingOptions.SaveAgentDirectory);
                    catch ex
                        me = MException(message('rl:general:TrainingManagerUnableToCreateSaveDir',this.TrainingOptions.SaveAgentDirectory));
                        throw(addCause(me,ex));
                    end
                end
            end

            % initialize the episode manager
            initializeEpisodeManager(this,checkpoint);

            % Initialize training statistics
            initializeTrainingStatistics(this);

            % Initialize device and learn rates
            initializeDeviceAndLearnRate(this);

            % Load the checkpoint, if applicable
            loadCheckpoint(this,checkpoint);
            
        end

        function stats = postTrain(this)
            % return the training stats post train

            % reset the stopTrain flag
            this.stopTrain(false);

            episodeIndex = this.EpisodeCount;
            elapsedTime = this.LastSessionElapsedTime + getDurationSinceReset(this.Watch);
            elapsedTime.Format = 'hh:mm:ss';
            this.TrainingElapsedTime = char(elapsedTime);
            episodeMgr = this.EpisodeMgr;

            % Find the agents still learning
            activeAgents = find(this.TrainingStatus~=0);

            % determine stop training reason
            % longReason updates final results label
            % stopReason and stopValue updates table in more details dialog
            if isempty(activeAgents)
                % training stopped with all agent reaching stop criteria
                longReason = "Training finished.";

            elseif episodeIndex >= this.TrainingOptions.MaxEpisodes
                % training stopped with max episodes condition
                longReason = string(message('rl:general:TextMaxEpisodesLongReason'));
                stopReason = string(message('rl:general:TextMaxEpisodes'));
                stopValue  = string(message('rl:general:TextEpisode')) + " " + episodeIndex;

            else
                % user clicked stop training button
                longReason = "User clicked stop training button.";
                stopReason = "User clicked stop training button.";
                stopValue  = "Stop Episode" + " " + episodeIndex;
            end

            % stop training active agents
            for idx = 1:numel(activeAgents)
                if ~isempty(episodeMgr) && isvalid(episodeMgr)
                    stopTrainingAgent(episodeMgr,activeAgents(idx),stopReason,stopValue);
                end
                this.TrainingStatus(activeAgents(idx)) = 0;
                this.TrainingStoppedReason(activeAgents(idx)) = stopReason;
                this.TrainingStoppedValue(activeAgents(idx))  = string(stopValue);
            end

            % set the episode manager to stop training state
            if ~isempty(episodeMgr) && isvalid(episodeMgr)
                stopTraining(episodeMgr,longReason);
            end

            % clean up unused training statistics
            stats = cleanupTrainingStats(this);

            % create training info struct for analysis and recreating plot
            trainingInfoStruct = createTrainingInfoStruct(this,longReason);

            % set the info struct to the training result
            for idx = 1:getNumAgents(this)
                stats(idx) = setInformation(stats(idx), trainingInfoStruct(idx));
            end
            delete(episodeMgr);
        end

        function stats = cleanupTrainingStats(this)
            maxEpisodes = this.TrainingOptions.MaxEpisodes;
            episodeIndex = this.EpisodeCount;
            % Clean up unused training statistics
            rmidx = (episodeIndex+1):maxEpisodes;
            stats = this.TrainingStatistics;
            for agentIndex = 1 : getNumAgents(this)
                if ~isempty(stats(agentIndex))
                    stats(agentIndex) = remove(stats(agentIndex),rmidx);
                    % attach the simulation info to the output structure
                    simInfo = this.SimulationInfo(1:episodeIndex);
                    stats(agentIndex) = setSimulationInfo(stats(agentIndex), vertcat(simInfo{:}));
                else
                    stats(agentIndex) = setSimulationInfo(stats(agentIndex), vertcat(this.SimulationInfo{:}));
                end
            end
        end

        function trainingInfoStruct = createTrainingInfoStruct(this,finalResultText)
            % create training info struct for analysis and plots
            % trainingInfoStruct is a 1xN struct array, where N = number of
            % trained agents.
            numAgents = getNumAgents(this);
            agentName = createUniqueAgentNames(this);

            trainData = rl.logging.internal.DataManager.getData("TrainData");

            for idx = numAgents:-1:1
                trainingInfoStruct(idx).EnvironmentName = string(getNameForEpisodeManager(this.Environment));
                trainingInfoStruct(idx).AgentName       = agentName(idx);
                if isa(this.Environment,'AgentBlock')
                    trainingInfoStruct(idx).BlockPath = this.Environment.AgentBlock(idx);
                else
                    trainingInfoStruct(idx).BlockPath = [];
                end
                trainingInfoStruct(idx).TrainingOpts        = this.TrainingOptions;
                trainingInfoStruct(idx).HardwareResource    = this.Devices(idx);
                trainingInfoStruct(idx).LearningRate        = this.LearnRates{idx};
                trainingInfoStruct(idx).TrainingStartTime   = this.TrainingStartTime;
                trainingInfoStruct(idx).ElapsedTime         = this.TrainingElapsedTime;
                trainingInfoStruct(idx).TimeStamp           = this.TrainingStatistics(idx).TimeStamp;
                trainingInfoStruct(idx).StopTrainingCriteria = this.TrainingStoppedReason(idx);
                trainingInfoStruct(idx).StopTrainingValue   = this.TrainingStoppedValue(idx);
                trainingInfoStruct(idx).FinalResult         = finalResultText;
                if ~isempty(trainData)
                    trainingInfoStruct(idx).TotalAgentLearnSteps = trainData.AgentLearnIndex(idx);
                else
                    trainingInfoStruct(idx).TotalAgentLearnSteps = [];
                end
            end
        end

        function train(this)
            % train the agent

            % create the trainer
            trainOpts = this.TrainingOptions;
            trainOpts.MaxEpisodes = trainOpts.MaxEpisodes - this.LastSessionEpisodeCount;
            trainer = train.createTrainer(this.Environment,this.Agent,trainOpts);

            % attach the trainer to the training manager
            attachTrainer(this,trainer);

            %             [progressData, progressDataQueue] = ProgressData('\t\t\tSimulating controller : ');
            %             incPercent = (1/numHorizons/k_num)*100;

            % on cleanup, detatch the trainer
            cln = onCleanup(@() detatchTrainer(this,trainer));
            
            if isa(trainer, 'rl.train.marl.MultiAgentTrainer')
                % pass the training status to the trainer to avoid training
                % agents that have finished training (resume training)
                trainer.DoTrain = this.TrainingStatus == 1;
            end

            % run the trainer
            this.Agent = run(trainer);
        end

        function attachTrainer(this,trainer)
            % attach the training manager to a trainer

            this.TaskListeners    = addlistener(trainer,'TasksRunningOnWorkers',...
                @(src,ed) setActionMessage(this,getString(message(...
                'rl:general:TrainingManagerRunningTasksOnWorkers'))));
            this.TaskListeners(2) = addlistener(trainer,'TasksCleanedUpOnWorkers',...
                @(src,ed) setActionMessage(this,getString(message(...
                'rl:general:TrainingManagerCleaningUpWorkers'))));
            this.TaskListeners(3) = addlistener(trainer,'ActionMessageReceived',...
                @(src,ed) setActionMessage(this,ed.Data));

            % set the update fcn here (listeners will drop events if not
            % marked as recursive)
            trainer.FinishedEpisodeFcn = @(info) update(this,info);
            trainer.FinishedStepFcn = @()updateCommandLineDisplayAfterStep(this);
        end

        function updateCommandLineDisplayAfterStep(this)
            % update the command line display
            this.percentDone =  this.percentDone + this.incPercent;
            if(this.isActive)
                episode_count = max(1,this.EpisodeCount -1);
                maReward = this.TrainingStatistics.AverageAdversarialEpisodeRewardEstimate(episode_count);
                proc_time = this.historical_run_time_seconds + toc(this.session_timer_value);
                time_unit = "seconds";

                timeRemaining = round((proc_time*(100-this.percentDone)/this.percentDone));
                if(timeRemaining>86400)
                    timeRemaining = timeRemaining/86400;
                    time_unit = "days";
                elseif(timeRemaining>3600)
                    timeRemaining = timeRemaining/3600;
                    time_unit = "hours";
                elseif(timeRemaining>60)
                    timeRemaining = timeRemaining/60;
                    time_unit = "minutes";
                end

                printBuffer_old = this.printBuffer;
                printBuffer_ = sprintf("MA Reward: %1.2f | Percent done: %3.2f | Est. time left: %3.2f%s",...
                    maReward, this.percentDone, timeRemaining, time_unit);
                reverseStr_ = repmat(sprintf('\b'), 1, strlength(printBuffer_old));
                fprintf(strcat(reverseStr_, printBuffer_));
                this.printBuffer = printBuffer_;
            end
        end

        function detatchTrainer(this,trainer)
            % detatch the trainer from the training manager
            delete(trainer);
            delete(this.TaskListeners);
        end

        function n = getNumAgents(this)
            n = numel(this.Agent);
        end

        function loadCheckpoint(this,checkpoint)
            % Load the data in checkpoint and set the training manager to
            % the previous training state
            if ~isempty(checkpoint)
                % Set TrainingManager properties
                this.EpisodeCount            = checkpoint(1).EpisodeIndex(end);
                this.LastSessionEpisodeCount = checkpoint(1).EpisodeIndex(end);
                this.LastSessionElapsedTime  = duration(checkpoint(1).Information(1).ElapsedTime);
                this.TrainingStartTime       = checkpoint(1).Information(1).TrainingStartTime;
                this.TrainingElapsedTime     = checkpoint(1).Information(1).ElapsedTime;

                this.historical_run_time_seconds = seconds(duration(this.TrainingElapsedTime,'InputFormat','hh:mm:ss'));
                this.percentDone = this.EpisodeCount*this.incPercent*this.TrainingOptions.MaxStepsPerEpisode;

                % Set the simulation info
                % If checkpoint is from a saved agent result it only
                % stores the siminfo from the last training episode.
                this.SimulationInfo(1:this.EpisodeCount) = ...
                    arrayfun(@(i) {checkpoint(1).SimulationInfo(i)}, 1:this.EpisodeCount);

                for idx = 1:numel(this.Agent)
                    % Set the training statistics
                    this.TrainingStatistics(idx) = update(this.TrainingStatistics(idx), ...
                        checkpoint(idx), 1:this.EpisodeCount);

                    this.TotalEpisodeStepCount(idx) = checkpoint(idx).TotalAgentSteps(end);

                    % Set averaging logs
                    window      = this.TrainingOptions.ScoreAveragingWindowLength(idx);
                    windowStart = max(1,this.EpisodeCount - window + 1);
                    windowEnd   = this.EpisodeCount;
                    this.RewardsInAveragingWindow{idx} = -abs(checkpoint(idx).EpisodeReward(windowStart:windowEnd));
                    this.AdversarialRewardEstimatesInAveragingWindow{idx} = -abs(checkpoint(idx).AdversarialEpisodeRewardEstimate(windowStart:windowEnd));
                    this.StepsInAveragingWindow{idx}   = checkpoint(idx).EpisodeSteps (windowStart:windowEnd);

                    % Set the TrainingStatus property.
                    % Agents will continue training if:
                    %    1. The previous session was terminated manually or 
                    %       by MaxEpisodes.
                    %       or,
                    %    2. The StopTrainingCriteria or StopTrainingValue 
                    %       training option was modified.
                    %
                    % TrainingStatus will be passed to the trainer in
                    % the TrainingManager/train method.
                    this.TrainingStatus(idx) = ...
                        double(checkManualStopTrainOrMaxEpisodes(checkpoint(idx)) || ...
                        checkStopTrainOptionChanged(checkpoint(idx),idx));

                    % update stop reason and value
                    if ~this.TrainingStatus(idx)
                        this.TrainingStoppedReason(idx) = checkpoint(idx).Information.StopTrainingCriteria;
                        this.TrainingStoppedValue(idx)  = checkpoint(idx).Information.StopTrainingValue;
                    end
                end
            end
        end
    end

    methods (Access = private)
        function initializeTrainingStatistics(this)
            % create an rlTrainingResult object
            for agentIndex = numel(this.Agent):-1:1
                if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                    sessionId = this.EpisodeMgr.Id;
                else
                    sessionId = [];
                end
                trainingResult{agentIndex} = train.RLTrainingResult(this.TrainingOptions,sessionId, length(this.EpisodeMgr.View.Document.LossNames));
            end
            this.TrainingStatistics = [trainingResult{:}];
        end

        function initializeEpisodeManager(this,checkpoint)
            % Initialize a new Episode Manager or reuse an existing one

            % Delete the listener
            delete(this.StopTrainingListener);

            % Episode Manager is spawned when Plots is set to
            % training-progress
            if strcmp(this.TrainingOptions.Plots,'training-progress')
                if isempty(checkpoint)
                    % Regular training branch, always create a new Episode
                    % Manager
                    if isa(this.Environment,'rl.env.SimulinkEnvWithAgent')
                        blk = this.Environment.AgentBlock;
                    else
                        blk = {};
                    end
                    this.EpisodeMgr = episodeManager.createEpisodeManager( ...
                        "new", ...
                        AgentName       = createUniqueAgentNames(this), ...
                        EnvironmentName = string(getNameForEpisodeManager(this.Environment)), ...
                        AgentBlock      = blk, ...
                        TrainingOptions = this.TrainingOptions,...
                        ShowGUI = this.ShowGUI);
                    setStartTime(this.EpisodeMgr,this.TrainingStartTime);
    
                else
                    % Resume training branch, create new or reuse an
                    % existing Episode Manager
                    this.EpisodeMgr = episodeManager.createEpisodeManager( ...
                        "reconstruct", ...
                        Checkpoint = checkpoint,...
                        ShowGUI = this.ShowGUI);
                    resumeTraining(this.EpisodeMgr,checkpoint);
                end
    
                % bridge request to terminate simulations from the episode
                % manager to the environment
                if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                    this.StopTrainingListener = addlistener(this.EpisodeMgr, ...
                        'RequestToStopTraining','PostSet', ...
                        @(src,ed) request2ManuallyTerminateCB(this,src,ed));
                end
            end
        end

        function initializeDeviceAndLearnRate(this)
            % get device and learning rate
            numAgents = getNumAgents(this);
            for agentIndex = numAgents:-1:1
                agent = this.Agent(agentIndex);
                try
                    actor = getActor(agent);
                    actorOptimOptions = agent.AgentOptions.ActorOptimizerOptions;
                    actorDevice    = actor.UseDevice;
                    actorLearnRate = actorOptimOptions.LearnRate;
                catch
                    actor = [];
                end
                try
                    critic = [getCritic(agent)];
                    criticOptimOptions = [agent.AgentOptions.CriticOptimizerOptions];
                    criticDevice    = critic(1).UseDevice;
                    criticLearnRate = criticOptimOptions(1).LearnRate;
                catch
                    critic = [];
                end
                % three cases: actor only, critic only, both actor and critic
                if ~isempty(actor) && ~isempty(critic)
                    devices(agentIndex).actorDevice  = actorDevice;
                    devices(agentIndex).criticDevice = criticDevice;
                    learnRates{agentIndex}  = [actorLearnRate,criticLearnRate];
                elseif ~isempty(actor) && isempty(critic)
                    devices(agentIndex).actorDevice  = actorDevice;
                    learnRates{agentIndex}  = actorLearnRate;
                elseif ~isempty(critic) && isempty(actor)
                    devices(agentIndex).criticDevice = criticDevice;
                    learnRates{agentIndex}  = criticLearnRate;
                else
                    devices(agentIndex).criticDevice = "unknown";
                    learnRates{agentIndex}  = 1;
                end
            end
            this.Devices = devices;
            this.LearnRates = learnRates;
        end

        function info = computeMetrics(this,epinfo)
            % returns relevant training progress metrics as a struct info.
            %
            %  Info.AverageSteps   : Running average of number of steps per episode
            %  Info.AverageReward  : Running average of reward per episode
            %  Info.EpisodeReward  : Reward for current episode
            %  Info.GlobalStepCount: Total times the agent was invoked
            %  Info.EpisodeCount   : Total number of episodes the agent has trained for
            %  Info.EpisodeSteps   : Number of episode steps
            %  Info.EpisodeQ0      : Episode Q0 value per episode
            %  Info.TrainingStatus : Training status of agents (0 or 1)

            episodeIndex    = this.EpisodeCount;
            episodeSteps    = [epinfo.StepsTaken];
            episodeReward   = [epinfo.CumulativeReward];
            episodeAdversarialRewardEstimate   = [epinfo.CumulativeAdversarialRewardEstimate];
            totalStepCount  = this.TotalEpisodeStepCount;
            q0              = [epinfo.Q0];

            % circular buffer index for averaging window
            numAgents = getNumAgents(this);
            for agentIdx = numAgents:-1:1
                numScoresToAverage = this.TrainingOptions.ScoreAveragingWindowLength(agentIdx);
                idx = mod(episodeIndex-1,numScoresToAverage)+1;
                this.RewardsInAveragingWindow{agentIdx}(idx) = episodeReward(agentIdx)/episodeSteps(agentIdx);
                this.AdversarialRewardEstimatesInAveragingWindow{agentIdx}(idx) = episodeAdversarialRewardEstimate(agentIdx)/episodeSteps(agentIdx);
                this.StepsInAveragingWindow{agentIdx}(idx)   = episodeSteps(agentIdx);
                numScores = min(episodeIndex,numScoresToAverage);
                avgReward(agentIdx) = sum(this.RewardsInAveragingWindow{agentIdx})/numScores;
                averageAdversarialRewardEstimate(agentIdx) = sum(this.AdversarialRewardEstimatesInAveragingWindow{agentIdx})/numScores;
                avgSteps(agentIdx) = sum(this.StepsInAveragingWindow{agentIdx})/numScores;
            end

            info.AverageSteps    = avgSteps;
            info.AverageEpisodeReward   = -avgReward;
            info.AverageAdversarialEpisodeRewardEstimate   = -averageAdversarialRewardEstimate;
            info.EpisodeReward   = -episodeReward./episodeSteps;
            info.AdversarialEpisodeRewardEstimate   = -episodeAdversarialRewardEstimate./episodeSteps;
            info.GlobalStepCount = totalStepCount;
            info.EpisodeCount    = episodeIndex;
            info.EpisodeSteps    = episodeSteps;
            info.EpisodeQ0 = -q0;
            info.TrainingStatus = this.TrainingStatus;
            %             info.ActorLoss = epinfo.AverageActorLoss;
            %             info.CriticLoss = epinfo.AverageCriticLoss;
            info.Loss = epinfo.AverageLoss;

            % information for model based agent
            isMBPOAgent = numel(this.Agent)==1 && isa(this.Agent,'rl.agent.rlMBPOAgent');
            isemptyModelTrainResults  = isfield(epinfo, "ModelTrainResults") && isempty(epinfo.ModelTrainResults);
            if isMBPOAgent && ~isemptyModelTrainResults  && ~isempty(epinfo.ModelTrainResults.TransitionLoss )
                info.TransitionLoss = epinfo.ModelTrainResults.TransitionLoss;
            end
            if isMBPOAgent && ~isemptyModelTrainResults  && ~isempty(epinfo.ModelTrainResults.RewardLoss )
                info.RewardLoss = epinfo.ModelTrainResults.RewardLoss;
            end
            if isMBPOAgent && ~isemptyModelTrainResults  && ~isempty(epinfo.ModelTrainResults.IsDoneLoss )
                info.IsDoneLoss = epinfo.ModelTrainResults.IsDoneLoss;
            end
        end

        function updateCommandLineDisplay(this,info)
            % update the command line display
            if(this.isActive)
                episode_count = max(1,this.EpisodeCount -1); 
                maReward = this.TrainingStatistics.AverageAdversarialEpisodeRewardEstimate(episode_count);
                proc_time = this.historical_run_time_seconds + toc(this.session_timer_value);
                time_unit = "seconds";

                timeRemaining = round((proc_time*(100-this.percentDone)/this.percentDone));
                if(timeRemaining>86400)
                    timeRemaining = timeRemaining/86400;
                    time_unit = "days";
                elseif(timeRemaining>3600)
                    timeRemaining = timeRemaining/3600;
                    time_unit = "hours";
                elseif(timeRemaining>60)
                    timeRemaining = timeRemaining/60;
                    time_unit = "minutes";
                end

                printBuffer_old = this.printBuffer;
                printBuffer_ = sprintf("MA Reward: %1.2f | Percent done: %3.2f | Est. time left: %3.2f%s",...
                    maReward, this.percentDone, timeRemaining, time_unit);
                reverseStr_ = repmat(sprintf('\b'), 1, strlength(printBuffer_old));
                fprintf(strcat(reverseStr_, printBuffer_));
                this.printBuffer = printBuffer_;

                if info.EpisodeCount == this.TrainingOptions.MaxEpisodes
                    terminateProgress(this);
                end
            end
        end

        function [timeTaken] = terminateProgress(this)
            episode_count = max(1,this.EpisodeCount -1);
            maReward = this.TrainingStatistics.AverageAdversarialEpisodeRewardEstimate(episode_count);
            if this.EpisodeCount == this.TrainingOptions.MaxEpisodes
                this.percentDone = 100;
            end
            timeTaken = toc(this.session_timer_value);
            time_unit = "seconds";
            if(timeTaken>86400)
                timeTaken = timeTaken/86400;
                time_unit = "days";
            elseif(timeTaken>3600)
                timeTaken = timeTaken/3600;
                time_unit = "hours";
            elseif(timeTaken>60)
                timeTaken = timeTaken/60;
                time_unit = "minutes";
            end

            printBuffer_old = this.printBuffer;
            printBuffer_ = sprintf("MA Reward: %1.2f | Percent done: %3.2f | Time taken: %3.0f%s\n",...
                maReward, this.percentDone, timeTaken, time_unit);
            reverseStr_ = repmat(sprintf('\b'), 1, strlength(printBuffer_old));
            fprintf(strcat(reverseStr_, printBuffer_));
            this.printBuffer = printBuffer_;
            this.isActive = false;
        end

        function  initializeProgress(this)
            fprintf(strcat("\t\t", class(this.Agent), " Training -- "));

            printBuffer_ = sprintf("MA Reward: %1.2f | Percent done: %3.2f | Est. time left: %3.0f%s",...
                nan, 0, nan, "");
            fprintf(printBuffer_);
            this.session_timer_value = tic;
            this.percentDone = 0;
            this.isActive = true;
            this.printBuffer = printBuffer_;
            this.incPercent = 100/this.TrainingOptions.MaxEpisodes/this.TrainingOptions.MaxStepsPerEpisode;
        end

        function updateEpisodeManager(this,info)
            % Push the training data onto the episode manager
            episodeMgr = this.EpisodeMgr;
            if ~isempty(episodeMgr) && isvalid(episodeMgr)
                stepEpisode(episodeMgr,info);
            end
        end

        function updateTrainingStats(this,info)
            % Keep track of statistics
            episodeIndex    = info.EpisodeCount;
            numAgents       = getNumAgents(this);
            for agentIndex = 1 : numAgents
                data.TimeStamp       = string(duration(getDurationSinceReset(this.Watch),'Format','hh:mm:ss'));
                data.EpisodeIndex    = episodeIndex;
                data.EpisodeReward   = info.EpisodeReward(agentIndex);
                data.AdversarialEpisodeRewardEstimate   = info.AdversarialEpisodeRewardEstimate(agentIndex);
                data.EpisodeSteps    = info.EpisodeSteps(agentIndex);
                data.AverageEpisodeReward   = info.AverageEpisodeReward(agentIndex);
                data.AverageAdversarialEpisodeRewardEstimate   = info.AverageAdversarialEpisodeRewardEstimate(agentIndex);
                data.TotalAgentSteps = info.GlobalStepCount(agentIndex);
                data.AverageSteps    = info.AverageSteps(agentIndex);
                data.EpisodeQ0   = info.EpisodeQ0(agentIndex);
                %                 data.ActorLoss = info.ActorLoss;
                %                 data.CriticLoss = info.CriticLoss;
                data.Loss = info.Loss;
                this.TrainingStatistics(agentIndex) = update(this.TrainingStatistics(agentIndex),data,episodeIndex);
            end
        end

        function saveAgentToDisk(this,info)
            % Save the agent to disk if the provided criteria has been met

            episodeIndex = info.EpisodeCount;
            saveValue = this.TrainingOptions.SaveAgentFunction(info);
            criteria = this.TrainingOptions.SaveAgentCriteria;
            windowLength = this.TrainingOptions.ScoreAveragingWindowLength;
            if any(saveValue & this.checkWindowPassed(criteria,episodeIndex,windowLength))
                numAgents = getNumAgents(this);
                if numAgents > 1
                    prefix = 'Agents';
                else
                    prefix = 'Agent';
                end
                SavedAgentFileName = fullfile(this.TrainingOptions.SaveAgentDirectory,[prefix, num2str(episodeIndex) '.mat']);
                saved_agent = this.Agent;
                savedAgentResult = createSavedAgentResult(this);
                % make sure the saved agent is in sim mode
                wasMode = [saved_agent.UseExplorationPolicy];
                try
                    save(SavedAgentFileName,'saved_agent', 'savedAgentResult');
                    if ~isempty(this.EpisodeMgr) && isvalid(this.EpisodeMgr)
                        savedMessage = getString(message('rl:general:TrainingManagerSavedAgent',prefix,episodeIndex));
                        setActionMessage(this.EpisodeMgr,savedMessage);
                    end
                catch
                    % g1928023: We do not want to interrupt the training
                    % due to saving errors. Therefore a warning is thrown.
                    warning(message('rl:general:TrainingManagerUnableToSaveAgent',this.TrainingOptions.SaveAgentDirectory))
                end
                % change the mode back
                for ct = 1:numAgents
                    saved_agent(ct).UseExplorationPolicy = wasMode(ct);
                end
            end
        end

        function savedAgentResult = createSavedAgentResult(this)
            % Create a rlTrainingResult object for the saved agent
            
            % Clean up unused training statistics
            savedAgentResult = cleanupTrainingStats(this);

            % Compute elapsed time for saved agent <geck>
            elapsedTime = this.LastSessionElapsedTime + getDurationSinceReset(this.Watch);
            elapsedTime.Format = 'hh:mm:ss';
            this.TrainingElapsedTime = char(elapsedTime);

            % Create and set the information struct
            trainingInfoStruct = createTrainingInfoStruct(this,getString(message('rl:general:TextTrainingInProgress')));
            for idx = 1:getNumAgents(this)
                savedAgentResult(idx) = setInformation(savedAgentResult(idx), trainingInfoStruct(idx));

                % Save only the last sim info <geck>
                if this.EpisodeCount > 1
                    msg = getString(message('rl:general:SimDataNotAvailable'));
                    if isa(this.Environment,'rl.env.SimulinkEnvWithAgent')
                        s.SavedAgentResultMessage = msg;
                        emptySimInfo = Simulink.SimulationOutput(s);
                    else
                        emptySimInfo.SavedAgentResultMessage = msg;
                        emptySimInfo.ErrorMessage = [];
                    end
                    if isempty(this.SimulationInfo{this.EpisodeCount})
                        simInfo = [repmat({emptySimInfo},1,this.EpisodeCount-1), {this.SimulationInfo{this.EpisodeCount}}]; %#ok<CCAT1> 
                    else
                        simInfo = [repmat({emptySimInfo},1,this.EpisodeCount-1), this.SimulationInfo(this.EpisodeCount)];
                    end
                    savedAgentResult(idx) = setSimulationInfo(savedAgentResult(idx), simInfo(:));
                end
            end
        end

        function stopFlag = checkGlobalStopTraining(this)
            % Stop training (manually requested, max episodes or all agent 
            % finished learning)
            % STOPFLAG is a 1xN logical array where N = num agents

            numAgents = getNumAgents(this);
            
            % If global stop condition then stop training all agents
            if this.EpisodeCount >= this.TrainingOptions.MaxEpisodes || ...
                    this.stopTrain() || all(this.TrainingStatus==0)
                stopFlag = true(1,numAgents);
            else
                stopFlag  = false(1,numAgents);
            end
        end

        function stopFlag = checkLocalStopTraining(this,info)
            % Stop training (when stopping criteria is true and averaging 
            % window has passed)
            % STOPFLAG is a 1xN logical array where N = num agents

            numAgents = getNumAgents(this);

            % check stop criteria
            stopFlag = this.TrainingOptions.StopTrainingFunction(info);
            if numAgents>1 && isscalar(stopFlag)
                % scalar expansion
                stopFlag = repmat(stopFlag,1,numAgents);
            end

            % continue training until at least the window has passed
            criteria = this.TrainingOptions.StopTrainingCriteria;
            windowLength = this.TrainingOptions.ScoreAveragingWindowLength;
            episodeIndex = info.EpisodeCount;
            stopFlag = stopFlag & this.checkWindowPassed(criteria,episodeIndex,windowLength);

            % Multi agent group stop learning:
            % Continue learning the group if at least one agent is learning
            if numAgents>1
                groups = this.TrainingOptions.AgentGroups;
                for ct = 1:numel(groups)
                    agentIndices = groups{ct};
                    if any(stopFlag(agentIndices)==false)
                        stopFlag(agentIndices) = false;
                    end
                end
            end
        end

        function updateDisplaysFromTrainingInfo(this,info)
            % update the user visible components
            updateCommandLineDisplay(this,info);
            if this.ShowGUI
                updateEpisodeManager(this,info);
            end
        end

        function request2ManuallyTerminateCB(this,~,~)
            % callback to manually terminate training
            this.stopTrain(true);
        end

        function [agentNames,agentTypes] = getAgentInfo(this)
            % return agent block names and types from the environment.
            numAgents = getNumAgents(this);
            agentNames = "";
            for idx = numAgents:-1:1
                if isa(this.Environment,'rl.env.SimulinkEnvWithAgent')
                    blkPaths = this.Environment.AgentBlock;
                    blkNames(idx) = string(get_param(blkPaths(idx),'Name'));
                    if numel(find(blkNames(idx)==blkNames)) > 1
                        % if there is another block with the same name (e.g.
                        % under a different subsystem) display full path
                        agentNames(idx) = blkPaths(idx);
                    else
                        agentNames(idx) = blkNames(idx);
                    end
                end
                agentTypes(idx) = string(regexprep(class(this.Agent(idx)),'\w*\.',''));
            end
        end

        function nameList = createUniqueAgentNames(this)
            numAgents = getNumAgents(this);
            nameList = "";
            for idx = numAgents:-1:1
                nameList(idx) = string(regexprep(class(this.Agent(idx)),'\w*\.',''));
            end
            nameList = matlab.lang.makeUniqueStrings(nameList);
        end
    end

    methods(Hidden)
        function mgr = getEpisodeManager(this)
            mgr = this.EpisodeMgr;
        end

        function qeSaveAgentToDisk(this,info)
            saveAgentToDisk(this,info);
        end
    end

    methods (Static)
        function stop = stopTrain(val)
            % static function to manage a "stop" train state

            persistent x_;
            if isempty(x_)
                x_ = false;
            end
            stop = x_;
            if nargin
                if isempty(val)
                    val = false;
                end
                x_ = logical(val);
            end
        end
    end

    methods (Static, Access = private)
        function result = checkWindowPassed(criteria,episodeCount,windowLength)
            % Check if the score averaging window length has passed
            if ismember(class(criteria),{'string','char'}) && ...
                    ismember(string(criteria),["AverageReward","AverageSteps"])
                % only return true if episodeCount > window length
                result = false(1,numel(windowLength));
                for ct = 1:numel(windowLength)
                    if episodeCount > windowLength(ct)
                        result(ct) = true;
                    end
                end
            else
                % always return true if not a window based criteria
                result = true(1,numel(windowLength));
            end
        end
    end
end

%% local utility functions
function str = localDateTimeAsString(dt)
defaultFormat = datetime().Format;
dt.Format = defaultFormat;
currentLocale = matlab.internal.datetime.getDatetimeSettings('locale');
str = char(dt,[],currentLocale);
end

function localMustBeCheckpointOrEmpty(checkpoint)
if ~isempty(checkpoint)
    mustBeA(checkpoint,'train.RLTrainingResult')
end
end
