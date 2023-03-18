classdef EnvSimulator < handle
    % MATLABSIMULATOR

    % Revised: 7-7-2021
    % Copyright 2021 The MathWorks, Inc.

    properties (Access = private)
        StepFcn_  function_handle
        InitialObservationFcn_ function_handle
        GetExperienceStructFcn_ function_handle
    end
    properties (Access = protected)
        % internal run ID
        RunID_ = 1
    end
    methods
        function setup(this,simData)
            % distribute the random seeds to the workers
            if simData.UseParallel && ~isequal(simData.WorkerRandomSeeds,-2)
                localDistributeRandomSeedOnWorkers(simData.WorkerRandomSeeds);
            end

            % call subclass setup
            setup_(this,simData);
        end
        function cleanup(this,simData)
            % call subclass cleanup
            cleanup_(this,simData)
            % reset the run id
            this.RunID_ = 1;
        end
        function out = sim(this,simData,agent,processExpFcn,processExpData, FinishedEpisodeFcn)
            % call the implementation of sim_, note if simData.UseParallel
            % == true, out should be a future object
            out = sim_(this,simData,agent,processExpFcn,processExpData, FinishedEpisodeFcn);
            % update the run ID
            this.RunID_ = this.RunID_ + 1;
        end
    end
    methods
        function this = EnvSimulator(stepFcn,initialObsFcn, getExperienceStructFcn)
            this.StepFcn_ = stepFcn;
            this.InitialObservationFcn_ = initialObsFcn;
            this.GetExperienceStructFcn_ = getExperienceStructFcn;
        end
    end
    
    methods
        function observation = getInitialObservation(this)
            observation = cellify(getInitialObservation_(this));
        end
        function [next_observation,reward,isdone, LoggedSignals] = step(this, action, actionInfo)
            if iscell(action) && isscalar(action)
                action = action{1};
            end
            [next_observation,reward,isdone, LoggedSignals] = step_(this, action, actionInfo);
            next_observation = cellify(next_observation);
            reward = double(reward);
            isdone = uint8(isdone);
        end
    end

    methods (Access = protected)
        function observation = getInitialObservation_(this)
            observation = feval(this.InitialObservationFcn_);
        end
        function [next_observation,reward,isdone, LoggedSignals] = step_(this, action, actionInfo)
            [next_observation,reward,isdone, LoggedSignals] = feval(this.StepFcn_, action, actionInfo);
        end
        function setup_(~,simData)
            % multi-agent not supported in MATLAB yet
            simData.NumAgents = 1;

            if simData.UseParallel
                % xfer base ws vars
                if simData.TransferBaseWorkspaceVariables
                    clientBaseWSVars = localGetBaseWSVars();
                else
                    clientBaseWSVars = struct;
                end

                % update attached files
                if ~isempty(simData.AttachedFiles)

                    % ignore warning about attached files
                    ws = warning("off","parallel:lang:pool:IgnoringAlreadyAttachedFiles");
                    cln1 = onCleanup(@() warning(ws));

                    pool = gcp("nocreate");
                    updateAttachedFiles(pool);
                    addAttachedFiles(pool,simData.AttachedFiles);
                end

                fetchOutputs(parfevalOnAll(@() localSetupOnWorker(simData,clientBaseWSVars),0));
            end
        end
        function cleanup_(~,simData)
            if simData.UseParallel
                fetchOutputs(parfevalOnAll(@() localCleanupOnWorker(simData),0));
            end
        end
        function out = sim_(this,simData,agent,processExpFcn,processExpData, FinishedEpisodeFcn)

            % create a struct for data that will effect each simulation
            simPkg.SimData          = simData;
            simPkg.Agent           = agent;
            simPkg.ProcessExpFcn    = processExpFcn;
            simPkg.ProcessExpData   = processExpData;

            % simulate
            if simData.UseParallel
                out = parsimInternal_(this,simPkg);
            else
                out = simInternal_(this,simPkg,FinishedEpisodeFcn);
            end
        end
    end
    methods
        function out = simInternal_(this,simPkg, FinishedEpisodeFcn)
            % run the sim loop on this process

            simData         = simPkg.SimData;
            agent          = simPkg.Agent;
            processExpFcn   = simPkg.ProcessExpFcn;
            processExpData  = simPkg.ProcessExpData;

            % make sure the experience processors are setup if not already.
            % Note, when executed in parallel, the ExperienceProcessors
            % will be rebuild for each simulation since the
            % ExperienceProcessors property is transient.
            setupExperienceProcessorsIfEmpty(simData);

            % update the experience processors
            updateData(simData,agent,processExpFcn,processExpData);

            simInfo = struct("SimulationError",[]);
            maxSteps = simData.MaxSteps;
            expProcessor = simData.ExperienceProcessors{1};
            % set the step delay to zero, as the initial action will be
            % explicitly invoked
            expProcessor.StepDelay = 0;
            %             try
            % reset the experience processor
            reset(expProcessor);
            % get the initial observation
            obs = getInitialObservation(this);

            for i = 1:maxSteps
                % compute the action
                [act, actInfo] = evaluateAction(expProcessor,obs);

                % step the environment
                %                     try
                [nobs,rwd,isd, LoggedSignals] = step(this, act, actInfo);
                %                     catch ex
                %                         % report is used instead of addCause to preserve stack trace
                %                         rpt = [sprintf('\t') regexprep(getReport(ex),'\n','\n\t')];
                %                         error(message("rl:env:ErrorExecutingEnvStep",rpt));
                %                     end

                % form the experience
                exp = feval(this.GetExperienceStructFcn_, nobs,rwd,isd, LoggedSignals, obs, act);
                exp.step_index = i;

                % step the processor (includes policy - use the step
                % index for simTime multiplied by sample time)
                stopsim = processExperience(expProcessor,exp,i);
                if stopsim
                    break;
                end
                % delay the experience states
                obs = nobs;
                FinishedEpisodeFcn();
            end
            %             catch ex
            %                 % check for a processExperience error
            %                 if ~isempty(expProcessor.ProcessExperienceError)
            %                     % report is used instead of addCause to preserve stack trace
            %                     rpt = [sprintf('\t') regexprep(getReport(expProcessor.ProcessExperienceError),'\n','\n\t')];
            %                     error("rl:env:ErrorExecutingProcessExpFcn: %s",rpt);
            %                 end
            %                 if simData.StopOnError
            %                     rethrow(ex);
            %                 end
            %                 simInfo.SimulationError = ex;
            %             end

            % create outputs
            out = createSimOut(simData,simInfo);
        end
    end
    methods (Access = private)
        function F = parsimInternal_(this,simPkg)
            feval_future = parfeval(@() simInternal_(this,simPkg),1);

            F = sim.EnvSimulatorFuture(feval_future,this.RunID_);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function localSetupOnWorker(simData,clientBaseWSVars)
% call user SetupFcn
if ~isempty(simData.SetupFcn)
    feval(simData.SetupFcn);
end

% install base ws vars
if simData.TransferBaseWorkspaceVariables
    addedvars = localDistributeToBaseWS(clientBaseWSVars);
    localSetGetWorkerAddedVars(addedvars);
else
    localSetGetWorkerAddedVars({});
end
end
function localCleanupOnWorker(simData)
% call user CleanupFcn
if ~isempty(simData.CleanupFcn)
    feval(simData.CleanupFcn);
end

% clean the base ws vars
addedvars = localSetGetWorkerAddedVars();
if ~isempty(addedvars)
    varstr = sprintf('%s ',addedvars{:});
    cmd = sprintf('clear %s', varstr);
    evalin('base',cmd)
end
localSetGetWorkerAddedVars({});
end
function vars = localGetBaseWSVars()
% get the base ws variables
varList = evalin('base','who');

vars = struct;
for i = 1:numel(varList)
    val = evalin('base',varList{i});
    if ~localIsNonSerializable(val)
        vars.(varList{i}) = val;
    end
end
end
function addedvars = localDistributeToBaseWS(varStruct)
% check for vars already in the base ws (may be opened by the model)
wsvars = evalin('base','who');
fs = fields(varStruct);
addedvars = setdiff(fs,wsvars);

for i = 1:numel(addedvars)
    varname = addedvars{i};
    varval  = varStruct.(varname);
    assignin('base',varname,varval);
end
end
function val = localIsNonSerializable(var) %#ok<INUSD>
% determine if a variable is serializeable (just distribute everything
% for now - a warning will show what is not serialize-able in many
% cases)
val = false;
end
function addedvars = localSetGetWorkerAddedVars(addedvars)
persistent x;
if nargin
    x = addedvars;
end
addedvars = x;
end
function localDistributeRandomSeedOnWorkers(wseeds)
    pool = gcp("nocreate");
    numWorkers = pool.NumWorkers;

    % get the ids from the workers g2036797
    F = parfevalOnAll(@rl.util.getWorkerID,1);
    ids = fetchOutputs(F);

    % expand the worker random seeds
    if isscalar(wseeds)
        wseeds = wseeds*ones(1,numWorkers);
    end
    if numel(wseeds) ~= numWorkers
        error(message('rl:general:ParallelTrainInvalidNumberOfWorkerSeeds',numel(wseeds),numWorkers));
    end

    % assign the seeds to the worker ids
    seedmap = containers.Map('KeyType','int32','ValueType','int32');
    for i = 1:numWorkers
        seedmap(ids(i)) = wseeds(i);
    end

    % setup the seeds
    fetchOutputs(parfevalOnAll(@localSetupRandomSeedOnWorkers,0,seedmap));
end
function localSetupRandomSeedOnWorkers(seedmap)
id = rl.util.getWorkerID();
    ws = seedmap(id);
    if ws ~= -2
        % if set to default (-1) use the id for the seed
        if ws == -1
            ws = id;
        end
        rng(ws);
    end
end