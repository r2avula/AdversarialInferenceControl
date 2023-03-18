function varargout = RunEpisode(env,agent,namedargs)
arguments
    env (1,1)
    agent (:,1)
    namedargs.MaxSteps (1,1) {mustBeInteger,mustBePositive} = 500
    namedargs.ProcessExperienceFcn = {}
    namedargs.ProcessExperienceData = {}
    % If CleanupPostSim == true, the env will be cleanup at termination of
    % runEpisode
    namedargs.CleanupPostSim (1,1) logical = true
    % Log experinces for each policy. If set to false, experiences and time
    % will be [].
    namedargs.LogExperiences (1,1) logical = true
    namedargs.FinishedStepFcn = {}
end

% make sure the env is setup
if ~isSetup(env)
    setup(env);
end

% cleanup post sim if requested
if namedargs.CleanupPostSim
    cln1 = onCleanup(@() cleanup(env));
end

% extract the simData
simData = env.SimData_;
numAgents = simData.NumAgents;

try
    % validate the processExperienceFcn
    processExpFcn = localValidateProcessExperienceFcn(...
        namedargs.ProcessExperienceFcn,numAgents);
    
    % validate the processExperienceData
    processExpData = localValidateProcessExperienceData(...
        namedargs.ProcessExperienceData,numAgents);
catch ex
    throwAsCaller(ex);
end

% push max steps and log experiences onto the simData
simData.MaxSteps = namedargs.MaxSteps;
simData.LogExperiences = nargout > 0 && namedargs.LogExperiences;

% get the simulator (delegated to sub-class)
simulator = getSimulator_(env);

% run the episode
out = sim(simulator,simData,agent,processExpFcn,processExpData, namedargs.FinishedStepFcn);

% extract outputs
if nargout
    varargout{1} = out;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fcn = localValidateProcessExperienceFcn(fcn,numPolicies)
vargs = {'runEpisode','processExperienceFcn'};
if isempty(fcn)
    fcn = cell(1,numPolicies);
else
    % make sure fcn is a cell of function handles or a function
    validateattributes(fcn,["function_handle","cell"],{},vargs{:});
    if iscell(fcn)
        if isscalar(fcn) && numPolicies > 1
            [c{1:numPolicies}] = deal(fcn{1});
            fcn = c;
        else
            validateattributes(fcn,"cell",{"numel",numPolicies},vargs{:});
            cellfun(@(x) validateattributes(x,"function_handle",{},vargs{:}),fcn);
        end
    else
        [c{1:numPolicies}] = deal(fcn);
        fcn = c;
    end
end

function data = localValidateProcessExperienceData(data,numPolicies)
vargs = {'runEpisode','processExperienceData'};
if isempty(data)
    data = cell(1,numPolicies);
else
    if ~iscell(data) && numPolicies == 1
        data = {data};
    end
    validateattributes(data,"cell",{"numel",numPolicies},vargs{:});
end