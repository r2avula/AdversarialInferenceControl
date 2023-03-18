function [trainingResult, trainedAgent] = customTrain(agent, env, showGUI, trainOptsOrResult)
arguments
    agent
    env (1,1) 
    showGUI
    trainOptsOrResult = rlTrainingOptions()
end

%% Resolve optional arguments
if isscalar(trainOptsOrResult) && ismember(class(trainOptsOrResult),{'rl.option.rlTrainingOptions',...
        'rl.option.rlMultiAgentTrainingOptions'})
    % Regular training workflow, arg is a training options
    trainingOptions = trainOptsOrResult;
    checkpoint = [];

elseif all(isa(trainOptsOrResult,'rl.train.rlTrainingResult')) || all(isa(trainOptsOrResult,'train.RLTrainingResult'))
    % Resume training from checkpoint workflow
    checkpoint = trainOptsOrResult;
    trainingOptions = checkpoint(1).TrainingOptions;
else
    error(message('rl:general:errInvalidTrainArg'))
end

% [trainingOptions, checkpoint] = rl.util.resolveTrainOptionsAndCheckpoint(trainOptsOrResult,this);

%% check compatibility with deployment settings
if isdeployed && strcmpi(trainingOptions.Plots,'training-progress')
    error(message('rl:general:TrainingPlotNotDeployable'))
end

%% perform validations
validateStopTrainingFunction(trainingOptions);
validateSaveAgentFunction(trainingOptions);

% validate agents, environment and training options
% rl.util.validateAgentsWithEnv(this,env,trainingOptions);

if trainingOptions.UseParallel
    assertParpool(0);
end

%% create the training manager
trainMgr = train.TrainingManager(env,agent,trainingOptions, showGUI);
clnup    = onCleanup(@() localCleanup(trainMgr));

%% run the training
[trainingResult, trainedAgent] = run(trainMgr,checkpoint);
end

%% Local functions
function localCleanup(trainMgr)
% clean up the logger
logger = rl.logging.internal.DataManager.getData("Logger");
if ~isempty(logger)
    cleanup(logger);
end
% clean up the static workspace
rl.logging.internal.DataManager.reset();
% delete the training manager
delete(trainMgr);
end
