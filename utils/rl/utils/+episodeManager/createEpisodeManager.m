function episodeManager_ = createEpisodeManager(createMode,args)
% Create an Episode manager instance for one of the following scenarios.
%
% 1. New training mode. Create a new episode manager instance.
% 2. Reconstruct mode. Rebuild episode manager from previous checkpoint.
% 3. App mode. Rebuild from previous result in the RL app.

% Copyright 2018-2021 The MathWorks, Inc.

arguments
    createMode string {mustBeMember(createMode,["new","reconstruct","app"])}
    args.AgentName {mustBeText} = "Agent"
    args.EnvironmentName {mustBeTextScalar} = "Environment"
    args.AgentBlock = {}
    args.TrainingOptions = rlTrainingOptions()
    args.Checkpoint = []
    args.ShowGUI = true
end

switch createMode
    case "new"
        trainOpts = args.TrainingOptions;
        emOptions = episodeManager.EpisodeManagerOptions.createDefault(...
            "AgentName",       args.AgentName, ...
            "EnvironmentName", args.EnvironmentName, ...
            "AgentBlock",      args.AgentBlock, ...
            "TrainingOptions", trainOpts,...
            "ShowGUI", args.ShowGUI);
        uiid = [];

    case "reconstruct"
        % Backward compatibility: convert checkpoint (19a-21b training 
        % statistics struct) to rlTrainingResult object. 
        % This also performs validation of the struct.
        result = args.Checkpoint;
        if isstruct(result)
            result = train.RLTrainingResult.struct2class(args.Checkpoint);
        end
        try
            info = [result.Information];
            trainOpts = result(1).TrainingOptions;
            emOptions = episodeManager.EpisodeManagerOptions.createDefault(...
                "AgentName",       [info.AgentName], ...
                "EnvironmentName", info(1).EnvironmentName, ...
                "AgentBlock",      [info.BlockPath], ...
                "TrainingOptions", trainOpts,...
                "ShowGUI", args.ShowGUI);
            uiid = result(1).SessionId;
        catch
            error(message('rl:general:InvalidResultsStruct'));
        end

    case "app"
        result = args.Checkpoint;
        info = [result.Information];
        trainOpts = result.TrainingOptions;
        emOptions = episodeManager.EpisodeManagerOptions.createDefault(...
            "AgentName",       [info.AgentName], ...
            "EnvironmentName", info.EnvironmentName, ...
            "AgentBlock",      [info.BlockPath], ...
            "TrainingOptions", trainOpts,...
            "ShowGUI", args.ShowGUI);
        % For the app reconstruct case, do not use any existing episode
        % manager, always force a rebuild
        uiid = [];  
end

% If the gui is available then reuse the existing gui, else build new gui
if emOptions.ShowGUI
    ui = episodeManager.EpisodeManager.getHandleById(uiid);
else
    ui = [];
end

if ~isempty(ui) && isvalid(ui)
    % Reuse existing GUI
    episodeManager_ = ui;
else
    % Create new episode manager
    if isfield(trainOpts,'View') && emOptions.ShowGUI
        view = trainOpts.View;
    else
        view = [];
    end
    episodeManager_ = episodeManager.EpisodeManager(emOptions,view);
    if ismember(createMode, ["reconstruct", "app"])
        % Update episode manager
        updateWithTrainingResult(episodeManager_,result);
    end
end