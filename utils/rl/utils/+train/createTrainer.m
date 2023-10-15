function trainer = createTrainer(env,agent,trainingOptions)
if numel(agent) > 1
    % multi agent training (no parallel support yet)
    trainer = rl.train.marl.MultiAgentTrainer(env,agent,trainingOptions);

elseif isa(trainingOptions,'rl.option.rlTrainingOptions') && trainingOptions.UseParallel
    
    popts           = trainingOptions.ParallelizationOptions;
    isasync         = strcmp(popts.Mode,"async");

    if isa(agent,"rl.agent.mixin.InternalMemoryTrainable")
        if isasync
            % async experience based training (aka APE-X)
            trainer = rl.train.parallel.ExpAsyncParallelTrainer(env,agent,trainingOptions);
        else
            % sync experience based training (async should be preferred)
            trainer = rl.train.parallel.ExpSyncParallelTrainer(env,agent,trainingOptions);
        end    
    elseif isa(agent,"rl.agent.rlPPOAgent")
        if isasync
            % AsyncPPO
            trainer = rl.train.parallel.AsyncPPOParallelTrainer(env,agent,trainingOptions);
        else
            % SyncPPO
            trainer = rl.train.parallel.PPOParallelTrainer(env,agent,trainingOptions);
        end
    elseif isa(agent,"rl.agent.rlTRPOAgent")
        if isasync
            % AsyncTRPO
            trainer = rl.train.parallel.AsyncTRPOParallelTrainer(env,agent,trainingOptions);
        else
            % SyncTRPO
            trainer = rl.train.parallel.TRPOParallelTrainer(env,agent,trainingOptions);
        end

    elseif isa(agent,"rl.agent.rlACAgent")
        if isasync
            % A3C
            trainer = rl.train.parallel.A3CParallelTrainer(env,agent,trainingOptions);
        else
            % A2C
            trainer = rl.train.parallel.A2CParallelTrainer(env,agent,trainingOptions);
        end
    elseif isa(agent,"rl.agent.rlPGAgent")
        % PG (always sync)
        trainer = rl.train.parallel.PGParallelTrainer(env,agent,trainingOptions);
    else
        error(message("rl:agent:errNoParallelTrainerForAgent",class(agent)));
    end
elseif isa(agent,"rl.agent.rlMBPOAgent")
    % series training for MBPO Agent
        trainer = rl.train.MBPOSeriesTrainer(env,agent,trainingOptions);

elseif isa(agent,"agents.AbstractActorCriticAgent")
        trainer = train.ActorCriticAgentSeriesTrainer(env,agent,trainingOptions);

else
        % series training
        trainer = rl.train.SeriesTrainer(env,agent,trainingOptions);
end

