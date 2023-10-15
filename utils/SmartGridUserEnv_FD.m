classdef SmartGridUserEnv_FD < handle
    %SMARTGRIDUSERENV:

    events
        EnvUpdated
    end

    %% Properties (set properties' attributes accordingly)
    properties
        z0_idx
        P_A0
        Params
        AgentClassString_or_Action2SubPolicyFcn
        P_AgU_YcAkn1
    end

    properties
        % Initialize internal flag to indicate episode termination
        IsDone = false
        k_idx
        state
        in_training_mode
        ExperienceStructFieldNames
    end

    %% Default properties from MATLABEnv
    properties
        Simulator_ = []
    end

    properties (Transient)
        % cached SimData created when the env is setup
        SimData_ = []
    end

    %% Necessary Methods
    methods
        function env = SmartGridUserEnv_FD(params, agentClassString_or_Action2SubPolicyFcn, pp_data, in_training_mode, ExperienceStructFieldNames)
            function_handles = get_vectorizing_function_handles(params);
            if(nargin==3)
                in_training_mode = true;
                ExperienceStructFieldNames = [];
            end

            env.ExperienceStructFieldNames      = ExperienceStructFieldNames;
            env.Params = params;
            env.Params.useparpool = false;
            env.Params.Function_handles = function_handles;
            env.Params.valid_XgYZn1 = pp_data.valid_XgYZn1;  
            env.Params.valid_YgXZn1 = pp_data.valid_YgXZn1;  
            env.Params.DRs_in_Rn = getDecisionRegionPolyhedrons(params,false);
            env.in_training_mode = in_training_mode;
            env.AgentClassString_or_Action2SubPolicyFcn = agentClassString_or_Action2SubPolicyFcn;
            if isa(agentClassString_or_Action2SubPolicyFcn, "string") || isa(agentClassString_or_Action2SubPolicyFcn, "char")
                getActionParamsInfoFcn = @(varargin) feval(strcat(env.AgentClassString_or_Action2SubPolicyFcn, ".getActionParamsInfo"), varargin{:});
                env.Params = getActionParamsInfoFcn(env.Params);
            else
                env.Params.subpolicy_params_num_con = env.Params.u_num;
            end

            if env.in_training_mode
                preprocessParamsForTrainingFcn =@(varargin) feval(strcat(agentClassString_or_Action2SubPolicyFcn, ".preprocessParamsForTraining"), varargin{:});
                [env.Params] = preprocessParamsForTrainingFcn(env.Params);
                env.P_AgU_YcAkn1 = [];
            else
                env.P_AgU_YcAkn1 = pp_data.P_AgU_YcAkn1;
            end
            reset(env);
        end

        function initial_observation = reset(env)
            params = env.Params;
            function_handles = params.Function_handles;
            a_num = params.a_num;
            HZs2A = function_handles.HZs2A;
            h_num = params.h_num;
            P_H0 = params.P_H0;
            paramsPrecision = params.paramsPrecision;
            z_num = params.z_num;

            if isempty(env.P_A0)                
                P_Z0 = ones(z_num,1)/z_num;
                P_A0_ = zeros(a_num,1);
                for h_idx = 1:h_num
                    P_A0_(HZs2A(h_idx, 1:z_num)) = P_H0(h_idx)*P_Z0;
                end
                P_A0_ = roundOffInSimplex(P_A0_,paramsPrecision);
                env.P_A0 = P_A0_;
            else
                P_A0_ = env.P_A0;
            end
            env.z0_idx = randi(z_num);

            if env.in_training_mode
                s_0_idx = env.generate_S([], env.z0_idx);
                env.state = {P_A0_, s_0_idx};
                initial_observation = [P_A0_;1];
            else
                initial_observation = P_A0_;
            end

            env.k_idx = 1;
        end

        function [next_observation, Reward, IsDone, LoggedSignals] = step(env, action, actionInfo)
            LoggedSignals = struct;

            params = env.Params;
            C_HgHh_design = params.C_HgHh_design;
            P_HgA = params.P_HgA;
            function_handles = params.Function_handles;
            S2XHAn1 = function_handles.S2XHAn1;

            if ~isempty(actionInfo) && isfield(actionInfo, 'P_Uk')
                P_Uk = actionInfo.P_Uk;
            else
                if isa(env.AgentClassString_or_Action2SubPolicyFcn, "string") || isa(env.AgentClassString_or_Action2SubPolicyFcn, "char")
                    action2SubPolicyFcn =@(varargin) feval(strcat(env.AgentClassString_or_Action2SubPolicyFcn, ".conAction2SubPolicy"), varargin{:});
                    [P_Uk] = action2SubPolicyFcn(params, action);
                else
                    [P_Uk] = feval(env.AgentClassString_or_Action2SubPolicyFcn, params, action);
                end
            end
            [x_k_idx_obs, h_k_idx, a_kn1_idx] = S2XHAn1(env.state{2});
            [Zk_idx, h_k_idx, YkIdx, Dk_idx, P_YksgSk] = updateSystemState(env, x_k_idx_obs, h_k_idx, a_kn1_idx, env.state{2}, P_Uk);
            [P_Aks, Hhk_idxs, P_YksgY12kn1, P_AgU_Yc] = env.get_possible_belief_transitions(params, env.state{1}, P_Uk, actionInfo.belief_trans_info, env.P_AgU_YcAkn1);

            P_Ak = P_Aks{YkIdx};
            Hhk_idx = Hhk_idxs(YkIdx);

            Reward = -C_HgHh_design(h_k_idx, Hhk_idx);
            if any(env.ExperienceStructFieldNames.matches("AdversarialRewardEstimate"))
                LoggedSignals.AdversarialRewardEstimate = -C_HgHh_design(:, Hhk_idx)'*P_HgA*P_Ak;
            end

            if any(env.ExperienceStructFieldNames.matches("MeanReward"))
                LoggedSignals.MeanReward = sum(-C_HgHh_design(Hhk_idxs, Hhk_idx).*P_YksgSk);
            end

            if any(env.ExperienceStructFieldNames.matches("P_YksgSk"))
                LoggedSignals.P_YksgSk = P_YksgSk;
            end

            if any(env.ExperienceStructFieldNames.matches("MeanAdversarialRewardEstimate"))
                LoggedSignals.MeanAdversarialRewardEstimate = env.computeMeanAdversarialRewardEstimate(params, P_Aks, Hhk_idxs, P_YksgY12kn1);
            end
			
            if any(env.ExperienceStructFieldNames.matches("P_YksgY12kn1"))
                LoggedSignals.P_YksgY12kn1 = P_YksgY12kn1;
            end  

            if any(env.ExperienceStructFieldNames.matches("P_AgU_Yc"))
                LoggedSignals.P_AgU_Yc = P_AgU_Yc;
            end
            if any(env.ExperienceStructFieldNames.matches("x_k_idx_obs"))
                LoggedSignals.x_k_idx_obs = x_k_idx_obs;
            end
            if any(env.ExperienceStructFieldNames.matches("Hk_idx"))
                LoggedSignals.Hk_idx = h_k_idx;
            end
            if any(env.ExperienceStructFieldNames.matches("Hhk_idx"))
                LoggedSignals.Hhk_idx = Hhk_idx;
            end
            if any(env.ExperienceStructFieldNames.matches("Dk_idx"))
                LoggedSignals.Dk_idx = Dk_idx;
            end
            if any(env.ExperienceStructFieldNames.matches("YkIdx"))
                LoggedSignals.YkIdx = YkIdx;
            end
            if any(env.ExperienceStructFieldNames.matches("Zk_idx"))
                LoggedSignals.Zk_idx = Zk_idx;
            end

            if env.in_training_mode
                skp1_idx = env.generate_S(h_k_idx, Zk_idx);
                env.state = {P_Ak, skp1_idx};
                if any(env.ExperienceStructFieldNames.matches("Hhk_idxs"))
                    LoggedSignals.Hhk_idxs =  Hhk_idxs;
                end
                if any(env.ExperienceStructFieldNames.matches("P_Aks"))
                    LoggedSignals.P_Aks = P_Aks;
                end
                if any(env.ExperienceStructFieldNames.matches("P_Uk"))
                    LoggedSignals.P_Uk = P_Uk;
                end       
                next_observation = [P_Ak; YkIdx];
            else
                next_observation = P_Ak;
            end

            IsDone = env.k_idx == params.k_num;
            env.IsDone = IsDone;
            if ~IsDone
                env.k_idx = env.k_idx + 1;
            end
        end

        function Exp = getExperienceStruct(env, nobs,rwd,isd, LoggedSignals, obs, act)  
            LoggedSignalsNames = string(fieldnames(LoggedSignals));
            for dataFieldName = env.ExperienceStructFieldNames
                if any(LoggedSignalsNames.matches(dataFieldName))
                    Exp.(dataFieldName) = LoggedSignals.(dataFieldName);
                elseif dataFieldName == "NextObservation"
                    Exp.NextObservation = {[nobs{1}(1:env.Params.a_num)]};
                elseif dataFieldName == "Action"
                    Exp.Action = cellify(act);
                elseif dataFieldName == "Observation"
                    Exp.Observation = {[obs{1}(1:env.Params.a_num)]};
                elseif dataFieldName == "Reward"
                    Exp.Reward = rwd;
                elseif dataFieldName == "IsDone"
                    Exp.IsDone = isd;
                end
            end
        end

        function simulator = getSimulator_(env,simInfo) %#ok<INUSD>
            if isempty(env.Simulator_)
                stepfcn = @(action,actionInfo) step(env, action, actionInfo);
                initialObsFcn  = @( ) getInitialObservation(env);
                getExperienceStructFcn  = @(nobs,rwd,isd, LoggedSignals, obs, act) getExperienceStruct(env, nobs,rwd,isd, LoggedSignals, obs, act);
                env.Simulator_ = sim.EnvSimulator(stepfcn, initialObsFcn, getExperienceStructFcn);
            end
            simulator = env.Simulator_;
        end

        function [z_k_idx, h_k_idx, yc_k_idx_star, d_k_idx_star, P_YksgSk] = updateSystemState(env, x_k_idx_obs, h_k_idx, a_kn1_idx, s_k_idx, P_Uk)
            params = env.Params;
            y_control_p_pu = params.y_control_p_pu;
            y_control_offset = params.y_control_offset;
            x_p_pu = params.x_p_pu;
            x_offset = params.x_offset;
            d_p_pu = params.d_p_pu;
            d_offset = params.d_offset;
            P_Zp1gZD = params.P_Zp1gZD;
            y_control_num = params.y_control_num;

            function_handles = params.Function_handles;
            A2HZ = function_handles.A2HZ;
            YcsS_2U = function_handles.YcsS_2U;
            y_control_range = 1:y_control_num;

            [~,z_kn1_idx] = A2HZ(a_kn1_idx);

            P_YksgSk = P_Uk(YcsS_2U(y_control_range,s_k_idx));
            cumulative_distribution = cumsum(P_YksgSk);
            yc_k_idx_star = find(cumulative_distribution>=rand(),1);
            if(isempty(yc_k_idx_star))
                error('isempty(yc_k_idx_star)');
            end
            
            d_k_idx_star = round(((yc_k_idx_star+y_control_offset)*y_control_p_pu - (x_k_idx_obs+x_offset)*x_p_pu)/d_p_pu) - d_offset;

            z_k_idx_prob = P_Zp1gZD(:,z_kn1_idx,d_k_idx_star);
            cumulative_distribution = cumsum(z_k_idx_prob);
            z_k_idx = find(cumulative_distribution>=rand(),1);
        end

        function setState(env, state)
            env.state = state;
        end

        function s_k_idx = generate_S(env, h_kn1_idx, z_kn1_idx)
            params = env.Params;
            P_H0 = params.P_H0;
            P_HgHn1 = params.P_HgHn1;
            P_XgH = params.P_XgH;
            function_handles = params.Function_handles;
            HZ2A = function_handles.HZ2A;
            XHAn1_2S = function_handles.XHAn1_2S;

            if (isempty(h_kn1_idx))
                env_step_distribution = P_H0;
                cumulative_distribution = cumsum(env_step_distribution);
                h_kn1_idx = find(cumulative_distribution>=rand(),1);
            end

            env_step_distribution = P_HgHn1(:,h_kn1_idx)';
            cumulative_distribution = cumsum(env_step_distribution);
            h_k_idx = find(cumulative_distribution>=rand(),1);

            x_distribution = P_XgH(:,h_k_idx)';
            cumulative_distribution = cumsum(x_distribution);
            x_k_idx = find(cumulative_distribution>=rand(),1);

            s_k_idx = XHAn1_2S(x_k_idx,h_k_idx,HZ2A(h_kn1_idx,z_kn1_idx));
        end
    end

    methods (Static)
        function [P_AgU_Yc] = preprocess_possible_belief_transitions(params, P_AgU_YcAkn1, P_Akn1)
            y_control_num = params.y_control_num;
            a_num = params.a_num;
            u_num = params.u_num;            
            paramsPrecision = params.paramsPrecision;            
            P_AgU_Yc = cell(y_control_num,1);
            NZ_P_Akn1_idxs = find(P_Akn1>0)';
            if params.useparpool
                parfor Yck_idx = 1:y_control_num
                    P_Akn1_ = P_Akn1;
                    P_AgU = sparse(a_num,u_num);
                    for Akn1_idx = NZ_P_Akn1_idxs
                        P_AgU = P_AgU + P_AgU_YcAkn1{Yck_idx, Akn1_idx}*P_Akn1_(Akn1_idx);
                    end
                    P_AgU(P_AgU<paramsPrecision) = 0;
                    P_AgU_Yc{Yck_idx} = (P_AgU);
                end
            else
                for Yck_idx = 1:y_control_num
                    P_Akn1_ = P_Akn1;
                    P_AgU = sparse(a_num,u_num);
                    for Akn1_idx = NZ_P_Akn1_idxs
                        P_AgU = P_AgU + P_AgU_YcAkn1{Yck_idx, Akn1_idx}*P_Akn1_(Akn1_idx);
                    end
                    P_AgU(P_AgU<paramsPrecision) = 0;
                    P_AgU_Yc{Yck_idx} = (P_AgU);
                end
            end
        end

        function [P_Aks, Hhk_idxs, P_YksgY12kn1, P_AgU_Yc] = get_possible_belief_transitions(params, P_Akn1, P_Uk, belief_trans_info, P_AgU_YcAkn1)
            if isempty(belief_trans_info)
                [P_AgU_Yc] = SmartGridUserEnv_FD.preprocess_possible_belief_transitions(params, P_AgU_YcAkn1, P_Akn1);
            elseif isfield(belief_trans_info, 'P_Aks')
                P_Aks = belief_trans_info.P_Aks;
                Hhk_idxs = belief_trans_info.Hhk_idxs;
                P_YksgY12kn1 = belief_trans_info.P_YksgY12kn1;
                P_AgU_Yc = belief_trans_info.P_AgU_Yc;
                return;
            elseif isfield(belief_trans_info, 'P_AgU_Yc')
                P_AgU_Yc = belief_trans_info.P_AgU_Yc;      
            else
                [P_AgU_Yc] = SmartGridUserEnv_FD.preprocess_possible_belief_transitions(params, P_AgU_YcAkn1, P_Akn1);
            end
            
            P_HgA= params.P_HgA;
            minLikelihoodFilter = params.minLikelihoodFilter;
            beliefSpacePrecision_adv = params.beliefSpacePrecision_adv;
            y_control_num = params.y_control_num;

            roundOffBelief_beliefSpacePrecision_fn = @(x)roundOffInSimplex(x,beliefSpacePrecision_adv);
            get_adv_guess_g_belief_k_fn = @(x)getHypothesisGuess(x,params.DRs_in_Rn);

            P_YksgY12kn1 = zeros(y_control_num, 1);
            Hhk_idxs = nan(y_control_num,1);
            P_Aks = cell(y_control_num,1);
            for Yck_idx = 1:y_control_num
                P_Ak = full(P_AgU_Yc{Yck_idx}*P_Uk);
                P_Aks{Yck_idx} = P_Ak;
                P_YksgY12kn1(Yck_idx) = sum(P_Ak);
            end
            P_YksgY12kn1 = P_YksgY12kn1./sum(P_YksgY12kn1);
            for Yck_idx = 1:y_control_num
                P_Ak = P_Aks{Yck_idx};
                P_Ak_sum = P_YksgY12kn1(Yck_idx);
                if(P_Ak_sum>=minLikelihoodFilter)
                    P_Ak = roundOffBelief_beliefSpacePrecision_fn(P_Ak);
                    Hhk_idx = get_adv_guess_g_belief_k_fn(P_HgA*P_Ak);
                    Hhk_idxs(Yck_idx) = Hhk_idx;
                    P_Aks{Yck_idx} = P_Ak;
                end
            end

            infeasible_y_idxs_flag = P_YksgY12kn1 < minLikelihoodFilter | isnan(Hhk_idxs);
            if(any(infeasible_y_idxs_flag))
                [P_Ak_sum_max_t,Yck_idx_t] = max(P_YksgY12kn1);
                if(P_Ak_sum_max_t>=minLikelihoodFilter)
                    Hhk_idxs(infeasible_y_idxs_flag) = Hhk_idxs(Yck_idx_t);
                    P_Aks(infeasible_y_idxs_flag) = repmat(P_Aks(Yck_idx_t), [1,sum(infeasible_y_idxs_flag)]);
                else
                    P_Aks(infeasible_y_idxs_flag) = repmat({P_Akn1}, [1,sum(infeasible_y_idxs_flag)]);
                    Hhk_idxs(infeasible_y_idxs_flag) = get_adv_guess_g_belief_k_fn(P_HgA*P_Akn1);                    
                end
            end
        end

        function [MeanAdversarialRewardEstimates] = computeMeanAdversarialRewardEstimate(params, P_Aks, Hhk_idxs, P_YksgY12kn1)
            C_HgHh_design = params.C_HgHh_design;
            P_HgA = params.P_HgA;
            [y_control_num, batchSize] = size(P_YksgY12kn1);
            AdversarialRewardEstimates = zeros(y_control_num,batchSize);
            for yk_idx = 1:y_control_num
                AdversarialRewardEstimates(yk_idx,:) = - sum(C_HgHh_design(:, Hhk_idxs(yk_idx,:)).*(P_HgA*P_Aks{yk_idx}),1);
            end
            MeanAdversarialRewardEstimates = sum(AdversarialRewardEstimates.*P_YksgY12kn1,1);
        end

        function [AdversarialRewardEstimates] = computeAdversarialRewardEstimates(params, experiences)
            C_HgHh_design = params.C_HgHh_design;
            P_HgA = params.P_HgA;
            y_control_num = params.y_control_num;
            Hhk_idxs = experiences.Hhk_idxs;
            P_Aks = experiences.P_Aks;

            AdversarialRewardEstimates = zeros(y_control_num,1);
            for yk_idx = 1:y_control_num
                AdversarialRewardEstimates(yk_idx) = -sum(C_HgHh_design(:, Hhk_idxs(yk_idx)).*(P_HgA*P_Aks{yk_idx}));
            end            
        end
    end

    %% Customised methods from AbstractEnv
    methods
        function setup(env,namedargs)
            arguments
                env (1,1)
                % general args
                namedargs.StopOnError (1,1) string ...
                    {mustBeMember(...
                    namedargs.StopOnError,...
                    ["on","off"])} = "on"
                % parallel args
                namedargs.UseParallel (1,1) logical {mustBeNumericOrLogical} = false;
                namedargs.SetupFcn   {localMustBeSetupFcn  }  = [];
                namedargs.CleanupFcn {localMustBeCleanupFcn}  = [];
                namedargs.TransferBaseWorkspaceVariables (1,1) string ...
                    {mustBeMember(...
                    namedargs.TransferBaseWorkspaceVariables,...
                    ["on","off"])} = "on"
                namedargs.AttachedFiles {localMustBeAttachedFiles} = string.empty;
                namedargs.WorkerRandomSeeds {mustBeNumeric,localMustBeWorkerRandomSeeds} = -1;
            end

            if isSetup(env)
                cleanup(env);
            end

            % create the simData object
            simData = sim.SimData();
            simData.StopOnError = strcmp(namedargs.StopOnError,"on");
            % assign parallel attributes
            simData.UseParallel = namedargs.UseParallel;
            if simData.UseParallel

                % make sure a pool is available. gcp will create a pool if
                % settings permit
                if isempty(gcp)
                    error(message("rl:general:ParallelSimNoPool"))
                end

                % make sure the pool is not a ThreadPool (not currently
                % supported)
                if isa(gcp,"parallel.ThreadPool")
                    error(message("rl:general:ParallelSimThreadPoolNotSupported"));
                end

                simData.SetupFcn        = namedargs.SetupFcn;
                simData.CleanupFcn      = namedargs.CleanupFcn;
                simData.TransferBaseWorkspaceVariables = ...
                    strcmp(namedargs.TransferBaseWorkspaceVariables,"on");
                simData.AttachedFiles     = cellstr(string(namedargs.AttachedFiles));
                simData.WorkerRandomSeeds = namedargs.WorkerRandomSeeds;
            end

            % call simulator setup
            simulator = getSimulator_(env);
            try
                setup(simulator,simData);
            catch ex
                throwAsCaller(ex);
            end
            env.SimData_ = simData;
        end
    end


    %% Default methods from AbstractEnv
    methods
        function delete(env)
            cleanup(env);
        end

        function val = isSetup(env)
            % ISSETUP(ENV)
            % Check to see if the environment is setup to run
            % repeated simulations with runEpisode
            val = ~isempty(env.SimData_);
        end

        function cleanup(env)
            % CLEANUP(ENV)
            % Cleanup the environment after running repeated
            % simulations with runEpisode

            if isSetup(env)
                % call subclass cleanup
                simulator = getSimulator_(env);
                try
                    cleanup(simulator,env.SimData_);
                catch ex
                    throwAsCaller(ex);
                end
                env.SimData_ = [];
            end
        end

    end

    methods (Hidden)
        function name = getNameForEpisodeManager(env)
            name = regexprep(class(env),'\w*\.','');
        end
        function simData = getSimData(env)
            simData = env.SimData_;
        end
    end

    %% Default methods from MATLABEnv
    methods(Hidden)
        function observation = getInitialObservation(env)
            % GETINITIALOBSERVATION(ENV) overloadable function that returns
            % the observation of an environment at its initial state. By
            % default, GETINITIALOBSERVATION calls the reset function.

            observation = reset(env);
        end

    end

    methods (Access = protected)
        function notifyEnvUpdated(env)
            % call env function any time you want to tell the world that
            % the environment has been updated

            % fire EnvUpdated
            notify(env,'EnvUpdated');

            % fire user-defined callback for environments that do plotting
            % internally
            envUpdatedCallback(env);
        end

        function envUpdatedCallback(env) %#ok<MANU>
            % overload env function to execute code once the environment
            % has been updated
        end
    end
end

%% Default functions from AbstractEnv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function localMustBeSetupFcn(val)
if ~isempty(val)
    validateattributes(val,{'function_handle'},{'scalar'},'','SetupFcn');
    if nargin(val)
        error(message('rl:general:ParallelTrainSetupFcnInvalidIputs'));
    end
end
end
function localMustBeCleanupFcn(val)
if ~isempty(val)
    validateattributes(val,{'function_handle'},{'scalar'},'','CleanupFcn');
    if nargin(val)
        error(message('rl:general:ParallelTrainCleanupFcnInvalidIputs'));
    end
end
end
function localMustBeWorkerRandomSeeds(val)
if ~(numel(val) == 1 && (val == -1 || val == -2))
    validateattributes(val,{'numeric'},{'integer','vector','nonnegative'},'','WorkerRandomSeeds');
end
end
function localMustBeAttachedFiles(val)
if ~isempty(val)
    localMustBeText(val);
end
end
function localMustBeText(val)
arguments
    val {mustBeText} %#ok<INUSA>
end
end
