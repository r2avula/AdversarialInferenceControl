classdef DeterministicActorCriticAgent_RDC < agents.AbstractActorCriticAgent
    %% Policy aware adversary model
    properties
        Params

        % Training utils
        ReplayBuffer
        trained_episodes
        optimization_params
        P_AgW_YcAkn1
        noiseModel = []
    end

    methods
        function agent = DeterministicActorCriticAgent_RDC(actor, critic, agentOpts, params, ReplayBuffer, trained_episodes, noiseModel)
            if nargin == 4
                ReplayBuffer = [];
                trained_episodes = 0;
                noiseOpts = rl.option.OrnsteinUhlenbeckActionNoise;
                actionSize = {actor.ActionInfo.Dimension};
                noiseModel = rl.policy.noisemodel.rlOUNoiseModel(actionSize,noiseOpts,1);
                noiseModel.EnableStandardDeviationDecay = false;
            end
            agent = agent@agents.AbstractActorCriticAgent(actor, critic, agentOpts);
            agent.Params = params;
            if isempty(ReplayBuffer)
                ReplayBuffer = replay.RLReplayMemory(agent.ObservationInfo,agent.ActionInfo,agent.AgentOptions.ReplayBufferLength);
            end
            agent.ReplayBuffer = ReplayBuffer;    
            agent.trained_episodes = trained_episodes;
            agent.noiseModel = noiseModel;
        end
            
        function q0 = evaluateQ0(agent, Observation)
            arguments
                agent (1,1)
                Observation cell
            end
            action = getAction(agent.Actor, Observation);
            q0 = getValue(agent.Critic, Observation, action)*(1-agent.Params.discountFactor);
        end

        function setP_AgW_YcAkn1(agent, P_AgW_YcAkn1)
            agent.P_AgW_YcAkn1 = P_AgW_YcAkn1;
        end

        function [action, actionInfo] = getAction(agent, Observation)
            params = agent.Params;
            exploration_epsilon = params.exploration_epsilon;
            noise_epsilon = params.noise_epsilon;

            P_Akn1 = Observation;
            P_Bkn1 = params.P_BgA*P_Akn1;
            [P_AgW_Yc] = SmartGridUserEnv_RD.preprocess_possible_belief_transitions(params, agent.P_AgW_YcAkn1, P_Akn1);
            belief_trans_info.P_AgW_Yc = P_AgW_Yc;

            [action, actionInfo] = getAction(agent.Actor, {P_Bkn1});
            rand_draw = rand();
            if rand_draw < noise_epsilon
                [action,agent.noiseModel] = addNoise(agent.noiseModel,action);
                [P_Wk] = agent.conAction2SubPolicy(params, action{1}, []);
            else
                [P_Wk] = agent.conAction2SubPolicy(params, action{1}, actionInfo);
                [P_Aks, Hhk_idxs, P_YksgY12kn1] = SmartGridUserEnv_RD.get_possible_belief_transitions(params, P_Akn1, P_Wk, belief_trans_info, []);
                action_from_nnet = true;

                if rand_draw < exploration_epsilon + noise_epsilon
                    y_control_num = params.y_control_num;
                    P_Bks = zeros(params.b_num,1,y_control_num);
                    for y_idx = 1:y_control_num
                        P_Bks(:,1,y_idx) = params.P_BgA*P_Aks{y_idx};
                    end

                    experience = struct;
                    experience.P_Wk = P_Wk;
                    experience.P_Bks = P_Bks;
                    experience.Hhk_idxs = Hhk_idxs;
                    experience.P_YksgY12kn1 = P_YksgY12kn1;

                    [noisy_P_Wk] = agent.getRandomStrategy(P_Akn1, P_AgW_Yc, experience);
                    if ~isempty(noisy_P_Wk)
                        noisy_action = agent.conSubPolicy2Action(params, noisy_P_Wk);
                        action = (action{1}+noisy_action)/2;
                        [P_Wk] = agent.conAction2SubPolicy(params, action, []);
                        action_from_nnet = false;
                    end
                end
                if action_from_nnet
                    belief_trans_info.P_Aks = P_Aks;
                    belief_trans_info.Hhk_idxs = Hhk_idxs;
                    belief_trans_info.P_YksgY12kn1 = P_YksgY12kn1;
                end
            end

            actionInfo.P_Akn1 = P_Akn1;     
            actionInfo.P_Bkn1 = P_Bkn1;     
            actionInfo.belief_trans_info = belief_trans_info;         
            actionInfo.P_Wk = P_Wk;         
            action = rl.util.cellify(action);
        end

        function [noisy_P_Wk] = getRandomStrategy(agent, P_Akn1, P_AgW_Yc, experience_orig)
            params = agent.Params;
            y_num_for_exploration = params.y_num_for_exploration;
            gurobi_model = params.gurobi_model;
            gurobi_model_params = params.gurobi_model_params;
            Aeq_cons = params.Aeq_cons;
            beq_cons = params.beq_cons;
            eq_cons_num = params.eq_cons_num;
            paramsPrecision = params.paramsPrecision;
            minLikelihoodFilter = params.minLikelihoodFilter;
            num_rand_adv_strats_for_exploration = params.num_rand_adv_strats_for_exploration;
            b_num = params.b_num;
            P_BgA = params.P_BgA;
            function_handles = params.Function_handles;

            noisy_P_Wk = [];
            
            if isempty(agent.optimization_params)
                optimization_params_ = struct;
                optimization_params_.gurobi_model = params.gurobi_model;
                optimization_params_.gurobi_model_params = params.gurobi_model_params;
                optimization_params_.Aeq_cons = params.Aeq_cons;
                optimization_params_.beq_cons = params.beq_cons;
                optimization_params_.eq_cons_num = params.eq_cons_num;
                optimization_params_.DRs_in_Rn = params.DRs_in_Rn;
                optimization_params_.w_num = params.w_num;
                optimization_params_.C_HgHh_design = params.C_HgHh_design;
                optimization_params_.paramsPrecision = params.paramsPrecision;
                optimization_params_.minLikelihoodFilter = params.minLikelihoodFilter;
                optimization_params_.P_XHgHn1 = params.P_XHgHn1;
                optimization_params_.h_num = params.h_num;
                optimization_params_.z_num = params.z_num;
                optimization_params_.P_ZgA = params.P_ZgA;
                optimization_params_.P_HgB = params.P_HgB;
                optimization_params_.P_BgA = params.P_BgA;
                optimization_params_.P_HgA = params.P_HgA;                
                optimization_params_.beliefSpacePrecision_adv = params.beliefSpacePrecision_adv;
                optimization_params_.DRs_in_Rn = params.DRs_in_Rn;
                optimization_params_.y_control_num = params.y_control_num;
                optimization_params_.b_num = params.b_num;
                optimization_params_.a_num = params.a_num;
                optimization_params_.discountFactor = params.discountFactor;
                optimization_params_.penalty_factor = params.penalty_factor;
                optimization_params_.Function_handles = function_handles;
                optimization_params_.b_nn_cells = get_NN_ClassifyingConstraints(eye(b_num),1);
                agent.optimization_params = optimization_params_;
            else
                optimization_params_ = agent.optimization_params;
            end
            a_nn_cells = optimization_params_.b_nn_cells;

            [B,I] = sort(experience_orig.P_YksgY12kn1,'descend');
            filter_flag = B>=minLikelihoodFilter;
            if any(filter_flag)
                y_idxs_for_exploration = I(filter_flag);
                y_idxs_for_exploration = y_idxs_for_exploration(:)';
                y_num_for_exploration = min(length(y_idxs_for_exploration), y_num_for_exploration);
                y_idxs_for_exploration = y_idxs_for_exploration(1:y_num_for_exploration);

                possible_BhIdxk_g_Yck_flag = false(b_num,y_num_for_exploration);
                for Ye_idx_t = 1:y_num_for_exploration
                    Yck_idx = y_idxs_for_exploration(Ye_idx_t);
                    P_BgW = P_BgA*P_AgW_Yc{Yck_idx};
                    P_YcgW = sum(P_BgW,1);
                    P_YcgW_full = full(P_YcgW);

                    gurobi_model_t = gurobi_model;
                    gurobi_model_t.obj  = -P_YcgW_full;
                    gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
                    if strcmp(gurobi_result_t.status, 'OPTIMAL') && -gurobi_result_t.objval >= minLikelihoodFilter
                        for Bhk_idx = 1:b_num
                            Aineq_cons_HhIdx = a_nn_cells(Bhk_idx).A*P_BgW - a_nn_cells(Bhk_idx).b*P_YcgW;
                            bineq_cons_HhIdx = -paramsPrecision*ones(length(a_nn_cells(Bhk_idx).b),1);

                            Aineq_cons_t = [Aineq_cons_HhIdx;-P_YcgW];
                            bineq_cons_t = [bineq_cons_HhIdx;-(minLikelihoodFilter+paramsPrecision)];

                            gurobi_model_t.A = ([Aeq_cons;Aineq_cons_t]);
                            gurobi_model_t.rhs   = [beq_cons;bineq_cons_t];
                            gurobi_model_t.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
                            gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
                            if(strcmp(gurobi_result_t.status, 'OPTIMAL'))
                                possible_BhIdxk_g_Yck_flag(Bhk_idx,Ye_idx_t) = true;
                            end
                        end
                    end
                end

                if all(any(possible_BhIdxk_g_Yck_flag,1))
                    possible_BhIdx_given_Yc_vecs = find(possible_BhIdxk_g_Yck_flag(:,1))';
                    for ye_idx = 2:y_num_for_exploration
                        temp_idxs = find(possible_BhIdxk_g_Yck_flag(:,ye_idx))';
                        possible_BhIdx_given_Yc_vecs = combvec(possible_BhIdx_given_Yc_vecs,temp_idxs);

                        num_possible_BhIdx_given_Yc_vecs = size(possible_BhIdx_given_Yc_vecs,2);
                        if num_possible_BhIdx_given_Yc_vecs > num_rand_adv_strats_for_exploration
                            strat_idxs_to_search = randperm(num_possible_BhIdx_given_Yc_vecs,num_rand_adv_strats_for_exploration);
                            possible_BhIdx_given_Yc_vecs = possible_BhIdx_given_Yc_vecs(:, strat_idxs_to_search);
                        end
                    end
                    num_possible_BhIdx_given_Yc_vecs = size(possible_BhIdx_given_Yc_vecs,2);

                    experiences = struct;
                    experiences.P_Wk = [];
                    experiences.P_Bks = [];
                    experiences.P_YksgY12kn1 = [];
                    experiences.MeanAdversarialRewardEstimate = [];
                    parfor BhIdx_given_Yc_vec_idx = 1:num_possible_BhIdx_given_Yc_vecs
                        [experiences(BhIdx_given_Yc_vec_idx)] =...
                            DeterministicActorCriticAgent_RDC.optimization_routine(optimization_params_, P_Akn1, P_AgW_Yc, y_idxs_for_exploration, possible_BhIdx_given_Yc_vecs(:,BhIdx_given_Yc_vec_idx)); %#ok<*FVAL>
                    end

                    MeanAdversarialRewardEstimates = [experiences.MeanAdversarialRewardEstimate];
                    valid_strategies_flag = ~isinf(MeanAdversarialRewardEstimates);
                    if any(valid_strategies_flag)
                        experience_orig.MeanAdversarialRewardEstimate = SmartGridUserEnv_RD.computeMeanAdversarialRewardEstimate(params, experience_orig.P_Bks, experience_orig.Hhk_idxs, experience_orig.P_YksgY12kn1);
                        experience_orig = rmfield(experience_orig, 'Hhk_idxs');
                        experiences = [experiences(valid_strategies_flag) experience_orig];
                        q_values = DeterministicActorCriticAgent_RDC.computeCriticTargets_with_exp(agent.Critic, agent.Actor, experiences, params);
                        [~,valid_strat_idx] = max(q_values);
                        [noisy_P_Wk] = experiences(valid_strat_idx).P_Wk;
                    end
                end
            end
        end
    end

    %======================================================================
    % Learning methods
    %======================================================================
    methods (Access = protected)
        function [agent,Data] = learn_(agent,Exp, Data)
            % store experiences
            appendWithoutSampleValidation(agent.ReplayBuffer,Exp);

            if agent.AgentOptions.InMemoryUpdateInterval == 0 || ...
                    mod(Exp.step_index, round(agent.AgentOptions.InMemoryUpdateInterval*agent.Params.k_num)) == 0
                [experiences, ~] = sample(agent.ReplayBuffer,...
                    agent.AgentOptions.MiniBatchSize);
                if ~isempty(experiences)
                    [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences);
                    if isempty(Data.CumulativeLoss)
                        Data.CumulativeLoss = [actorLoss;criticLoss];
                    else
                        Data.CumulativeLoss = Data.CumulativeLoss + [actorLoss;criticLoss];
                    end
                    Data.LearntEventsCount = Data.LearntEventsCount + 1;
                end
            end
            if Exp.IsDone && Data.LearntEventsCount == 0
                [experiences, ~] = sample(agent.ReplayBuffer,...
                    agent.ReplayBuffer.Length);
                [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences);
                if isempty(Data.CumulativeLoss)
                    Data.CumulativeLoss = [actorLoss;criticLoss];
                else
                    Data.CumulativeLoss = Data.CumulativeLoss + [actorLoss;criticLoss];
                end
                Data.LearntEventsCount = Data.LearntEventsCount + 1;
            end
            agent.trained_episodes = agent.trained_episodes + uint64(Exp.IsDone);
        end

        function [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences)
            BatchSize = length(experiences);
            b_num = agent.Params.b_num;
            Observations = reshape(cell2mat([experiences.Observation]),[b_num,1,BatchSize]);
            Actions = reshape(cell2mat([experiences.Action]),[agent.Params.subpolicy_params_num_con,1,BatchSize]);
            [criticTargets] = agent.computeCriticTargets(agent.Critic, agent.Actor, experiences, agent.Params);
            criticGradInput.Target = criticTargets;

            [criticGradient, gradInfo] = gradient(agent.Critic, @DeterministicActorCriticAgent_RDC.criticLossFn,...
                [{Observations}, {Actions}],criticGradInput);
            [Critic_new, agent.CriticOptimizer] = update(agent.CriticOptimizer,agent.Critic,...
                criticGradient);
            criticLoss = rl.logging.internal.util.extractLoss(gradInfo);

            [actorGradient, actorLoss]  = agent.computeActorGradients(agent.Critic, agent.Actor,...
                {Observations});
            [Actor_new, agent.ActorOptimizer] = update(agent.ActorOptimizer,agent.Actor,...
                actorGradient);

            agent.Critic = syncParameters(agent.Critic,Critic_new,agent.Params.TargetSmoothFactor_C);
            agent.Actor = syncParameters(agent.Actor,Actor_new,agent.Params.TargetSmoothFactor_Ac);

            if isa(actorLoss,"gpuArray")
                actorLoss = gather(actorLoss);
            end
            if isa(criticLoss,"gpuArray")
                criticLoss = gather(criticLoss);
            end
        end
    end

    methods (Static)
        function [exp] = optimization_routine(params_,P_Akn1,P_AgW_Yc_in, y_idxs_for_exploration, BhIdx_given_Ye)
            if isa(P_AgW_Yc_in,'parallel.pool.Constant')
                P_AgW_Yc = P_AgW_Yc_in.Value;
            else
                P_AgW_Yc = P_AgW_Yc_in;
                clear P_AgW_Yc_in
            end
            P_AgW_Ye = P_AgW_Yc(y_idxs_for_exploration);
            gurobi_model_ = params_.gurobi_model;
            gurobi_model_params_ = params_.gurobi_model_params;
            Aeq_cons_ = params_.Aeq_cons;
            beq_cons_ = params_.beq_cons;
            eq_cons_num_ = params_.eq_cons_num;
            w_num = params_.w_num;
            C_HgHh_adv_ = params_.C_HgHh_design;
            paramsPrecision_ = params_.paramsPrecision;
            function_handles = params_.Function_handles;
            minLikelihoodFilter_ = params_.minLikelihoodFilter;
            P_HgB_ = params_.P_HgB;
            P_BgA_ = params_.P_BgA;
            y_num_for_exploration = length(BhIdx_given_Ye);
            B2HL=function_handles.B2HL;
            b_nn_cells = params_.b_nn_cells;

            P_Wk = [];
            MeanAdversarialRewardEstimate_ = -inf;
            P_Bks = [];
            P_YksgY12kn1 = [];

            Aineq_cons_ = sparse([]);
            bineq_cons_ = [];
            alpha_vector_ = sparse(1,w_num);
            for Ye_idx_ = 1:y_num_for_exploration
                Bhk_idx_ = BhIdx_given_Ye(Ye_idx_);
                [Hhk_idx, ~] = B2HL(Bhk_idx_);
                P_BgW_ = sparse(P_BgA_*P_AgW_Ye{Ye_idx_});
                P_YcgW_ = sum(P_BgW_,1);

                Aineq_cons_HhIdx_ = b_nn_cells(Bhk_idx_).A*P_BgW_ - b_nn_cells(Bhk_idx_).b*P_YcgW_;
                bineq_cons_HhIdx_ = -paramsPrecision_*ones(length(b_nn_cells(Bhk_idx_).b),1);

                Aineq_cons_ = [Aineq_cons_;Aineq_cons_HhIdx_;-P_YcgW_];
                bineq_cons_ = [bineq_cons_;bineq_cons_HhIdx_;-(minLikelihoodFilter_+paramsPrecision_)]; %#ok<*AGROW>
                alpha_vector_ = alpha_vector_ + C_HgHh_adv_(Hhk_idx,:)*P_HgB_*P_BgW_;
            end

            gurobi_model_.A = [Aeq_cons_;Aineq_cons_];
            gurobi_model_.rhs   = [beq_cons_;bineq_cons_];
            gurobi_model_.sense =  [repmat('=',[1,eq_cons_num_]),repmat('<',[1,length(bineq_cons_)])];
            gurobi_model_.obj  = full(alpha_vector_);
            gurobi_result_ = gurobi(gurobi_model_, gurobi_model_params_);
            if strcmp(gurobi_result_.status, 'OPTIMAL')
                P_Wk = sparse(gurobi_result_.x);
                belief_trans_info.P_AgW_Yc = P_AgW_Yc;
                [P_Aks, Hhk_idxs, P_YksgY12kn1] = SmartGridUserEnv_RD.get_possible_belief_transitions(params_, P_Akn1, P_Wk, belief_trans_info, []);
                y_control_num = params_.y_control_num;
                P_Bks = zeros(params_.b_num,1,y_control_num);
                for y_idx = 1:y_control_num
                    P_Bks(:,1,y_idx) = params_.P_BgA*P_Aks{y_idx};
                end

                MeanAdversarialRewardEstimate_ = SmartGridUserEnv_RD.computeMeanAdversarialRewardEstimate(params_, P_Bks, Hhk_idxs, P_YksgY12kn1);
            end

            exp = struct;
            exp.P_Wk = P_Wk;
            exp.P_Bks = P_Bks;
            exp.P_YksgY12kn1 = P_YksgY12kn1;
            exp.MeanAdversarialRewardEstimate = MeanAdversarialRewardEstimate_;
        end
    
        function [criticTargets] = computeCriticTargets(critic, actor, experiences, params)
            b_num = params.b_num;
            BatchSize = length(experiences);

            rewards = [experiences.AdversarialRewardEstimate];
            NextObservations = reshape(cell2mat([experiences.NextObservation]),[b_num,1,BatchSize]);
            NextActions = getAction(actor,{NextObservations});
            NextQValues = getValue(critic, {NextObservations}, NextActions);
            criticTargets = rewards + params.discountFactor*NextQValues;
        end

        function [criticTargets] = computeCriticTargets_with_exp(critic, actor, experiences, params)
            discountFactor = params.discountFactor;
            BatchSize = length(experiences);

            rewards = [experiences.MeanAdversarialRewardEstimate];
            P_YksgY12kn1 = [experiences.P_YksgY12kn1];

            NextObservations = {experiences.P_Bks};
            NextObservations = cat(3,NextObservations{:});
            NextActions = getAction(actor,{NextObservations});
            NextQValues = reshape(getValue(critic, {NextObservations}, NextActions),params.y_control_num,BatchSize);
            criticTargets = rewards + discountFactor*sum(NextQValues.*P_YksgY12kn1,1);
        end

        function Loss = criticLossFn(ModelOutput,GradInput)
            Loss = mse(ModelOutput{1}, reshape(GradInput.Target,size(ModelOutput{1})));
        end

        function [actorGradient, actorLoss] = computeActorGradients(critic, actor, Observation)
            % Static method to computeGradients outside of the agent (e.g.
            % on a worker)
            gradFcn = @DeterministicActorCriticAgent_RDC.actorGradientFn;
            actorGradInput.BatchSize = size(Observation{1},3);

            if rl.util.rlfeature("BuiltinGradientAcceleration")
                % if accelerated function is not available, use dlnetwork
                % grad input
                % if accelerated function is available, use learnables and
                % states
                fcn = getAcceleratedGradientFcn(getInternalModel(actor),"custom",actorGradInput.BatchSize,1,gradFcn);
                gradUseLearnable = ~isempty(fcn);
                if gradUseLearnable
                    for ct = 1:numel(critic)
                        actorGradInput.("Critic"+string(ct)) = getLearnableParameters(critic(ct));
                    end
                else
                    actorGradInput.Critic = getModel(critic);
                end
            else
                actorGradInput.Critic = getModel(critic);
            end
            [actorGradient, gradInfo] = customGradient(actor, gradFcn, Observation{1}, actorGradInput);
            actorLoss = rl.logging.internal.util.extractLoss(gradInfo);
        end

        function [GradVal, QVal] = actorGradientFn(ActorDLNet, Observation, GradInput)
            Action = forward(ActorDLNet, Observation{1});
            QVal = predict(GradInput.Critic, Observation{1}, Action);
            QVal = -sum(QVal,'all')/GradInput.BatchSize;
            GradVal = dlgradient(QVal, ActorDLNet.Learnables);
        end    

        function [params] = getActionParamsInfo(params)
            x_num = params.x_num;
            l_num = params.l_num;
            w_num = params.w_num;
            valid_YgXLn1 = params.valid_YgXLn1;
            h_num  = params.h_num;

            function_handles = params.Function_handles;
            XB_2T = function_handles.XB_2T;
            YsT_2W = function_handles.YsT_2W;
            HL2B = function_handles.HL2B;

            paramIdx2W_flag = false(w_num,1);
            for l_kn1_idx = 1:l_num
                for x_idx = 1:x_num
                    valid_Yidxs = valid_YgXLn1{x_idx, l_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num>1
                        for h_k_idx = 1:h_num
                            paramIdx2W_flag(YsT_2W(valid_Yidxs(1:(end-1)),XB_2T(x_idx, HL2B(h_k_idx, l_kn1_idx)))) = true;
                        end
                    end
                end
            end
            paramIdx2W = find(paramIdx2W_flag);
            subpolicy_params_num_con = length(paramIdx2W);
            W2paramIdx = zeros(w_num,1);
            W2paramIdx(paramIdx2W) = 1:subpolicy_params_num_con;
            params.W2paramIdx = W2paramIdx;
            params.subpolicy_params_num_con = subpolicy_params_num_con;   
        end

        function [params] = preprocessParamsForTraining(params)
            t_num = params.t_num;
            h_num = params.h_num;
            l_num = params.l_num;
            w_num = params.w_num;
            x_num = params.x_num;
            y_control_num  = params.y_control_num;
            valid_YgXLn1 = params.valid_YgXLn1;

            h_range = 1:h_num;

            t_range = 1:t_num;
            y_control_range = 1:y_control_num;

            YsT_2W = params.Function_handles.YsT_2W;
            HsL2B= params.Function_handles.HsL2B;
            XBs_2T= params.Function_handles.XBs_2T;
            YTs_2W= params.Function_handles.YTs_2W;

            gurobi_model.modelsense = 'min';
            gurobi_model.vtype = repmat('C', w_num, 1);
            gurobi_model.lb    = zeros(w_num, 1);
            gurobi_model.ub   = ones(w_num, 1);
            gurobi_model_params.outputflag = 0;

            Aeq_cons = zeros(t_num,w_num);
            for t_k_idx = t_range
                Aeq_cons(t_k_idx,YsT_2W(y_control_range,t_k_idx)) = 1;
            end
            beq_cons = ones(t_num,1);

            Aeq_cons_2 = zeros(1,w_num);
            for l_kn1_idx = 1:l_num
                b_kn1_idxs = HsL2B(h_range,l_kn1_idx);
                for x_idx = 1:x_num
                    invalid_y_idxs = setdiff(y_control_range, valid_YgXLn1{x_idx, l_kn1_idx});
                    for yc_idx = invalid_y_idxs
                        t_k_idxs = XBs_2T(x_idx,b_kn1_idxs');
                        Aeq_cons_2(YTs_2W(yc_idx,t_k_idxs)) = 1;
                    end
                end
            end

            if(any(Aeq_cons_2>0))
                Aeq_cons = [Aeq_cons;Aeq_cons_2];
                beq_cons = [beq_cons;0];
            end

            Aeq_cons = sparse(Aeq_cons);
            gurobi_model.A = Aeq_cons;
            gurobi_model.rhs  = beq_cons;
            gurobi_model.sense = repmat('=',[1,length(beq_cons)]);

            params.gurobi_model = gurobi_model;
            params.gurobi_model_params = gurobi_model_params;
            params.Aeq_cons = Aeq_cons;
            params.beq_cons = beq_cons;
            params.eq_cons_num = length(beq_cons);
        end

        function [P_Wks] = conAction2SubPolicy(params, actions, ~)
            x_num = params.x_num;
            l_num = params.l_num;
            h_num = params.h_num;
            valid_YgXLn1 = params.valid_YgXLn1;
            function_handles = params.Function_handles;
            XB_2T = function_handles.XB_2T;
            YsT_2W = function_handles.YsT_2W;
            HL2B = function_handles.HL2B;

            w_num = params.w_num;
            num_actions = size(actions,2);
            P_Wks = zeros(w_num,num_actions);
            for l_kn1_idx = 1:l_num
                for x_idx = 1:x_num
                    valid_Yidxs = valid_YgXLn1{x_idx, l_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num>1
                        for h_k_idx = 1:h_num
                            WIdxs = YsT_2W(valid_Yidxs,XB_2T(x_idx, HL2B(h_k_idx, l_kn1_idx)));

                            param_idxs = params.W2paramIdx(WIdxs(1:(end-1)));
                            P_Wks(WIdxs,:) = DeterministicActorCriticAgent_RDC.nnet_params_to_simplex_transform(actions(param_idxs,:));
                        end
                    else
                        for h_k_idx = 1:h_num
                            WIdxs = YsT_2W(valid_Yidxs,XB_2T(x_idx, HL2B(h_k_idx, l_kn1_idx)));
                            P_Wks(WIdxs,:) = 1;
                        end
                    end
                end
            end
            P_Wks = sparse(P_Wks);
        end

        function [actions] = conSubPolicy2Action(params, P_Wks)
            x_num = params.x_num;
            l_num = params.l_num;
            h_num = params.h_num;
            valid_YgXLn1 = params.valid_YgXLn1;
            function_handles = params.Function_handles;
            XB_2T = function_handles.XB_2T;
            YsT_2W = function_handles.YsT_2W;
            HL2B = function_handles.HL2B;
            logistic_param_limit = params.logistic_param_limit;
            paramsPrecision = params.paramsPrecision;

            subpolicy_params_num_con = params.subpolicy_params_num_con;
            num_actions = size(P_Wks,2);
            actions = zeros(subpolicy_params_num_con,num_actions);
            for l_kn1_idx = 1:l_num
                for x_idx = 1:x_num
                    valid_Yidxs = valid_YgXLn1{x_idx, l_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num>1
                        for h_k_idx = 1:h_num
                            WIdxs = YsT_2W(valid_Yidxs,XB_2T(x_idx, HL2B(h_k_idx, l_kn1_idx)));

                            param_idxs = params.W2paramIdx(WIdxs(1:(end-1)));
                            actions(param_idxs, :) = DeterministicActorCriticAgent_RDC.simplex_to_nnet_params_transform(P_Wks(WIdxs,:), logistic_param_limit, paramsPrecision);
                        end
                    end
                end
            end
        end

        function [X_12D_vecs] = nnet_params_to_simplex_transform(Y_12Dn1_vecs)
            [D_num, num_actions] = size(Y_12Dn1_vecs);
            D_num = D_num + 1;
            X_12D_vecs = zeros(D_num, num_actions);
            for action_idx = 1:num_actions
                X_12Dn1_vec = exp(Y_12Dn1_vecs(:,action_idx));
                X_12D_vec = [X_12Dn1_vec;1];
                sum_X_12D_vec = sum(X_12D_vec);
                X_12D_vecs(:,action_idx) = X_12D_vec/sum_X_12D_vec;
            end
        end

        function [Y_12Dn1_vecs] = simplex_to_nnet_params_transform(X_12D_vecs, params_limit, paramsPrecision)
            [D_num, num_actions] = size(X_12D_vecs);
            Y_12Dn1_vecs = zeros(D_num-1, num_actions);
            for action_idx = 1:num_actions
                X_12D_vecs(end,action_idx) = max(paramsPrecision, X_12D_vecs(end,action_idx));
                X_12D_vecs(:,action_idx) = X_12D_vecs(:,action_idx)/sum(X_12D_vecs(:,action_idx));
                Y_12Dn1_vecs(:,action_idx) = max(-params_limit, min(params_limit, log(X_12D_vecs(1:D_num-1,action_idx)/X_12D_vecs(end,action_idx))));
            end
        end

    end

    %======================================================================
    % save/load
    %======================================================================
    methods
        function s = saveobj(agent)
            s = struct;
            s.Actor = agent.Actor;
            s.Critic = agent.Critic;
            s.Actor.UseDevice = 'cpu';
            s.Critic.UseDevice = 'cpu';

            s.ActorOptimizer = agent.ActorOptimizer;
            s.CriticOptimizer = agent.CriticOptimizer;

            s.AgentOptions = agent.AgentOptions;
            s.Params = agent.Params;
            s.trained_episodes = agent.trained_episodes;
            s.ReplayBuffer = agent.ReplayBuffer;
            s.noiseModel = agent.noiseModel;
        end
    end
    
    methods (Static)
        function agent = loadobj(s)
            if isfield(s, 'ReplayBuffer')
                ReplayBuffer = s.ReplayBuffer;
            else
                ReplayBuffer = [];
            end
            agent = DeterministicActorCriticAgent_RDC(s.Actor, s.Critic, s.AgentOptions, s.Params, ReplayBuffer, s.trained_episodes, s.noiseModel);
            agent.CriticOptimizer = s.CriticOptimizer;
            agent.ActorOptimizer = s.ActorOptimizer;
        end
    end
end