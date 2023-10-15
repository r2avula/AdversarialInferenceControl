classdef DeterministicActorCriticAgent < agents.AbstractActorCriticAgent
    %% Policy aware adversary model
    properties
        Params

        % Training utils
        ReplayBuffer
        trained_episodes
        optimization_params
        P_AgU_YcAkn1
        useparpool = false
    end

    methods
        function agent = DeterministicActorCriticAgent(actor, critic, agentOpts, params, ReplayBuffer, trained_episodes)
            if  nargin == 4
                ReplayBuffer = [];
                trained_episodes = 0;
            end
            agent = agent@agents.AbstractActorCriticAgent(actor, critic, agentOpts);
            agent.Params = params;
            if isempty(ReplayBuffer)
                ReplayBuffer = replay.RLReplayMemory(agent.ObservationInfo,agent.ActionInfo,agent.AgentOptions.ExperienceBufferLength);
            end
            agent.ReplayBuffer = ReplayBuffer;
            agent.trained_episodes = trained_episodes;
        end

        function q0 = evaluateQ0(agent, Observation)
            arguments
                agent (1,1)
                Observation cell
            end
            action = getAction(agent.Actor, {Observation{1}(1:agent.Params.a_num)});
            q0 = getValue(agent.Critic, Observation, action)*(1-agent.Params.discountFactor);
        end

        function [action, actionInfo] = getAction(agent, Observation)
            params = agent.Params;
            P_Akn1 = Observation(1:agent.Params.a_num);
            exploration_epsilon = params.exploration_epsilon;
            noise_epsilon = params.noise_epsilon;

            [P_AgU_Yc] = SmartGridUserEnv_FD.preprocess_possible_belief_transitions(params, agent.P_AgU_YcAkn1, P_Akn1);
            belief_trans_info.P_AgU_Yc = P_AgU_Yc;
            [action, actionInfo] = getAction(agent.Actor, {P_Akn1});
            P_Uk = agent.conAction2SubPolicy(params,action{1});
            rand_draw = rand();
            if rand_draw < exploration_epsilon
                [P_Uk_exploratory, belief_trans_info] = agent.getExploratoryStrategy(P_Akn1, P_AgU_Yc, P_Uk); 
                action = agent.conSubPolicy2Action(params,P_Uk_exploratory);
                P_Uk = P_Uk_exploratory;
            elseif rand_draw < exploration_epsilon + noise_epsilon
                    valid_YgXZn1 = params.valid_YgXZn1;
                    function_handles = params.Function_handles;
                    S2XHAn1 = function_handles.S2XHAn1;
                    A2HZ = function_handles.A2HZ;
                    YcsS_2U = function_handles.YcsS_2U;
                    P_Uk_noisy = P_Uk;
                    rand_s_idxs = randperm(params.s_num,randi(params.s_num));
                    for s_idx = rand_s_idxs
                        [x_k_idx, ~, a_kn1_idx] = S2XHAn1(s_idx);
                        [~,z_kn1_idx] = A2HZ(a_kn1_idx);
                        valid_Yidxs = valid_YgXZn1{x_k_idx, z_kn1_idx};
                        valid_YcIdxs_num = length(valid_Yidxs);
                        if valid_YcIdxs_num>1
                            U_idxs = YcsS_2U(1:params.y_control_num, s_idx);
                            P_Y = P_Uk_noisy(U_idxs);
                            rand_y_idxs = valid_Yidxs(randperm(valid_YcIdxs_num,randi(valid_YcIdxs_num)));
                            P_Y(rand_y_idxs) = min(max(P_Y(rand_y_idxs) + rand(length(rand_y_idxs),1)*params.noise_sd, 0),1);
                            if sum(P_Y) > 0
                                P_Y = P_Y./sum(P_Y);
                                P_Uk_noisy(U_idxs) = P_Y;
                            end
                        end
                    end
                    action = agent.conSubPolicy2Action(params,P_Uk_noisy);
                    P_Uk = P_Uk_noisy;
            end
            action = rl.util.cellify(action);
            action = {single(action{1})};
            actionInfo.P_Uk = P_Uk;     
            actionInfo.belief_trans_info = belief_trans_info; 
        end

        function [P_Uk,belief_trans_info] = getExploratoryStrategy(agent, P_Akn1, P_AgU_Yc, P_Uk_nnet)
            params = agent.Params;
            y_num_for_exploration = params.y_num_for_exploration;
            gurobi_model = params.gurobi_model;
            gurobi_model_params = params.gurobi_model_params;
            Aeq_cons = params.Aeq_cons;
            beq_cons = params.beq_cons;
            eq_cons_num = params.eq_cons_num;
            paramsPrecision = params.paramsPrecision;
            h_num = params.h_num;
            a_num = params.a_num;
            minLikelihoodFilter = params.minLikelihoodFilter;
            num_rand_adv_strats_for_exploration = params.num_rand_adv_strats_for_exploration;
            function_handles = params.Function_handles;
            P_HgA = (params.P_HgA);
            params.useparpool = agent.useparpool;

            belief_trans_info.P_AgU_Yc = P_AgU_Yc;
            [P_Aks_nnet, Hhk_idxs_nnet, P_YksgY12kn1_nnet] = SmartGridUserEnv_FD.get_possible_belief_transitions(params, P_Akn1, P_Uk_nnet, belief_trans_info, []);
            experience_nnet = struct;
            experience_nnet.P_Uk = P_Uk_nnet;
            experience_nnet.P_Aks = P_Aks_nnet;
            experience_nnet.Hhk_idxs = Hhk_idxs_nnet;
            experience_nnet.P_YksgY12kn1 = P_YksgY12kn1_nnet;
            experience_nnet.MeanAdversarialRewardEstimate = sum(SmartGridUserEnv_FD.computeAdversarialRewardEstimates(params, experience_nnet).*...
                                experience_nnet.P_YksgY12kn1,1);

            experiences = experience_nnet;

            if isempty(agent.optimization_params)
                optimization_params_ = struct;
                optimization_params_.gurobi_model = params.gurobi_model;
                optimization_params_.gurobi_model_params = params.gurobi_model_params;
                optimization_params_.Aeq_cons = params.Aeq_cons;
                optimization_params_.beq_cons = params.beq_cons;
                optimization_params_.eq_cons_num = params.eq_cons_num;
                optimization_params_.DRs_in_Rn = params.DRs_in_Rn;
                optimization_params_.with_a_nncells = params.with_a_nncells;
                optimization_params_.u_num = params.u_num;
                optimization_params_.C_HgHh_design = (params.C_HgHh_design);
                optimization_params_.paramsPrecision = params.paramsPrecision;
                optimization_params_.minLikelihoodFilter = params.minLikelihoodFilter;
                optimization_params_.P_HgA = (params.P_HgA);
                optimization_params_.beliefSpacePrecision_adv = params.beliefSpacePrecision_adv;
                optimization_params_.DRs_in_Rn = params.DRs_in_Rn;
                optimization_params_.y_control_num = params.y_control_num;
                optimization_params_.a_num = params.a_num;
                optimization_params_.discountFactor = params.discountFactor;
                optimization_params_.Function_handles = function_handles;
                optimization_params_.useparpool = agent.useparpool;
                if params.with_a_nncells
                    optimization_params_.nn_cells = get_NN_ClassifyingConstraints(eye(params.a_num),1);
                else
                    optimization_params_.nn_cells = get_NN_ClassifyingConstraints(eye(params.h_num),1);
                end
                optimization_params_.U2paramIdx = params.U2paramIdx;
                agent.optimization_params = optimization_params_;
            else
                optimization_params_ = agent.optimization_params;
            end

            [B,I] = sort(experience_nnet.P_YksgY12kn1,'descend');
            filter_flag = B>=minLikelihoodFilter;
            if any(filter_flag)
                y_idxs_for_exploration = I(filter_flag);
                y_idxs_for_exploration = y_idxs_for_exploration(:)';
                y_num_for_exploration = min(length(y_idxs_for_exploration), y_num_for_exploration);
                y_idxs_for_exploration = y_idxs_for_exploration(1:y_num_for_exploration);

                nn_cells = optimization_params_.nn_cells;
                if params.with_a_nncells
                    possible_HhoAhIdxk_g_Yck_flag = false(a_num,y_num_for_exploration);
                    aoh_num = a_num;
                else
                    possible_HhoAhIdxk_g_Yck_flag = false(h_num,y_num_for_exploration);
                    aoh_num = h_num;
                end
                for Ye_idx_t = 1:y_num_for_exploration
                    Yck_idx = y_idxs_for_exploration(Ye_idx_t);
                    P_AgU = P_AgU_Yc{Yck_idx};
                    P_YcgU = sum(P_AgU,1);
                    P_YcgU_full = full(P_YcgU);
                    if params.with_a_nncells
                        P_AoHgU = P_AgU;
                    else
                        P_AoHgU = P_HgA*P_AgU;
                    end

                    gurobi_model_t = gurobi_model;
                    gurobi_model_t.obj  = -P_YcgU_full;
                    gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
                    if strcmp(gurobi_result_t.status, 'OPTIMAL') && -gurobi_result_t.objval >= minLikelihoodFilter
                        for AoHhk_idx = 1:aoh_num
                            Aineq_cons_AhIdx = nn_cells(AoHhk_idx).A*P_AoHgU - nn_cells(AoHhk_idx).b*P_YcgU;
                            bineq_cons_AhIdx = -paramsPrecision*ones(length(nn_cells(AoHhk_idx).b),1);

                            Aineq_cons_t = [Aineq_cons_AhIdx;-P_YcgU];
                            bineq_cons_t = [bineq_cons_AhIdx;-(minLikelihoodFilter+paramsPrecision)];

                            gurobi_model_t.A = [Aeq_cons;Aineq_cons_t];
                            gurobi_model_t.rhs   = [beq_cons;bineq_cons_t];
                            gurobi_model_t.sense =  [repmat('=',[1,eq_cons_num]),repmat('<',[1,length(bineq_cons_t)])];
                            gurobi_result_t = gurobi(gurobi_model_t, gurobi_model_params);
                            if(strcmp(gurobi_result_t.status, 'OPTIMAL'))
                                possible_HhoAhIdxk_g_Yck_flag(AoHhk_idx,Ye_idx_t) = true;
                            end
                        end
                    end
                end
                
                if all(any(possible_HhoAhIdxk_g_Yck_flag,1))
                    possible_AoHhIdx_given_Yc_vecs = find(possible_HhoAhIdxk_g_Yck_flag(:,1))';
                    for ye_idx = 2:y_num_for_exploration
                        temp_idxs = find(possible_HhoAhIdxk_g_Yck_flag(:,ye_idx))';
                        possible_AoHhIdx_given_Yc_vecs = combvec(possible_AoHhIdx_given_Yc_vecs,temp_idxs);

                        num_possible_HhIdx_given_Yc_vecs = size(possible_AoHhIdx_given_Yc_vecs,2);

                        if num_possible_HhIdx_given_Yc_vecs > num_rand_adv_strats_for_exploration
                            strat_idxs_to_search = randperm(num_possible_HhIdx_given_Yc_vecs,num_rand_adv_strats_for_exploration);
                            possible_AoHhIdx_given_Yc_vecs = possible_AoHhIdx_given_Yc_vecs(:, strat_idxs_to_search);
                        end
                    end
                    num_possible_HhIdx_given_Yc_vecs = size(possible_AoHhIdx_given_Yc_vecs,2);

                    % start_t = tic;
                    if agent.useparpool
                        P_AgU_Yc = parallel.pool.Constant(P_AgU_Yc);
                        parfor HhIdx_given_Yc_vec_idx = 1:num_possible_HhIdx_given_Yc_vecs
                            experience_ =...
                                DeterministicActorCriticAgent.optimization_routine(optimization_params_, P_Akn1, P_AgU_Yc, y_idxs_for_exploration, possible_AoHhIdx_given_Yc_vecs(:,HhIdx_given_Yc_vec_idx)); %#ok<*FVAL>
                            if ~isempty(experience_.P_Uk)
                                experience_.MeanAdversarialRewardEstimate = ...
                                    sum(SmartGridUserEnv_FD.computeAdversarialRewardEstimates(optimization_params_, experience_).*...
                                    experience_.P_YksgY12kn1,1);
                            end
                            experiences(1+HhIdx_given_Yc_vec_idx) = experience_;
                        end
                    else
                        for HhIdx_given_Yc_vec_idx = 1:num_possible_HhIdx_given_Yc_vecs
                            experience_ =...
                                DeterministicActorCriticAgent.optimization_routine(optimization_params_, P_Akn1, P_AgU_Yc, y_idxs_for_exploration, possible_AoHhIdx_given_Yc_vecs(:,HhIdx_given_Yc_vec_idx)); %#ok<*FVAL>
                            if ~isempty(experience_.P_Uk)
                                experience_.MeanAdversarialRewardEstimate = ...
                                    sum(SmartGridUserEnv_FD.computeAdversarialRewardEstimates(optimization_params_, experience_).*...
                                    experience_.P_YksgY12kn1,1);
                            end
                            experiences(1+HhIdx_given_Yc_vec_idx) = experience_;
                        end
                    end
                    % fprintf("\n%f\n",toc(start_t));

                    MeanAdversarialRewardEstimates = [experiences.MeanAdversarialRewardEstimate];
                    valid_strategies_flag = ~isinf(MeanAdversarialRewardEstimates);
                    if any(valid_strategies_flag)
                        experiences = experiences(valid_strategies_flag);
                        % [~,valid_strat_idx] = max([experiences.MeanAdversarialRewardEstimate]);
                        q_values = DeterministicActorCriticAgent.computeCriticTargets_with_exp(agent.Critic, agent.Actor, experiences, params);
                        [~,valid_strat_idx] = max(q_values);
                        [P_Uk] = experiences(valid_strat_idx).P_Uk;
                        belief_trans_info.P_Aks = experiences(valid_strat_idx).P_Aks;
                        belief_trans_info.Hhk_idxs = experiences(valid_strat_idx).Hhk_idxs;
                        belief_trans_info.P_YksgY12kn1 = experiences(valid_strat_idx).P_YksgY12kn1;
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
            [experiences, ~] = sample(agent.ReplayBuffer, agent.AgentOptions.MiniBatchSize);
            if ~isempty(experiences)
                [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences);
                if isempty(Data.CumulativeLoss)
                    Data.CumulativeLoss = [actorLoss;criticLoss];
                else
                    Data.CumulativeLoss = Data.CumulativeLoss + [actorLoss;criticLoss];
                end
                Data.LearntEventsCount = Data.LearntEventsCount + 1;
            end

            if Exp.IsDone
                if Data.LearntEventsCount == 0
                    [experiences, ~] = sample(agent.ReplayBuffer, agent.ReplayBuffer.Length);
                    [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences);
                    if isempty(Data.CumulativeLoss)
                        Data.CumulativeLoss = [actorLoss;criticLoss];
                    else
                        Data.CumulativeLoss = Data.CumulativeLoss + [actorLoss;criticLoss];
                    end
                    Data.LearntEventsCount = Data.LearntEventsCount + 1;
                end
                agent.trained_episodes = agent.trained_episodes + 1;
            end
        end

        function [agent,actorLoss,criticLoss] = learnFromExperiences_(agent, experiences)
            BatchSize = length(experiences);
            Observations = reshape(cell2mat([experiences.Observation]),agent.Params.actor_nnet_intput_size,1,BatchSize);
            Actions = reshape(cell2mat([experiences.Action]),agent.Params.actor_nnet_output_size,1,BatchSize);

            %% Controller critic learning
            if agent.Params.with_mean_reward
                [criticTargets] = agent.computeCriticTargets_with_exp(agent.Critic, agent.Actor, experiences, agent.Params);
            else
                [criticTargets] = agent.computeControllerCriticTargets(agent.Critic, agent.Actor, experiences, agent.Params);
            end
            gradInput.Target = criticTargets;

            [gradientVal, gradInfo] = gradient(agent.Critic, @DeterministicActorCriticAgent.criticLossFn,...
                [{Observations}, {Actions}],gradInput);
            [Critic_new,agent.CriticOptimizer] = update(agent.CriticOptimizer,agent.Critic, gradientVal);
            criticLoss = rl.logging.internal.util.extractLoss(gradInfo);
            agent.Critic = syncParameters(agent.Critic,Critic_new,agent.Params.TargetSmoothFactor_C);

            %% Controller learning
            [gradientVal, actorLoss]  = agent.computeActorGradients(agent.Critic, agent.Actor, {Observations});
            [nnet_new,agent.ActorOptimizer] = update(agent.ActorOptimizer,agent.Actor, gradientVal);
            agent.Actor = syncParameters(agent.Actor,nnet_new,agent.Params.TargetSmoothFactor_Ac);

            if isa(actorLoss,"gpuArray")
                actorLoss = gather(actorLoss);
            end
            if isa(criticLoss,"gpuArray")
                criticLoss = gather(criticLoss);
            end
        end
    end

    methods (Static)
        function [experience] = optimization_routine(params_,P_Akn1,P_AgU_Yc_in, y_idxs_for_exploration, AoHhIdx_given_Ye)
            if isa(P_AgU_Yc_in,'parallel.pool.Constant')
                P_AgU_Yc = P_AgU_Yc_in.Value;
            else
                P_AgU_Yc = P_AgU_Yc_in;
                clear P_AgU_Yc_in
            end
            P_AgU_Ye = P_AgU_Yc(y_idxs_for_exploration);
            gurobi_model_ = params_.gurobi_model;
            gurobi_model_params_ = params_.gurobi_model_params;
            Aeq_cons_ = params_.Aeq_cons;
            beq_cons_ = params_.beq_cons;
            eq_cons_num_ = params_.eq_cons_num;
            nn_cells = params_.nn_cells;
            u_num = params_.u_num;
            P_HgA = params_.P_HgA;
            C_HgHh_adv_ = params_.C_HgHh_design;
            paramsPrecision_ = params_.paramsPrecision;
            minLikelihoodFilter_ = params_.minLikelihoodFilter;
            y_num_for_exploration = length(AoHhIdx_given_Ye);
            Function_handles = params_.Function_handles;
            A2HZ = Function_handles.A2HZ;

            experience = struct;
            experience.P_Uk = [];
            experience.P_Aks = [];
            experience.Hhk_idxs = [];
            experience.P_YksgY12kn1 = [];
            experience.MeanAdversarialRewardEstimate = -inf;


            Aineq_cons_ = sparse([]);
            bineq_cons_ = [];
            alpha_vector_ = zeros(1,u_num);
            for Ye_idx_ = 1:y_num_for_exploration
                AoHhk_idx = AoHhIdx_given_Ye(Ye_idx_);
                P_AgU_ = P_AgU_Ye{Ye_idx_};
                P_HgU_ = P_HgA*P_AgU_;
                P_YcgU_ = sum(P_HgU_,1);
                if params_.with_a_nncells
                    P_AoHgU = P_AgU_;
                    [Hhk_idx, ~] = A2HZ(AoHhk_idx);
                else
                    P_AoHgU = P_HgA*P_AgU_;
                    Hhk_idx = AoHhk_idx;
                end

                Aineq_cons_HhIdx_ = nn_cells(AoHhk_idx).A*P_AoHgU - nn_cells(AoHhk_idx).b*P_YcgU_;
                bineq_cons_HhIdx_ = -paramsPrecision_*ones(length(nn_cells(AoHhk_idx).b),1);

                Aineq_cons_ = [Aineq_cons_;Aineq_cons_HhIdx_;-P_YcgU_];
                bineq_cons_ = [bineq_cons_;bineq_cons_HhIdx_;-(minLikelihoodFilter_+paramsPrecision_)]; %#ok<*AGROW>
                alpha_vector_ = alpha_vector_ + C_HgHh_adv_(Hhk_idx,:)*P_HgU_;
            end

            gurobi_model_.A = [Aeq_cons_;Aineq_cons_];
            gurobi_model_.rhs   = [beq_cons_;bineq_cons_];
            gurobi_model_.sense =  [repmat('=',[1,eq_cons_num_]),repmat('<',[1,length(bineq_cons_)])];
            gurobi_model_.obj  = alpha_vector_;
            gurobi_result_ = gurobi(gurobi_model_, gurobi_model_params_);
            if strcmp(gurobi_result_.status, 'OPTIMAL')
                P_Uk = (gurobi_result_.x);
                belief_trans_info.P_AgU_Yc = P_AgU_Yc;
                [P_Aks, Hhk_idxs, P_YksgY12kn1] = SmartGridUserEnv_FD.get_possible_belief_transitions(params_, P_Akn1, P_Uk, belief_trans_info, []);

                experience.P_Uk = P_Uk;
                experience.P_Aks = P_Aks;
                experience.Hhk_idxs = Hhk_idxs;
                experience.P_YksgY12kn1 = P_YksgY12kn1;
            end
        end
        
        function [criticTargets] = computeCriticTargets_with_exp(critic, actor, experiences, params)
            discountFactor = params.discountFactor;
            BatchSize = length(experiences);
            y_control_num = params.y_control_num;
            if params.with_controller_reward
                criticTargets = [experiences.MeanReward];
                P_Yksg_ = [experiences.P_YksgSk];
            else
                criticTargets = [experiences.MeanAdversarialRewardEstimate];
                P_Yksg_ = [experiences.P_YksgY12kn1];
            end
            NextObservations = [experiences.P_Aks];
            NextObservations = cat(3, NextObservations{:});
            NextActions = getAction(actor,{NextObservations});
            NextQValues = reshape(getValue(critic, {NextObservations}, NextActions),y_control_num, BatchSize);
            criticTargets = criticTargets + discountFactor*sum(NextQValues.*P_Yksg_,1);
        end

        function [criticTargets] = computeControllerCriticTargets(critic, actor, experiences, params)
            discountFactor = params.discountFactor;
            BatchSize = length(experiences);
            if params.with_controller_reward
                criticTargets = [experiences.Reward];
            else
                criticTargets = [experiences.AdversarialRewardEstimate];
            end
            NextObservations = reshape(cell2mat([experiences.NextObservation]),params.actor_nnet_intput_size,1,BatchSize);
            NextActions = getAction(actor,{NextObservations});
            NextQValues = reshape(getValue(critic, {NextObservations}, NextActions),1, BatchSize);
            criticTargets = criticTargets + discountFactor*NextQValues;         
        end

        function Loss = criticLossFn(ModelOutput,GradInput)
            Loss = mse(ModelOutput{1}, reshape(GradInput.Target,size(ModelOutput{1})));
        end

        function [actorGradient, actorLoss] = computeActorGradients(critic, actor, Observation)
            % Static method to computeGradients outside of the agent (e.g.
            % on a worker)
            gradFcn = @DeterministicActorCriticAgent.actorGradientFn;
            actorGradInput.BatchSize = size(Observation{1},3);
            actorGradInput.Critic = getModel(critic);
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
            h_num = params.h_num;
            x_num = params.x_num;
            z_num = params.z_num;
            h_range = 1:h_num;
            u_num = params.u_num;
            function_handles = params.Function_handles;
            valid_YgXZn1 = params.valid_YgXZn1;

            HsZ2A = function_handles.HsZ2A;
            XHsAn1s_2S = function_handles.XHsAn1s_2S;
            YcSs_2U = function_handles.YcSs_2U;

            paramIdx2U_flag = false(u_num,1);
            for z_kn1_idx = 1:z_num
                a_kn1_idxs = HsZ2A(h_range,z_kn1_idx);
                for x_k_idx = 1:x_num
                    valid_Yidxs = valid_YgXZn1{x_k_idx, z_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num>1
                        s_k_idxs = XHsAn1s_2S(x_k_idx,h_range',a_kn1_idxs');
                        for reachable_yc_idx = valid_Yidxs(1:(end-1))
                            paramIdx2U_flag(YcSs_2U(reachable_yc_idx,s_k_idxs)) = true;
                        end
                    end
                end
            end
            paramIdx2U = find(paramIdx2U_flag);
            subpolicy_params_num_con = length(paramIdx2U);
            U2paramIdx = zeros(u_num,1);
            U2paramIdx(paramIdx2U) = 1:subpolicy_params_num_con;
            params.paramIdx2U = paramIdx2U;
            params.U2paramIdx = U2paramIdx;
            params.subpolicy_params_num_con = subpolicy_params_num_con;
        end

        function [params] = preprocessParamsForTraining(params)
            [gurobi_model,gurobi_model_params,Aeq_cons,beq_cons] = get_gurobi_model_FDC(params,params.Function_handles);

            gurobi_model.lb = gurobi_model.lb;
            params.gurobi_model = gurobi_model;
            params.gurobi_model_params = gurobi_model_params;
            params.Aeq_cons = sparse(Aeq_cons);
            params.beq_cons = beq_cons;
            params.eq_cons_num = length(beq_cons);
        end

        function [P_Uks] = conAction2SubPolicy(params, actions)
            u_num = params.u_num;
            y_control_num = params.y_control_num;
            valid_YgXZn1= params.valid_YgXZn1;
            y_control_range = 1:y_control_num;
            num_actions = size(actions,2);
            P_Uks = zeros(u_num,num_actions);

            function_handles = params.Function_handles;
            YcsS_2U = function_handles.YcsS_2U;

            z_num = params.z_num;
            x_num = params.x_num;
            h_num = params.h_num;
            HZ2A = function_handles.HZ2A;
            XHAn1_2S = function_handles.XHAn1_2S;
            for z_kn1_idx = 1:z_num
                for x_k_idx = 1:x_num
                    valid_Yidxs = valid_YgXZn1{x_k_idx, z_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num==1
                        for h_kn1_idx = 1:h_num
                            a_kn1_idx = HZ2A(h_kn1_idx,z_kn1_idx);
                            for h_k_idx = 1:h_num
                                s_k_idx = XHAn1_2S(x_k_idx,h_k_idx,a_kn1_idx);
                                P_Uks(YcsS_2U(valid_Yidxs,s_k_idx),:) = 1;
                            end
                        end
                    else
                        for h_kn1_idx = 1:h_num
                            a_kn1_idx = HZ2A(h_kn1_idx,z_kn1_idx);
                            for h_k_idx = 1:h_num
                                s_k_idx = XHAn1_2S(x_k_idx,h_k_idx,a_kn1_idx);
                                P_YckgSk_control = zeros(y_control_num,num_actions);
                                UIdxs = YcsS_2U(valid_Yidxs,s_k_idx);
                                param_idxs = params.U2paramIdx(UIdxs(1:(valid_YcIdxs_num-1)));
                                P_YckgSk_control(valid_Yidxs,:) = DeterministicActorCriticAgent.nnet_params_to_simplex_transform(actions(param_idxs,:));
                                P_Uks(YcsS_2U(y_control_range,s_k_idx),:) = P_YckgSk_control;
                            end
                        end
                    end
                end
            end
        end

        function [actions] = conSubPolicy2Action(params, P_Uks)
            z_num = params.z_num;
            x_num = params.x_num;
            h_num = params.h_num;
            valid_YgXZn1= params.valid_YgXZn1;

            function_handles = params.Function_handles;
            HZ2A = function_handles.HZ2A;
            XHAn1_2S = function_handles.XHAn1_2S;
            YcsS_2U = function_handles.YcsS_2U;

            logistic_param_limit = params.logistic_param_limit;
            paramsPrecision = params.paramsPrecision;

            subpolicy_params_num_con = params.subpolicy_params_num_con;
            num_actions = size(P_Uks,2);
            actions = zeros(subpolicy_params_num_con,num_actions);
            for z_kn1_idx = 1:z_num
                for x_k_idx = 1:x_num
                    valid_Yidxs = valid_YgXZn1{x_k_idx, z_kn1_idx};
                    valid_YcIdxs_num = length(valid_Yidxs);
                    if valid_YcIdxs_num>1
                        for h_kn1_idx = 1:h_num
                            a_kn1_idx = HZ2A(h_kn1_idx,z_kn1_idx);
                            for h_k_idx = 1:h_num
                                s_k_idx = XHAn1_2S(x_k_idx,h_k_idx,a_kn1_idx);
                                UIdxs = YcsS_2U(valid_Yidxs,s_k_idx);
                                param_idxs = params.U2paramIdx(UIdxs(1:(valid_YcIdxs_num-1)));
                                actions(param_idxs, :) = DeterministicActorCriticAgent.simplex_to_nnet_params_transform(P_Uks(UIdxs,:), logistic_param_limit, paramsPrecision);
                            end
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
                Y_12Dn1_vec = Y_12Dn1_vecs(:,action_idx);
                X_12Dn1_vec = exp(Y_12Dn1_vec);
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
            s.Actor = (agent.Actor);         
            s.Critic = (agent.Critic);       
            s.Actor.UseDevice = 'cpu';
            s.Critic.UseDevice = 'cpu';
            s.ActorOptimizer = agent.ActorOptimizer;
            s.CriticOptimizer = agent.CriticOptimizer;
            s.AgentOptions = agent.AgentOptions;
            s.Params = agent.Params;
            s.trained_episodes = agent.trained_episodes;
            s.ReplayBuffer = agent.ReplayBuffer;
        end
    end
    
    methods (Static)
        function agent = loadobj(s)
            if isfield(s, 'ReplayBuffer')
                ReplayBuffer = s.ReplayBuffer;
            else
                ReplayBuffer = [];
            end
            agent = DeterministicActorCriticAgent(s.Actor, s.Critic, s.AgentOptions, s.Params, ReplayBuffer, s.trained_episodes);
            agent.ActorOptimizer = s.ActorOptimizer;
            agent.CriticOptimizer = s.CriticOptimizer;          
        end
    end
end
