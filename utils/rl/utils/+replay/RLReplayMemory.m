classdef RLReplayMemory < matlab.mixin.Copyable
    % REPLAYMEMORY: Stores agent experiences (states, actions,
    % rewards and logged signals) in a buffer. Agents sample data uniformly from this buffer
    % during training.
    properties (Access = protected)
        InternalReplayMemory_
    end

    properties (Dependent, SetAccess = protected)
        MaxLength
        Length
    end

    %======================================================================
    % Public API
    %======================================================================
    methods
        function obj = RLReplayMemory(ObsInfo,ActInfo,MaxLength)
            % construct InternalReplayMemory_
            % use internal memory for uniforma sampling
            obj.InternalReplayMemory_ = replay.InMemoryReplay(ObsInfo,ActInfo,MaxLength);
        end

        function append(obj,Experiences,DataSourceID)
            % append:  append new experience to replay memory
            %
            %   APPEND(REPLAYMEMORY,EXPERIENCE) appends scalar or vector
            %   experience struct EXPERIENCE to the default data source ID
            %   0 to replay memory REPLAYMEMORY. EXPERIENCE is a struct
            %   array that holds "Observation", "Action", "Reward",
            %   "NextObservation", and "IsDone" fields.
            %
            %   APPEND(REPLAYMEMORY,EXPERIENCE,DATASOURCEID) appends scalar
            %   or vector experience struct EXPERIENCE from data source
            %   DATASOURCEID to replay memory REPLAYMEMORY.

            arguments
                obj
                Experiences (:,1) struct
                DataSourceID (:,1) uint64 = 0
            end

            validateExperience(obj,Experiences);
            appendWithoutSampleValidation(obj,Experiences,DataSourceID);
        end

        function [Experiences,Idx] = sample(obj,BatchSize)
            % sample: Sample a minibatch from replay memory
            %
            % [DATA, MASK, IDX, WEIGHTS] = sample(REPLAYMEMORY,BATCHSIZE,'SequenceLength',SEQUENCELENGTH,'NStepHorizon',NSTEPHORIZON,'DiscountFactor',DISCOUNTFACTOR,'DataSourceID',DATASOURCEID)
            % returns scalar experience struct DATA with 5 fields
            % (Observation, Action, Reward, NextObservation, IsDone).
            %
            % Each field of DATA has batched sequence data sampled from storage(s) in
            % REPLAYMEMORY. Each field has data in the form
            % DataSize-by-BATCHSIZE-by-SEQUENCELENGTH
            %
            %   - If DATASOURCEID is not specified, evenly sample DATA from
            %   the buffer
            %   - If DATASOURCEID is specified, sample DATA from
            %   DATASOURCEID source
            %
            % MASK is the sequence padding mask, returned as a logical array
            % with length equal to SEQUENCELENGTH.
            %
            % IDX is indices of sampled experiences.
            %
            % WEIGHTS is 1 if REPLAYMEMORY is rlReplayMemory. WEIGHTS
            % contains the importance-sampling weights for sampled expriences
            % if REPLAYMEMORY is rlPrioritizedReplayMemory.

            arguments
                obj
                BatchSize (1,1) {mustBeInteger,mustBePositive}
            end

            [Experiences, Idx] = sample(obj.InternalReplayMemory_,BatchSize);
        end

        function reset(obj)
            % reset: reset replay memory
            %
            % reset(REPLAYMEMORY) clear all data from replay memory REPLAYMEMORY

            reset(obj.InternalReplayMemory_);
        end

        function Length = get.Length(obj)
            Length = getLength(obj.InternalReplayMemory_);
        end

        function MaxLength = get.MaxLength(obj)
            MaxLength = obj.InternalReplayMemory_.MaxLength;
        end

        function Experiences = allExperiences(obj)

            arguments
                obj
            end

            Experiences = allExperiences(obj.InternalReplayMemory_);
        end

        function resize(obj,newLength)
            % resize: Resizes replay memory
            %
            % Resize(REPLAYMEMORY, NEWLENGTH) resizes replay memory
            % REPLAYMEMORY with a new length NEWLENGTH. If NEWLENGTH is
            % longer than the current length, it keeps all data. If
            % NEWLENGTH is shorter than the current length, it may lose
            % some data in the replay memory.

            arguments
                obj
                newLength (1,1) {mustBePositive,mustBeFinite}
            end

            % Change length of buffer while keeping old memory
            resize(obj.InternalReplayMemory_,newLength);
        end

        function validateExperience(obj,NewExperience)
            % validateExperience: validate input experiences whether
            % it is compatible with a replay memory
            %
            % validateExperience(REPLAYMEMORY,NEWEXPRIENCES) validates
            % input experiences NEWEXPERIENCES with a replay memory REPLAY
            % MEMORY specification.
            %
            % NEWEXPERIENCES is a struct array that holds "Obseration",
            % "Action", "Reward", "NextObservation", "Reward", and "IsDone"
            % fields. The dimensions of Observation and NextObservation in
            % each experience must be the same as the dimensions specified
            % in ObserationInfo. The dimension of Action must be the same
            % as the dimension specified in ActionInfo. Reward and IsDone
            % in each experience must be a scalar.

            if ~isstruct(NewExperience)
                error(message('rl:general:errAppendExperienceNotStruct'));
            end

            if ~isfield(NewExperience,"Observation")
                error(message('rl:general:errExperienceMustHaveObservationField'));
            end
            if ~isfield(NewExperience,"Action")
                error(message('rl:general:errExperienceMustHaveActionField'));
            end
            if ~isfield(NewExperience,"Reward")
                error(message('rl:general:errExperienceMustHaveRewardField'));
            end
            if ~isfield(NewExperience,"NextObservation")
                error(message('rl:general:errExperienceMustHaveNextObservationField'));
            end
            if ~isfield(NewExperience,"IsDone")
                error(message('rl:general:errExperienceMustHaveIsDoneField'));
            end

            % validate dimension of NewExperience with data specs
            % validate only first experience
            NumExperiences = numel(NewExperience);

            for experienceIdx = 1:NumExperiences
                NewObs = NewExperience(experienceIdx).Observation;
                NewAction = NewExperience(experienceIdx).Action;
                NewReward = NewExperience(experienceIdx).Reward;
                NewNextObs = NewExperience(experienceIdx).NextObservation;
                NewIsDone = NewExperience(experienceIdx).IsDone;

                % validate obsrevation and nextobservation

                if ~iscell(NewObs)
                    error(message('rl:general:errObservationNotCell'));
                end

                if ~iscell(NewNextObs)
                    error(message('rl:general:errNextObservationNotCell'));
                end

                numObsChannels = numel(obj.InternalReplayMemory_.ObservationDimension);
                if numObsChannels ~=  numel(NewObs)
                    error(message('rl:general:errIncorrectObservationDim'));
                end
                if numObsChannels ~=  numel(NewNextObs)
                    error(message('rl:general:errIncorrectNextObservationDim'));
                end

                for obsCh = 1:numObsChannels
                    if ~all(size(NewObs{obsCh}) == obj.InternalReplayMemory_.ObservationDimension{obsCh})
                        error(message('rl:general:errIncorrectObservationDim'));
                    end
                    if ~all(size(NewNextObs{obsCh}) == obj.InternalReplayMemory_.ObservationDimension{obsCh})
                        error(message('rl:general:errIncorrectNextObservationDim'));
                    end
                end

                % validate action
                if ~iscell(NewAction)
                    error(message('rl:general:errActionNotCell'));
                end

                numActionChannels = numel(obj.InternalReplayMemory_.ActionDimension);
                for obsCh = 1:numActionChannels
                    if ~size(NewAction{obsCh}) == obj.InternalReplayMemory_.ActionDimension{obsCh}
                        error(message('rl:general:errIncorrectActionDim'));
                    end
                end

                % validate reward
                if ~isscalar(NewReward)
                    error(message('rl:general:errIncorrectRewardDim'));
                end

                % validate isdone
                if ~isscalar(NewIsDone)
                    error(message('rl:general:errIncorrectIsDoneDim'));
                end
            end
        end

        function ObsInfo = getObservationInfo(this)
            ObsInfo = getObservationInfo(this.InternalReplayMemory_);
        end

        function ActionInfo = getActionInfo(this)
            ActionInfo = getActionInfo(this.InternalReplayMemory_);
        end
    end

    methods (Access=protected)
        function that = copyElement(this)
            % hard copy
            that = copyElement@matlab.mixin.Copyable(this);
            that.InternalReplayMemory_ = copy(this.InternalReplayMemory_);
        end
    end

    methods (Hidden)
        function [Experiences,Mask] = getExperiences(obj, Indices, NameValueArgs)
            % [DATA,MASK] = getExperiences(REPLAYMEMORY,INDICES)
            % samples specific scalar experience struct DATA with 5 fields
            % (Observation, Action, Reward, NextObservation, IsDone) given
            % their indices in the replay memory. Each field has batched
            % sequence data sampled from storage(s) in REPLAYMEMORY. Each
            % field has data in the form
            % DataSize-by-BatchSize-by-SequenceLength
            %
            %   - If DATASOURCEID is not specified, evenly sample DATA from
            %   the buffer
            %   - If DATASOURCEID is specified, sample DATA from
            %   DATASOURCEID source

            arguments
                obj
                Indices (1,:) {mustBeInteger,mustBePositive}
                NameValueArgs.SequenceLength (1,1) {mustBeInteger,mustBePositive} = 1
                NameValueArgs.DataSourceID (1,1) {mustBeInteger,mustBeGreaterThanOrEqual(NameValueArgs.DataSourceID,-1)} = -1
            end

            [Experiences,Mask] = getExperiences(obj.InternalReplayMemory_,...
                Indices, NameValueArgs.SequenceLength, NameValueArgs.DataSourceID);
        end

        function collectExperienceFromOldFormat(obj,bufferPrevVersion)
            arguments
                obj
                bufferPrevVersion (1,1) rl.util.ExperienceBuffer
            end

            % get cell array experience in chronological order
            data = getLastNData(bufferPrevVersion,bufferPrevVersion.Length);
            % convert data to struct format
            data = [data{:}];
            data = reshape(data,5,[]);
            structFieldNames = ["Observation" "Action" "Reward" "NextObservation" "IsDone"];
            data = cell2struct(data, structFieldNames, 1);
            % append to new buffer
            append(obj,data);
        end

        function appendWithoutSampleValidation(obj,Experiences,DataSourceID)
            % append experiences without validating the experiences

            arguments
                obj
                Experiences (:,1) struct
                DataSourceID (:,1) uint64 = 0
            end
            % REVISIT: vectorize, optimize performance as append happens
            % every learning step
            if isscalar(DataSourceID)
                DataSourceID = repmat(DataSourceID,size(Experiences));
            else
                if numel(Experiences) ~= numel(DataSourceID)
                    error(message('rl:general:errSizeDataSourceIDSize'));
                end
            end
            append(obj.InternalReplayMemory_,Experiences,DataSourceID);
        end

        function nextIndex = qeGetNextIndex(obj)
            % getNextIndex returns the index for the next sample.
            nextIndex = getNextIndex(obj.InternalReplayMemory_);
        end

        function isFull = qeGetIsFull(obj)
            % getIsFull returns whether the memory is full.
            isFull = getIsFull(obj.InternalReplayMemory_);
        end

        function structMemory = qeGetMemory(obj)
            structMemory = getMemory(obj.InternalReplayMemory_);
        end

        function convertedExperiences = qeConvertToNStepExperiences_(obj,idxes,discountFactor,nStepHorizon)
            convertedExperiences = qeConvertToNStepExperiences_(obj.InternalReplayMemory_,idxes,discountFactor,nStepHorizon);
        end
    end
end