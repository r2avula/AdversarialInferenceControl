classdef InMemoryReplay < matlab.mixin.Copyable
    properties (Access = private)
        % Actual storage for the circular buffer
        Memory_ = struct([])
        % Flag for is full
        IsFull_ = false
        % Index to next location to fill
        NextIndex_ = 1

        % Bookeeping
        Tracker_
        DataSourceIDList_
    end

    properties (Dependent, Access = private)
        IsDoneVector
    end

    properties
        MaxLength
    end

    properties (SetAccess = private)
        ObservationInfo_
        ActionInfo_
    end

    properties (SetAccess = private, Dependent)
        ObservationDimension
        ActionDimension
    end

    %======================================================================
    % Public API
    %======================================================================
    methods
        function obj = InMemoryReplay(ObservationInfo,ActionInfo,MaxLength)
            obj.MaxLength = MaxLength;
            obj.ObservationInfo_ = ObservationInfo;
            obj.ActionInfo_ = ActionInfo;
            reset_(obj);
        end

        function IsDoneVector = get.IsDoneVector(obj)
            IsDoneVector = [obj.Memory_.IsDone];
        end

        function Experiences = batchExperience(obj, ExpStructArray)
            % EXPERIENCES = batchExperience(obj, EXPSTRUCTARRAY) return
            % scalar struct EXPERIENCE with 5 fields (Observation, Action,
            % Reward, NextObservation, IsDone). Each field has batched
            % value from column vector of struct array ExpStructArray.
            %
            % e.g. BatchSize experiences, 2 observation channels, 1 action
            % channel.
            %   ExpStructArray.Observation = {rand([4 4 2]); rand([4 1])}
            %   ExpStructArray.Action = {rand([1 3])}
            %   ExpStructArray.Reward = rand
            %   ExpStructArray.NextObservation = {rand([4 4 2]); rand([4 1])}
            %   ExpStructArray.IsDone = false
            %   ExpStructArray = repmat(ExpStructArray,BatchSize,1);
            %
            %   >> Experiences = batchExperience(obj, ExpStructArray)
            %
            %   => Experiences.Observation = {[4 4 2 BatchSize}; [4 1 BatchSize]}
            %   => Experiences.Action = {1 3 BatchSize]}
            %   => Experiences.NextObservation = {[4 4 2 BatchSize}; [4 1 BatchSize]}
            %   => size(Experiences.Reward) = [1 BatchSize]
            %   => size(Experiences.IsDone) = [1 BatchSize]

            Experiences = sim.getBatchExperienceArray(...
                ExpStructArray,obj.ObservationDimension,obj.ActionDimension);
        end

        function append(obj,Experiences,DataSourceID)
            append_(obj,Experiences,DataSourceID);
        end

        function reset(obj)
            reset_(obj);
        end

        function [Experiences,Idx] = sample(obj,BatchSize)
            [Experiences,Idx] = sample_(obj,BatchSize);
        end

        function Length = getLength(obj)
            Length = getLength_(obj);
        end

        function ObservationInfo = getObservationInfo(obj)
            ObservationInfo = obj.ObservationInfo_;
        end

        function ActionInfo = getActionInfo(obj)
            ActionInfo = obj.ActionInfo_;
        end

        function Experiences = allExperiences(obj)
            Experiences = allExperiences_(obj);
        end

        function resize(obj,newLength)
            resize_(obj,newLength);
        end

        function [Experiences,Mask] = getExperiences(obj,Indices,SequenceLength,DataSourceID)
            % [DATA,MASK] = getExperiences(REPLAYMEMORY,INDICES)
            % samples specific scalar experience struct DATA with 5 fields
            % (Observation, Action, Reward, NextObservation, IsDone) given
            % their indices in the replay memory specified by INDICES. Each
            % field has batched sequence data sampled from storage(s) in
            % REPLAYMEMORY. Each field has data in the form
            % DataSize-by-BatchSize-by-SequenceLength.
            %
            %   - If DATASOURCEID is not specified, evenly sample DATA from
            %   the buffer
            %   - If DATASOURCEID is specified, sample DATA from
            %   DATASOURCEID source

            [Experiences,Mask] = getExperiences_(obj,Indices, SequenceLength,DataSourceID);
            if SequenceLength < 2
                Experiences = batchExperience(obj, Experiences);
            else
                Experiences = sequenceExperience(obj, Experiences, SequenceLength);
            end
            if ~isempty(Mask)
                % Mask from getExperiences_ is (batchSize*SequenceLength) x
                % 1. Hence, it is necessary to reshape the size to 1 x
                % batchSize x sequenceLength. To correctly reshape Mask, it
                % first reshapes [1 SequenceLength BatchSize], and then
                % permutes 2nd and 3rd dimension.
                BatchSize = numel(Indices);
                Mask = reshape(Mask,[1 SequenceLength BatchSize]);
                Mask = permute(Mask, [1 3 2]);
            end
        end

        function Experiences = sequenceExperience(obj, ExpStructArray, SequenceLength)
            % EXPERIENCES = sequenceExperience(obj, EXPSTRUCTARRAY, BATCHSIZE, SEQUENCELENGTH) return
            % scalar struct EXPERIENCE with 5 fields (Observation, Action,
            % Reward, NextObservation, IsDone). Each field has batched
            % value from column vector of struct array ExpStructArray
            % (SequenceLength*BatchSize elements)
            %
            % e.g. BatchSize experiences, 2 observation channels, 1 action
            % channel.
            %   ExpStructArray.Observation = {rand([4 4 2]); rand([4 1])}
            %   ExpStructArray.Action = {rand([1 3])}
            %   ExpStructArray.Reward = rand
            %   ExpStructArray.NextObservation = {rand([4 4 2]); rand([4 1])}
            %   ExpStructArray.IsDone = false
            %   ExpStructArray = repmat(ExpStructArray,SequenceLength*BatchSize);
            %
            %   >> Experiences = sequenceExperience(obj, ExpStructArray)
            %
            %   => Experiences.Observation = {[4 4 2 BatchSize SequenceLength}; [4 1 BatchSize SequenceLength]}
            %   => Experiences.Action = {1 3 BatchSize SequenceLength]}
            %   => Experiences.NextObservation = {[4 4 2 BatchSize SequenceLength}; [4 1 BatchSize SequenceLength]}
            %   => size(Experiences.Reward) = [1 BatchSize SequenceLength]
            %   => size(Experiences.IsDone) = [1 BatchSize SequenceLength]

            Experiences = rl.replay.sequenceExperienceArray(...
                ExpStructArray,obj.ObservationDimension,obj.ActionDimension,SequenceLength);
        end

        function value = get.ObservationDimension(obj)
            value = {obj.ObservationInfo_.Dimension};
        end

        function value = get.ActionDimension(obj)
            value = {obj.ActionInfo_.Dimension};
        end

    end

    %======================================================================
    % Implement abstract methods
    %======================================================================
    methods (Access = protected)
        function append_(obj,Experiences,DataSourceID)
            % Length of DataSourceID is the same as Experiences.
            obj.DataSourceIDList_ = unique([obj.DataSourceIDList_;DataSourceID]);
            numSamplesFromNextIndexToEnd = obj.MaxLength - obj.NextIndex_ + 1;
            experienceLength = numel(Experiences);

            if experienceLength <= numSamplesFromNextIndexToEnd
                % In this case, it can store all of the experiences
                % without spliting it.
                selectedIdx = obj.NextIndex_:obj.NextIndex_+experienceLength-1;
                if isempty(obj.Memory_)
                    obj.Memory_ = Experiences(:);
                else
                    obj.Memory_(selectedIdx, 1) = Experiences;
                end
                % bookeeping
                obj.Tracker_(selectedIdx, 1) = DataSourceID(1:experienceLength);

                obj.NextIndex_ = obj.NextIndex_ + experienceLength;
                if obj.NextIndex_ > obj.MaxLength
                    obj.IsFull_ = true;
                    obj.NextIndex_ = 1;
                end
            else
                % If the experiences need to be split, the first half of
                % the experiences is appended at the end of the replay
                % buffer. The second half is appended at the beginning of
                % the replay buffer.

                if experienceLength >= obj.MaxLength
                    % If Experiences are bigger than the experience buffer size,
                    % it uses the last obj.MaxLength samples in Experiences.
                    selectedIndices = experienceLength - obj.MaxLength + 1 : experienceLength;
                    Experiences = Experiences(selectedIndices);
                    DataSourceID = DataSourceID(selectedIndices);

                    % To make this process same as a for-loop version, the
                    % starting index in the replay memory must be
                    % determinied appropriately. The starting position of
                    % Epxeriences is determinied based on next index, the
                    % original experience Length, and obj.MaxLength.
                    modval = mod(experienceLength, obj.MaxLength);
                    firstHalfStartIdx = obj.NextIndex_ + modval;
                    if firstHalfStartIdx > obj.MaxLength
                        firstHalfStartIdx = firstHalfStartIdx - obj.MaxLength;
                    end
                    firstHalfEndIdx = obj.MaxLength;
                    experienceLength = numel(Experiences);
                else
                    firstHalfStartIdx = obj.NextIndex_;
                    firstHalfEndIdx = obj.MaxLength;
                end

                firstHalfLength = firstHalfEndIdx - firstHalfStartIdx + 1;
                secondHalfStartIdx = 1;
                secondHalfEndIdx = experienceLength - firstHalfLength;

                % You cannot assign values to empty struct using indices.
                if isempty(obj.Memory_)
                    if secondHalfEndIdx==0
                        obj.Memory_ = Experiences(1:firstHalfLength);
                        obj.Tracker_ = DataSourceID(1:firstHalfLength);
                    else
                        obj.Memory_ = [Experiences(firstHalfLength+1:end);Experiences(1:firstHalfLength)];
                        obj.Tracker_ = [DataSourceID(firstHalfLength+1:end);DataSourceID(1:firstHalfLength)];
                    end
                else
                    obj.Memory_(firstHalfStartIdx:firstHalfEndIdx,1) = Experiences(1:firstHalfLength);
                    obj.Tracker_(firstHalfStartIdx:firstHalfEndIdx,1) = DataSourceID(1:firstHalfLength);

                    if secondHalfEndIdx~=0
                        obj.Memory_(secondHalfStartIdx:secondHalfEndIdx,1) = Experiences(firstHalfLength+1:end);
                        obj.Tracker_(secondHalfStartIdx:secondHalfEndIdx,1) = DataSourceID(firstHalfLength+1:end);
                    end
                end
                obj.IsFull_ = true;
                obj.NextIndex_ = secondHalfEndIdx + 1;
            end
        end

        function reset_(obj)
            obj.Memory_ = struct([]);
            obj.NextIndex_ = 1;
            obj.IsFull_ = false;
            obj.Tracker_ = zeros(obj.MaxLength,1,'uint64');
            obj.DataSourceIDList_ = [];
        end

        function [Experiences,Idx] = sample_(obj,BatchSize)
            Idx = [];
            CurrentLength = getLength(obj);
            if CurrentLength < BatchSize
                Experiences = struct([]);
            else
                % if DataSourceID is -1, randomly sample from all data
                Idx = randperm(getLength(obj),BatchSize);
                Experiences = obj.Memory_(Idx);
            end
        end

        function convertedExperiences = convertToNStepExperiences_(obj,idxes,discountFactor,nStepHorizon)
            % Converts sampled raw data in ReplayMemory to computed data
            % with n-step look-ahead.

            % This function computes weighted rewards, next observation at nStepHorizon, and isdone
            % at nStepHorizon.
            % REVISIT: Performance improvement.

            newestDataIndex = findNewOldDataIdxes_(obj);% index for newest data in memory
            numSamples = length(idxes);                % number of sampled experiences
            convertedExperiences = obj.Memory_(idxes); %place holder
            % Convert sampled experiences
            for idxCt = 1:numSamples
                % Initialization
                idx = idxes(idxCt);                         % sampled experience index
                rewards = zeros(nStepHorizon,1);            % vector of rewards for preview buffer
                maxIdx = min(idx+nStepHorizon-1,...         % maximum index for preview buffer
                    newestDataIndex + (idx>newestDataIndex)*obj.MaxLength);
                % Find index for last state and preview length
                for ct = idx:maxIdx
                    memoryIndex = mod(ct -1, obj.MaxLength) + 1;
                    rewards(ct-idx+1) = obj.Memory_(memoryIndex).Reward;
                    if obj.Memory_(memoryIndex).IsDone>=1 || (ct == maxIdx)
                        previewLength = ct-idx+1;
                        lastIndex = memoryIndex;
                        break;
                    end
                end
                % Get rewards, weights vector to compute return
                previewRewards = rewards(1:previewLength);
                weights = discountFactor.^(0:previewLength-1);
                computedReturn = weights*previewRewards;
                % Update Converted experience
                convertedExperiences(idxCt).Observation = obj.Memory_(idx).Observation;
                convertedExperiences(idxCt).Action = obj.Memory_(idx).Action;
                convertedExperiences(idxCt).Reward = computedReturn;
                convertedExperiences(idxCt).NextObservation = obj.Memory_(lastIndex).NextObservation;
                convertedExperiences(idxCt).IsDone = obj.Memory_(lastIndex).IsDone;
            end
        end

        function [Experiences,Masks] = getExperiences_(obj,Indices,SequenceLength,DataSourceID)
            % [DATA,MASK] = getExperiences(REPLAYMEMORY,INDICES)
            % samples specific scalar experience struct DATA with 5 fields
            % (Observation, Action, Reward, NextObservation, IsDone) given
            % their indices in the replay memory specified by INDICES. Each
            % field has batched sequence data sampled from storage(s) in
            % REPLAYMEMORY. Each field has data in the form
            % DataSize-by-BatchSize-by-SequenceLength.

            arguments
                obj
                Indices (1,:) {mustBeInteger,mustBePositive}
                SequenceLength (1,1) {mustBeInteger,mustBePositive} = 1
                DataSourceID (1,1) {mustBeInteger,mustBeGreaterThanOrEqual(DataSourceID,-1)} = -1
            end

            CurrentLength = getLength(obj);
            if CurrentLength < numel(Indices)
                Experiences = struct([]);
                Masks = [];
            else
                if SequenceLength < 2
                    if DataSourceID == -1
                        % if DataSourceID is -1, randomly sample from all data
                        Idx = Indices; %randperm(getLength(obj),BatchSize);
                        Experiences = obj.Memory_(Idx);
                        Masks = [];
                    else
                        % if scalar DataSourceID is specified, sample
                        % from that storage
                        MemoryIdx = find(obj.Tracker_ == DataSourceID);
                        if isempty(MemoryIdx)
                            Experiences = struct([]);
                        else
                            Idx = Indices;
                            Experiences = obj.Memory_(MemoryIdx(Idx));
                        end
                        Masks = [];
                    end
                else
                    if DataSourceID == -1
                        % randomly pick BatchSize starting point
                        Idx = Indices; %randperm(getLength(obj),BatchSize);
                        [Experiences,Masks] = getPaddedSequence_(obj,numel(Indices),SequenceLength,Idx);
                    else
                        % REVISIT
                        error(message('rl:general:errSequenceSamplingWithDataSourceIDNotSupported'));
                    end
                end
            end
        end


        function Length = getLength_(obj)
            % LENGTH = getLength(STORAGE) returns current length of
            % circular buffer STORAGE

            if obj.IsFull_
                Length = obj.MaxLength;
            else
                Length = obj.NextIndex_ - 1;
            end
        end

        function Experiences = allExperiences_(obj)
            if obj.IsFull_
                Experiences = obj.Memory_;
            else
                Experiences = obj.Memory_(1:getLength(obj));
            end
        end

        function resize_(obj,newLength)
            if newLength ~= obj.MaxLength
                if newLength > obj.MaxLength
                    [oldData,trackerID] = getLastNData_(obj,getLength_(obj));
                else
                    warning(message('rl:general:warnDataLostResize'));
                    [oldData,trackerID] = getLastNData_(obj,newLength);
                end
                obj.MaxLength = newLength;
                reset_(obj);
                obj.append(oldData,trackerID);
            end
        end
    end

    methods
        function [data,trackerIDList] = getLastNData(obj,n)
            [data,trackerIDList] = getLastNData_(obj,n);
        end
    end


    methods (Access = private)
        function [data,trackerIDList] = getLastNData_(obj,n)
            % Returns last N data in Memory_

            n = min(obj.MaxLength,n);
            if obj.IsFull_
                newestDataIndex = (obj.NextIndex_-1) + (obj.NextIndex_ == 1) * obj.MaxLength;
            else
                newestDataIndex = obj.NextIndex_-1;
            end
            startIdx = newestDataIndex - n + 1;
            if obj.IsFull_
                if n <= newestDataIndex
                    data = obj.Memory_(startIdx:newestDataIndex);
                    trackerIDList = obj.Tracker_(startIdx:newestDataIndex);
                else
                    data = [obj.Memory_(obj.MaxLength-abs(startIdx):obj.MaxLength); obj.Memory_(1:newestDataIndex)];
                    trackerIDList = [obj.Tracker_(obj.MaxLength-abs(startIdx):obj.MaxLength); obj.Tracker_(1:newestDataIndex)];
                end
            else
                if n >= newestDataIndex
                    data = obj.Memory_(1:newestDataIndex);
                    trackerIDList = obj.Tracker_(1:newestDataIndex);
                else
                    data = obj.Memory_(startIdx:newestDataIndex);
                    trackerIDList = obj.Tracker_(startIdx:newestDataIndex);
                end
            end
        end

        function [Experiences,Masks] = getPaddedSequence_(obj,BatchSize,SequenceLength,Idx)

            % Idx: Start position of sub sequences. Idx can be
            % the middle of a trajectory.

            Experiences = struct([]);
            Masks = logical([]);
            for ct = 1:BatchSize
                StartIdx = Idx(ct);
                Experience = obj.Memory_(StartIdx);
                Mask = true(SequenceLength,1);

                % if starting index is end of episode => padd with same
                % data, only the first element is real data. If the
                % experience is the last experience the agent obtained
                % (startIdx + 1 == obj.NextIndex_), the episode has not
                % finished yet.
                if StartIdx == obj.MaxLength
                    NextStepIdx = 1;
                else
                    NextStepIdx = StartIdx + 1;
                end
                if Experience.IsDone || NextStepIdx == obj.NextIndex_
                    Experience = repmat(Experience,SequenceLength,1);
                    Mask(2:end) = false;
                    Experiences = [Experiences;Experience]; %#ok<AGROW> 
                    Masks = [Masks;Mask]; %#ok<AGROW> 
                    continue
                end

                % if starting index is not end of episode, search
                % for next element in the same source
                CurrentID = obj.Tracker_(StartIdx);
                Step = 1;
                SequenceStep = 1;
                PreviewStep = StartIdx;
                while SequenceStep <= SequenceLength - 1
                    CurrentStep = PreviewStep;
                    PreviewStep = StartIdx + Step;
                    if PreviewStep > obj.MaxLength
                        PreviewStep = StartIdx + Step - obj.MaxLength;
                    end
                    if PreviewStep == obj.NextIndex_
                        % obj.Memory_(PreviewStep) may not exist, or may be in a different sequence.
                        RemainingStep = (SequenceLength - SequenceStep);
                        if RemainingStep > 0
                            CurrentExp = obj.Memory_(CurrentStep);
                            Experience = [Experience; repmat(CurrentExp,RemainingStep,1)]; %#ok<AGROW> 
                            Mask(end-RemainingStep+1:end) = false;
                        end
                        break
                    end
                    if obj.Tracker_(PreviewStep) == CurrentID
                        CurrentExp = obj.Memory_(PreviewStep);
                        Experience = [Experience; CurrentExp]; %#ok<AGROW> 
                        if CurrentExp.IsDone
                            % padd if not enough steps until end of
                            % episode
                            RemainingStep = (SequenceLength - 1 - SequenceStep);
                            if RemainingStep > 0
                                Experience = [Experience; repmat(CurrentExp,RemainingStep,1)]; %#ok<AGROW> 
                                Mask(end-RemainingStep+1:end) = false;
                            end
                            break
                        end
                        SequenceStep = SequenceStep + 1;
                    end
                    Step = Step + 1;
                end
                Experiences = [Experiences;Experience]; %#ok<AGROW> 
                Masks = [Masks;Mask]; %#ok<AGROW> 
            end
        end

        function [newestDataIndex,oldestDataIndex] = findNewOldDataIdxes_(obj)
            % Find the index for newest data and oldest data in memory
            % e.g. 1) if the memory is full and NextIndex_ is 3. the oldest
            % data is store at index=3, and the newest data is stored at
            % index=3.
            %
            % e.g. 2) if the memory is full and NextIndex_ is 1. the oldest
            % data is store at index=1, and the newest data is stored at
            % the last index of the memory.
            %
            % e.g. 3) if the memoery is not full, the oldest data is stored
            % at index = NextIndex_, and the newest data is stored at index
            % = NextIndex_ - 1

            if obj.IsFull_
                oldestDataIndex = obj.NextIndex_;
                newestDataIndex = (obj.NextIndex_-1) + (obj.NextIndex_ == 1) * obj.MaxLength;
            else
                oldestDataIndex = 1;
                newestDataIndex = obj.NextIndex_-1;
            end
        end
    end

    methods
        function NextIndex = getIsFull(obj)
            NextIndex = obj.IsFull_;
        end

        function NextIndex = getNextIndex(obj)
            NextIndex = obj.NextIndex_;
        end

        function structMemory = getMemory(obj)
            structMemory = obj.Memory_;
        end

        function convertedExperiences = qeConvertToNStepExperiences_(obj,idxes,discountFactor,nStepHorizon)
            convertedExperiences = convertToNStepExperiences_(obj,idxes,discountFactor,nStepHorizon);
        end
    end
end