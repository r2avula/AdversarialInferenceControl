classdef EpisodeInfo < matlab.mixin.Copyable
% EPISODEINFO

% Revised: 11-2-2018
% Copyright 2018 The MathWorks, Inc.
    properties
        % cumulative episode reward
        CumulativeReward (1,1) {mustBeNumeric} = 0
        CumulativeAdversarialRewardEstimate (1,1) {mustBeNumeric} = 0
        % number of steps take in the current episode
        StepsTaken (1,1) {mustBeNumeric} = 0
        % estimate of the value function at the beginning of the episode
        Q0 (1,1) {mustBeNumeric} = 0
        % observation captured at the beginning of the simulation
        InitialObservation = []
    end
    methods
        function this = EpisodeInfo()
            reset(this);
        end
        function reset(this)
            this.CumulativeReward = 0;
            this.CumulativeAdversarialRewardEstimate = 0;
            this.StepsTaken = 0;
            this.Q0 = 0;
            this.InitialObservation = [];
        end
        function update(this,reward, AdversarialRewardEstimate)
            this.CumulativeReward = this.CumulativeReward + reward;
            this.CumulativeAdversarialRewardEstimate = this.CumulativeAdversarialRewardEstimate + AdversarialRewardEstimate;            
            this.StepsTaken = this.StepsTaken + 1; 
        end
    end
end