function simStartup(wait_time,rng_id)
if(nargin==1)
    rng_id = 1;
elseif(nargin==0)
    rng_id = 1;
    wait_time = 2;
end

pathCell = regexp(path, pathsep, 'split');
[fileDir,~,~] = fileparts(mfilename('fullpath'));
cd(fileDir)
test_dir = [fileDir filesep 'utils'];
onPath = any(strcmpi(test_dir, pathCell));
if (~onPath)
    path(pathdef);
    if ispc
        run('startup');
    elseif exist('gurobi', 'file') == 0
        path_to_gurobi_setup = ''; 
        if exist(path_to_gurobi_setup, 'file') == 0
            error('Gurobi setup file not found!')
        else
            fprintf('Adding gurobi solver to the Matlab path...');
            evalc('run(path_to_gurobi_setup)');
            fprintf('Done.\n');
        end
    end
    addpath('utils');
    addpath(['utils', filesep, 'rl']);
    addpath(genpath(['utils', filesep, 'adversary']));
    addpath(genpath(['utils', filesep, 'battery']));
    addpath(genpath(['utils', filesep, 'common']));
    addpath(genpath(['utils', filesep, 'submodules']));
    addpath(genpath(['utils', filesep, 'mdp']));
    addpath(genpath(['utils', filesep, 'pomdp']));
    addpath(genpath(['utils', filesep, 'rl', filesep, 'utils']));
    addpath(genpath(['utils', filesep, 'rl', filesep, 'ActorCriticAgent_PolicyAwareAdversary']));
    addpath(genpath('configs'));
    fprintf('Initializing MPT...');
    evalc('mpt_init');
    fprintf('Done.\n');
end

if(wait_time>0)
    assertParpool(wait_time);
end

rng(rng_id,'twister');
end
