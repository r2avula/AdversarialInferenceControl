function plot_RL_training_data(policy_fileFullPath, lambda1, lambda2)
saved_data = load(policy_fileFullPath,'trainStats');
trainStats = saved_data.trainStats;
episodes_num = trainStats.EpisodeIndex(end);
trainStats.TrainingOptions.MaxEpisodes = episodes_num;
% trainStats.Information.TrainingOpts = trainStats.TrainingOptions;
% elapsed_duration = trainStats.Information.ElapsedTime;
% checkpoint = train.RLTrainingResult.struct2class(trainStats);
episodeManager = plot(trainStats);
episodeManager.View.Container.RightWidth = 480;
episodeManager.View.Container.WindowMaximized = 1;
episodeManager.cbShowEpisodeQ0(struct('Value',1))
document = getDocument(episodeManager.View);
LossNames = document.LossNames;
loss_num = length(LossNames);


set(0,'DefaultFigureWindowStyle','normal')
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
fig_ = figure('Color',[1 1 1], 'Position', [10 10 800 600]);

t = tiledlayout(2,loss_num);
ax = nexttile([1,loss_num]);
copyobj(document.Widgets.Ax(1).Children,ax);
legend(ax,episodeManager.Options.DisplayName(episodeManager.Options.ShowOnFigure==1), "Location","southeast","Orientation","vertical","NumColumns",3);
objhl = findobj( ax, 'type', 'line' );
set( objhl, 'Markersize', 5, MarkerIndices=1:50:episodes_num)
mkrs={'o','x','*',"diamond", "square"};
for idx = 1:length(objhl)
    set( objhl(idx), 'Marker', mkrs{idx})
end
ax.XLabel.String = "Episode Number";
ax.YLabel.String = "Episode Reward";
xlim(ax, [1, episodes_num])
ylim(ax, [0, 2])
if ~isempty(lambda1)
    title(ax, strcat("Reinforcement Learning for HMM with $\lambda_0$ = ",num2str(lambda1),", $\lambda_1$ = ",num2str(lambda2)),'Interpreter','latex');
else
    title(ax, "Reinforcement Learning with real data",'Interpreter','latex');
end

for loss_idx = 1:loss_num
    ax = nexttile;
    copyobj(episodeManager.View.Document.Widgets.Ax(loss_idx+1).Children,ax);
    ax.XLabel.String = "Episode Number";
    ax.YLabel.String = LossNames(loss_idx);
    xlim(ax, [1, episodes_num])
end

set(fig_,'SelectionHighlight','off');
export_filename = policy_fileFullPath(1:end-4);

exportgraphics(t,strcat(export_filename,'.pdf'),'ContentType','vector');
exportgraphics(t,strcat(export_filename,'.jpg'));
saveas(fig_, strcat(export_filename,'.fig'))
delete(episodeManager)
close(fig_)

set(0,'DefaultFigureWindowStyle','docked')
end