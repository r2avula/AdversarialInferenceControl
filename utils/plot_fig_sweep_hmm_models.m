function plot_fig_sweep_hmm_models(plot_inputs)
NC= plot_inputs.NC;
inst_opt_FDC = plot_inputs.inst_opt_FDC;
subopt_det_subpolicy_FDC = plot_inputs.subopt_det_subpolicy_FDC;
subopt_DBS_FDC = plot_inputs.subopt_DBS_FDC;
RL_DeterministicActorCriticAgent = plot_inputs.RL_DeterministicActorCriticAgent;
RL_DeterministicActorCriticAgent_RD = plot_inputs.RL_DeterministicActorCriticAgent_RD;
subopt_DBS_FDC_UA = plot_inputs.subopt_DBS_FDC_UA;
valid_model_flag = plot_inputs.valid_model_flag;
P_HgHn1_elem_range = plot_inputs.P_HgHn1_elem_range;
P_HgHn1_elem_num = length(P_HgHn1_elem_range);

if(NC)
    mean_correct_detection_NC = plot_inputs.mean_correct_detection_NC;
    reward_NC = plot_inputs.reward_NC;
    fscores_NC = plot_inputs.fscores_NC;
    precision_NC = plot_inputs.precision_NC;
    recall_NC = plot_inputs.recall_NC;
    mean_PYkgY12kn1_NC = plot_inputs.mean_PYkgY12kn1_NC;
end

if(inst_opt_FDC)
    mean_correct_detection_inst_opt_FDC = plot_inputs.mean_correct_detection_inst_opt_FDC;
    reward_inst_opt_FDC = plot_inputs.reward_inst_opt_FDC;
    fscores_inst_opt_FDC = plot_inputs.fscores_inst_opt_FDC;
    precision_inst_opt_FDC = plot_inputs.precision_inst_opt_FDC;
    recall_inst_opt_FDC = plot_inputs.recall_inst_opt_FDC;
    mean_PYkgY12kn1_inst_opt_FDC= plot_inputs.mean_PYkgY12kn1_inst_opt_FDC;

    mean_correct_detection_inst_opt_FDC_UA = plot_inputs.mean_correct_detection_inst_opt_FDC_UA;
    reward_inst_opt_FDC_UA = plot_inputs.reward_inst_opt_FDC_UA;
    fscores_inst_opt_FDC_UA = plot_inputs.fscores_inst_opt_FDC_UA;
    precision_inst_opt_FDC_UA = plot_inputs.precision_inst_opt_FDC_UA;
    recall_inst_opt_FDC_UA = plot_inputs.recall_inst_opt_FDC_UA;
end

if(subopt_det_subpolicy_FDC)
    mean_correct_detection_subopt_det_SP_FDC = plot_inputs.mean_correct_detection_subopt_det_SP_FDC;
    reward_subopt_det_SP_FDC = plot_inputs.reward_subopt_det_SP_FDC;
    fscores_subopt_det_SP_FDC = plot_inputs.fscores_subopt_det_SP_FDC;
    precision_subopt_det_SP_FDC = plot_inputs.precision_subopt_det_SP_FDC;
    recall_subopt_det_SP_FDC = plot_inputs.recall_subopt_det_SP_FDC;
    mean_PYkgY12kn1_subopt_det_SP_FDC = plot_inputs.mean_PYkgY12kn1_subopt_det_SP_FDC;

    mean_correct_detection_subopt_det_SP_FDC_UA = plot_inputs.mean_correct_detection_subopt_det_SP_FDC_UA;
    reward_subopt_det_SP_FDC_UA = plot_inputs.reward_subopt_det_SP_FDC_UA;
    fscores_subopt_det_SP_FDC_UA = plot_inputs.fscores_subopt_det_SP_FDC_UA;
    precision_subopt_det_SP_FDC_UA = plot_inputs.precision_subopt_det_SP_FDC_UA;
    recall_subopt_det_SP_FDC_UA = plot_inputs.recall_subopt_det_SP_FDC_UA;
end

if(subopt_DBS_FDC)
    mean_correct_detection_subopt_DBS_FDC = plot_inputs.mean_correct_detection_subopt_DBS_FDC;
    reward_subopt_DBS_FDC = plot_inputs.reward_subopt_DBS_FDC;
    fscores_subopt_DBS_FDC = plot_inputs.fscores_subopt_DBS_FDC;
    precision_subopt_DBS_FDC = plot_inputs.precision_subopt_DBS_FDC;
    recall_subopt_DBS_FDC = plot_inputs.recall_subopt_DBS_FDC;
    mean_PYkgY12kn1_subopt_DBS_FDC = plot_inputs.mean_PYkgY12kn1_subopt_DBS_FDC;

    mean_correct_detection_subopt_DBS_FDC_UA = plot_inputs.mean_correct_detection_subopt_DBS_FDC_UA;
    reward_subopt_DBS_FDC_UA = plot_inputs.reward_subopt_DBS_FDC_UA;
    fscores_subopt_DBS_FDC_UA = plot_inputs.fscores_subopt_DBS_FDC_UA;
    precision_subopt_DBS_FDC_UA = plot_inputs.precision_subopt_DBS_FDC_UA;
    recall_subopt_DBS_FDC_UA = plot_inputs.recall_subopt_DBS_FDC_UA;
end

if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
    mean_correct_detection_RLDeterministicActorCriticAgent = plot_inputs.mean_correct_detection_RLDeterministicActorCriticAgent;
    reward_RLDeterministicActorCriticAgent = plot_inputs.reward_RLDeterministicActorCriticAgent;
    fscores_RLDeterministicActorCriticAgent = plot_inputs.fscores_RLDeterministicActorCriticAgent;
    precision_RLDeterministicActorCriticAgent = plot_inputs.precision_RLDeterministicActorCriticAgent;
    recall_RLDeterministicActorCriticAgent = plot_inputs.recall_RLDeterministicActorCriticAgent;
    mean_PYkgY12kn1_RLDeterministicActorCriticAgent = plot_inputs.mean_PYkgY12kn1_RLDeterministicActorCriticAgent;

    mean_correct_detection_RLDeterministicActorCriticAgent_UA = plot_inputs.mean_correct_detection_RLDeterministicActorCriticAgent_UA;
    reward_RLDeterministicActorCriticAgent_UA = plot_inputs.reward_RLDeterministicActorCriticAgent_UA;
    fscores_RLDeterministicActorCriticAgent_UA = plot_inputs.fscores_RLDeterministicActorCriticAgent_UA;
    precision_RLDeterministicActorCriticAgent_UA = plot_inputs.precision_RLDeterministicActorCriticAgent_UA;
    recall_RLDeterministicActorCriticAgent_UA = plot_inputs.recall_RLDeterministicActorCriticAgent_UA;
end

if(subopt_DBS_FDC_UA)
    mean_correct_detection_UA_subopt_DBS_FDC = plot_inputs.mean_correct_detection_UA_subopt_DBS_FDC;
    reward_UA_subopt_DBS_FDC = plot_inputs.reward_UA_subopt_DBS_FDC;
    fscores_UA_subopt_DBS_FDC = plot_inputs.fscores_UA_subopt_DBS_FDC;
    precision_UA_subopt_DBS_FDC = plot_inputs.precision_UA_subopt_DBS_FDC;
    recall_UA_subopt_DBS_FDC = plot_inputs.recall_UA_subopt_DBS_FDC;

    mean_correct_detection_UA_subopt_DBS_FDC_UA = plot_inputs.mean_correct_detection_UA_subopt_DBS_FDC_UA;
    reward_UA_subopt_DBS_FDC_UA = plot_inputs.reward_UA_subopt_DBS_FDC_UA;
    fscores_UA_subopt_DBS_FDC_UA = plot_inputs.fscores_UA_subopt_DBS_FDC_UA;
    precision_UA_subopt_DBS_FDC_UA = plot_inputs.precision_UA_subopt_DBS_FDC_UA;
    recall_UA_subopt_DBS_FDC_UA = plot_inputs.recall_UA_subopt_DBS_FDC_UA;
end

plot_mean_correct_detection = false;
plot_reward = true;
plot_fscores = false;
plot_precision = false;
plot_recall = false;
plot_mean_PYkgY12kn1 = false;

plot_ua = false;
font_size = 20;
line_width = 4;
marker_size = 16;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% set(0,'DefaultFigureWindowStyle','docked')
set(0,'DefaultFigureWindowStyle','normal')
figure_size = [1,1,1100,900];
num_columns = 3;
tiles_size_1 = 2;
tiles_size_2 = 2;
TileSpacing = 'loose';
Padding = 'loose';

if plot_mean_PYkgY12kn1
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_PYkgY12kn1_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_PYkgY12kn1_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end
        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_PYkgY12kn1_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_PYkgY12kn1_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete beliefs','LineStyle',':');
        end
        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_PYkgY12kn1_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG');
        end

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Average likelihood of smart meter data ($E[P_{Y_k\vert P_{Y_{1:k-1}}}]$)','FontSize',font_size+2,'Interpreter','latex')
    title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
    %     title(t,'Performance comparision','FontSize',font_size+2);
end

if plot_mean_correct_detection
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end
        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete beliefs','LineStyle',':');
        end
        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG');
        end

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Average correct inference (with strong adversary)','FontSize',font_size+2,'Interpreter','latex')
    title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
    %     title(t,'Performance comparision','FontSize',font_size+2);
end

if plot_reward
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        %         if subopt_DBS_FDC_UA
        %             plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
        %                 'DisplayName','Discrete beliefs(UA)','LineStyle',':');
        %         end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end

        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete AA beliefs','LineStyle',':');
        end

        if(subopt_DBS_FDC_UA)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
                'DisplayName','Discrete UA beliefs','LineStyle',':');
        end

        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG algorithm');
        end

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 2]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    %     ylabel(t,'Average Bayesian reward (with strong adversary)','FontSize',font_size+2,'Interpreter','latex')
    ylabel(t,'Average Bayesian reward','FontSize',font_size+2,'Interpreter','latex')
    %     title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_fscores
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end

        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete beliefs','LineStyle',':');
        end
        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG');
        end
        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Fscore (with strong adversary)','FontSize',font_size+2,'Interpreter','latex')
    title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_precision
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end

        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete beliefs','LineStyle',':');
        end
        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG');
        end

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Precision','FontSize',font_size+2,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_recall
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end

        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete beliefs','LineStyle',':');
        end
        if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_RLDeterministicActorCriticAgent(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG');
        end
        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Recall','FontSize',font_size+2,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_fscores && subopt_DBS_FDC_UA
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
            'DisplayName','strong adversary','LineStyle',':');

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_UA_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
            'DisplayName','weak adversary','LineStyle',':');

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Fscore','FontSize',font_size+2,'Interpreter','latex')
    title(t, 'Privacy Control designed with weak adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_mean_correct_detection && subopt_DBS_FDC_UA
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
            'DisplayName','strong adversary','LineStyle',':');

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_UA_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
            'DisplayName','weak adversary','LineStyle',':');

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Average correct inference','FontSize',font_size+2,'Interpreter','latex')
    title(t, 'Privacy Control designed with weak adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_reward && subopt_DBS_FDC_UA
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','AA \& UA with original data');
        end

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
            'DisplayName','AA with modified data','LineStyle',':');

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
            'DisplayName','UA with modified data','LineStyle',':');

        % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
        hold off
    end
    %     H(1).YAxis.Visible = 'on'; % adds y-axis
    linkaxes(H,'xy');
    ylim(H, [0 2]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');

    ylabel(t,'Average Bayesian reward','FontSize',font_size+2,'Interpreter','latex')
%     title(t, 'Privacy Control designed with weak adversarial model','FontSize',font_size+6,'Interpreter','latex')

    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_ua
    if plot_mean_correct_detection
        figure_ = figure('Color',[1 1 1]);
        figure_.Position = figure_size;
        t = tiledlayout(tiles_size_1, tiles_size_2);
        t.TileSpacing = TileSpacing;
        t.Padding = Padding;
        H = gobjects(1,P_HgHn1_elem_num-2);
        for idx = 2:P_HgHn1_elem_num-1
            H(idx-1) = nexttile;
            hold on

            if(NC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
            end

            if(inst_opt_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_inst_opt_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                    'DisplayName','Instantaneously-optimal','LineStyle',':');
            end
            if(subopt_det_subpolicy_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_subopt_det_SP_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                    'DisplayName','Degenerate sub-policies','LineStyle',':');
            end
            if(subopt_DBS_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                    'DisplayName','Discrete beliefs','LineStyle',':');
            end
            if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),mean_correct_detection_RLDeterministicActorCriticAgent_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                    'DisplayName','AMDPG');
            end

            % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
            xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
            H(idx-1).FontSize = font_size;
            %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
            hold off
        end
        %     H(1).YAxis.Visible = 'on'; % adds y-axis
        linkaxes(H,'xy');
        ylim(H, [0 1]);
        xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
        xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
        box(H,'on');
        grid(H,'on');

        ylabel(t,'Average correct inference (with weak adversary)','FontSize',font_size+2,'Interpreter','latex')
        title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

        lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
        lg.Layout.Tile = 'north';
        set(figure_,'SelectionHighlight','off');
        %     title(t,'Performance comparision','FontSize',font_size+2);
    end

    if plot_reward
        figure_ = figure('Color',[1 1 1]);
        figure_.Position = figure_size;
        t = tiledlayout(tiles_size_1, tiles_size_2);
        t.TileSpacing = TileSpacing;
        t.Padding = Padding;
        H = gobjects(1,P_HgHn1_elem_num-2);
        for idx = 2:P_HgHn1_elem_num-1
            H(idx-1) = nexttile;
            hold on

            if(NC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
            end

            if(inst_opt_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_inst_opt_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                    'DisplayName','Instantaneously-optimal','LineStyle',':');
            end

            if(subopt_det_subpolicy_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_det_SP_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                    'DisplayName','Degenerate sub-policies','LineStyle',':');
            end
            if(subopt_DBS_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                    'DisplayName','Discrete beliefs','LineStyle',':');
            end
            if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_RLDeterministicActorCriticAgent_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                    'DisplayName','AMDPG');
            end

            % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
            xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
            H(idx-1).FontSize = font_size;
            %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
            hold off
        end
        %     H(1).YAxis.Visible = 'on'; % adds y-axis
        linkaxes(H,'xy');
        ylim(H, [0 2]);
        xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
        xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
        box(H,'on');
        grid(H,'on');

        ylabel(t,'Average Bayesian reward (with weak adversary)','FontSize',font_size+2,'Interpreter','latex')

        lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
        lg.Layout.Tile = 'north';
        set(figure_,'SelectionHighlight','off');
    end

    if plot_fscores
        figure_ = figure('Color',[1 1 1]);
        figure_.Position = figure_size;
        t = tiledlayout(tiles_size_1, tiles_size_2);
        t.TileSpacing = TileSpacing;
        t.Padding = Padding;
        H = gobjects(1,P_HgHn1_elem_num-2);
        for idx = 2:P_HgHn1_elem_num-1
            H(idx-1) = nexttile;
            hold on

            if(NC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
            end

            if(inst_opt_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_inst_opt_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                    'DisplayName','Instantaneously-optimal','LineStyle',':');
            end

            if(subopt_det_subpolicy_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_subopt_det_SP_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                    'DisplayName','Degenerate sub-policies','LineStyle',':');
            end
            if(subopt_DBS_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                    'DisplayName','Discrete beliefs','LineStyle',':');
            end
            if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),fscores_RLDeterministicActorCriticAgent_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                    'DisplayName','AMDPG');
            end
            % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
            xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
            H(idx-1).FontSize = font_size;
            %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
            hold off
        end
        %     H(1).YAxis.Visible = 'on'; % adds y-axis
        linkaxes(H,'xy');
        ylim(H, [0 1]);
        xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
        xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
        box(H,'on');
        grid(H,'on');

        ylabel(t,'Fscore (with weak adversary)','FontSize',font_size+2,'Interpreter','latex')
        title(t, 'Privacy Control designed with strong adversarial model','FontSize',font_size+6,'Interpreter','latex')

        lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
        lg.Layout.Tile = 'north';
        set(figure_,'SelectionHighlight','off');
    end

    if plot_precision
        figure_ = figure('Color',[1 1 1]);
        figure_.Position = figure_size;
        t = tiledlayout(tiles_size_1, tiles_size_2);
        t.TileSpacing = TileSpacing;
        t.Padding = Padding;
        H = gobjects(1,P_HgHn1_elem_num-2);
        for idx = 2:P_HgHn1_elem_num-1
            H(idx-1) = nexttile;
            hold on

            if(NC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
            end

            if(inst_opt_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_inst_opt_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                    'DisplayName','Instantaneously-optimal','LineStyle',':');
            end

            if(subopt_det_subpolicy_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_det_SP_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                    'DisplayName','Degenerate sub-policies','LineStyle',':');
            end
            if(subopt_DBS_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                    'DisplayName','Discrete beliefs','LineStyle',':');
            end
            if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_RLDeterministicActorCriticAgent_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                    'DisplayName','AMDPG');
            end

            % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
            xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
            H(idx-1).FontSize = font_size;
            %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
            hold off
        end
        %     H(1).YAxis.Visible = 'on'; % adds y-axis
        linkaxes(H,'xy');
        ylim(H, [0 1]);
        xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
        xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
        box(H,'on');
        grid(H,'on');

        ylabel(t,'Precision (with weak adversary)','FontSize',font_size+2,'Interpreter','latex')

        lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
        lg.Layout.Tile = 'north';
        set(figure_,'SelectionHighlight','off');
    end

    if plot_recall
        figure_ = figure('Color',[1 1 1]);
        figure_.Position = figure_size;
        t = tiledlayout(tiles_size_1, tiles_size_2);
        t.TileSpacing = TileSpacing;
        t.Padding = Padding;
        H = gobjects(1,P_HgHn1_elem_num-2);
        for idx = 2:P_HgHn1_elem_num-1
            H(idx-1) = nexttile;
            hold on

            if(NC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
            end

            if(inst_opt_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_inst_opt_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                    'DisplayName','Instantaneously-optimal','LineStyle',':');
            end

            if(subopt_det_subpolicy_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_subopt_det_SP_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                    'DisplayName','Degenerate sub-policies','LineStyle',':');
            end
            if(subopt_DBS_FDC)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                    'DisplayName','Discrete beliefs','LineStyle',':');
            end
            if(RL_DeterministicActorCriticAgent || RL_DeterministicActorCriticAgent_RD)
                plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),recall_RLDeterministicActorCriticAgent_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                    'DisplayName','AMDPG');
            end

            % title(['$\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex');
            xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
            H(idx-1).FontSize = font_size;
            %         H(idx-1).YAxis.Visible = 'off'; % removes y-axis
            hold off
        end
        %     H(1).YAxis.Visible = 'on'; % adds y-axis
        linkaxes(H,'xy');
        ylim(H, [0 1]);
        xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
        xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
        box(H,'on');
        grid(H,'on');

        ylabel(t,'Recall (with weak adversary)','FontSize',font_size+2,'Interpreter','latex')

        lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
        lg.Layout.Tile = 'north';
        set(figure_,'SelectionHighlight','off');
    end
end

end