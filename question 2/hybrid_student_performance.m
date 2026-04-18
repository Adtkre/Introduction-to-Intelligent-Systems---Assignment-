%% =================================================================
%  HYBRID INTELLIGENT SYSTEM — Student Performance Prediction
%  Method   : ANFIS (Adaptive Neuro-Fuzzy Inference System)
%  Inputs   : Attendance (%), Assignment Marks (0-100), Test Marks (0-100)
%  Output   : Performance Score (0-100)  →  Poor / Average / Good
%  Toolbox  : Fuzzy Logic Toolbox (MATLAB R2019b+)
%% =================================================================

clear; clc; close all;

fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║   HYBRID FUZZY-NEURAL STUDENT PERFORMANCE SYSTEM    ║\n');
fprintf('╚══════════════════════════════════════════════════════╝\n\n');

rng(42);   % reproducibility

%% ──────────────────────────────────────────────────────────────────
%  SECTION 1 :  GENERATE SYNTHETIC TRAINING DATA
%  Performance = weighted blend + noise
%  Poor → 0-40 | Average → 41-70 | Good → 71-100
%% ──────────────────────────────────────────────────────────────────
N = 500;

attendance   = rand(N,1) * 100;          % 0-100 %
assignment   = rand(N,1) * 100;          % 0-100 marks
testMarks    = rand(N,1) * 100;          % 0-100 marks

% Ground-truth output: weighted sum + noise, clipped to [0,100]
performance  = 0.25*attendance + 0.35*assignment + 0.40*testMarks ...
               + randn(N,1)*4;
performance  = min(max(performance, 0), 100);

data = [attendance, assignment, testMarks, performance];

% Split 80/20 train/test
idx      = randperm(N);
trainIdx = idx(1:400);
testIdx  = idx(401:end);

trainData = data(trainIdx,:);
testData  = data(testIdx,:);

fprintf('[1] Training data : %d samples\n', size(trainData,1));
fprintf('[1] Test data     : %d samples\n\n', size(testData,1));

%% ──────────────────────────────────────────────────────────────────
%  SECTION 2 :  BUILD INITIAL SUGENO FIS  (Fuzzy Layer)
%  Grid partition: 3 MFs per input  (trimf)
%  This defines the fuzzy structure BEFORE neural training
%% ──────────────────────────────────────────────────────────────────
fprintf('[2] Generating initial FIS via grid partitioning ...\n');

genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = [3 3 3];     % Low/Med/High per input
genOpt.InputMembershipFunctionType = 'trimf';

initFIS = genfis(trainData(:,1:3), trainData(:,4), genOpt);

% Label MFs meaningfully
inputNames = {'Attendance','AssignmentMarks','TestMarks'};
mfNames    = {'Low','Medium','High'};

for i = 1:3
    initFIS.Inputs(i).Name = inputNames{i};
    for j = 1:3
        initFIS.Inputs(i).MembershipFunctions(j).Name = mfNames{j};
    end
end
initFIS.Outputs(1).Name = 'PerformanceScore';

fprintf('   FIS type   : %s\n', initFIS.Type);
fprintf('   Inputs     : %d | MFs each: 3\n', numel(initFIS.Inputs));
fprintf('   Output     : %s\n\n', initFIS.Outputs(1).Name);

%% ──────────────────────────────────────────────────────────────────
%  SECTION 3 :  ANFIS TRAINING  (Neural Learning Layer)
%  Hybrid algorithm: Backpropagation (premise) + LSE (consequent)
%  The neural network tunes:
%   → Premise params : MF centres & widths   (gradient descent)
%   → Consequent params : Sugeno coefficients (least squares)
%% ──────────────────────────────────────────────────────────────────
fprintf('[3] Training ANFIS hybrid system ...\n');

trainOpt                   = anfisOptions();
trainOpt.InitialFIS        = initFIS;
trainOpt.EpochNumber       = 120;
trainOpt.InitialStepSize   = 0.01;
trainOpt.StepSizeIncreaseRate = 1.1;
trainOpt.StepSizeDecreaseRate = 0.9;
trainOpt.ValidationData    = testData;
trainOpt.DisplayANFISInformation = false;
trainOpt.DisplayErrorValues      = false;
trainOpt.DisplayStepSize         = false;
trainOpt.DisplayFinalResults     = false;

[trainedFIS, trainError, ~, ~, testError] = anfis(trainData, trainOpt);

fprintf('   Epochs trained         : %d\n', length(trainError));
fprintf('   Final training RMSE    : %.4f\n', trainError(end));
fprintf('   Final validation RMSE  : %.4f\n\n', testError(end));

%% ──────────────────────────────────────────────────────────────────
%  SECTION 4 :  PREDICT & CLASSIFY
%% ──────────────────────────────────────────────────────────────────
trainPred = evalfis(trainedFIS, trainData(:,1:3));
testPred  = evalfis(trainedFIS, testData(:,1:3));
testPred  = min(max(testPred,0),100);

% Classify into labels
classify = @(x) cellstr(repmat('Average',numel(x),1));  % default

function labels = classifyPerf(scores)
    labels = cell(numel(scores),1);
    for i = 1:numel(scores)
        if scores(i) <= 40
            labels{i} = 'Poor';
        elseif scores(i) <= 70
            labels{i} = 'Average';
        else
            labels{i} = 'Good';
        end
    end
end

trueLabels  = classifyPerf(testData(:,4));
predLabels  = classifyPerf(testPred);

accuracy = mean(strcmp(trueLabels, predLabels)) * 100;
fprintf('[4] Classification Accuracy : %.2f%%\n\n', accuracy);

%% ──────────────────────────────────────────────────────────────────
%  SECTION 5 :  PLOTS
%% ──────────────────────────────────────────────────────────────────
palette = struct('poor','#E05C6E','avg','#E8A24A','good','#4CAF80',...
                 'blue','#5B8DEE','muted','#8A8AAA','dark','#141417');

%% ── Figure 1: Membership Functions (Before vs After Training) ──
fig1 = figure('Name','Membership Functions','Color','white',...
              'Position',[60 60 1300 720]);

inputLabels = {'Attendance (%)','Assignment Marks (0-100)','Test Marks (0-100)'};
mfColors = {'#5B8DEE','#E8A24A','#E05C6E'};

for inp = 1:3
    % BEFORE training
    ax = subplot(3,2,(inp-1)*2+1);
    x = linspace(0,100,500);
    hold on;
    for k = 1:3
        p = initFIS.Inputs(inp).MembershipFunctions(k).Parameters;
        y = max(min((x-p(1))./(p(2)-p(1)+eps),(p(3)-x)./(p(3)-p(2)+eps)),0);
        plot(x,y,'Color',mfColors{k},'LineWidth',2.2,'DisplayName',mfNames{k});
        fill([x fliplr(x)],[y zeros(size(y))],mfColors{k},...
             'FaceAlpha',0.07,'EdgeColor','none');
    end
    hold off;
    if inp == 1, title('Before ANFIS Training','FontSize',11,'FontWeight','bold'); end
    ylabel(inputLabels{inp},'FontSize',9); ylim([0 1.15]);
    legend('Low','Medium','High','Location','northeast','FontSize',8,'Box','off');
    grid on; box off; set(ax,'FontSize',9);
    
    % AFTER training
    ax2 = subplot(3,2,(inp-1)*2+2);
    hold on;
    for k = 1:3
        p = trainedFIS.Inputs(inp).MembershipFunctions(k).Parameters;
        y = max(min((x-p(1))./(p(2)-p(1)+eps),(p(3)-x)./(p(3)-p(2)+eps)),0);
        plot(x,y,'Color',mfColors{k},'LineWidth',2.2,'DisplayName',mfNames{k});
        fill([x fliplr(x)],[y zeros(size(y))],mfColors{k},...
             'FaceAlpha',0.07,'EdgeColor','none');
    end
    hold off;
    if inp == 1, title('After ANFIS Training (Tuned)','FontSize',11,'FontWeight','bold'); end
    ylim([0 1.15]); legend('Low','Medium','High','Location','northeast','FontSize',8,'Box','off');
    grid on; box off; set(ax2,'FontSize',9);
end
sgtitle('Membership Functions: Before vs After Neural Tuning',...
        'FontSize',14,'FontWeight','bold');

%% ── Figure 2: Learning Curves ──
fig2 = figure('Name','Learning Curves','Color','white','Position',[60 820 640 350]);
epochs = 1:length(trainError);
plot(epochs, trainError, 'Color','#5B8DEE','LineWidth',2.2,'DisplayName','Training RMSE');
hold on;
plot(epochs, testError,  'Color','#E05C6E','LineWidth',2.2,'LineStyle','--','DisplayName','Validation RMSE');
hold off;
xlabel('Epoch','FontSize',11); ylabel('RMSE','FontSize',11);
title('ANFIS Learning Curves','FontSize',13,'FontWeight','bold');
legend('Location','northeast','FontSize',9,'Box','off');
grid on; box off;

%% ── Figure 3: Predicted vs Actual (scatter) ──
fig3 = figure('Name','Predicted vs Actual','Color','white','Position',[720 820 580 420]);
% Color by class
colors_pt = cell(size(testData,1),1);
for i = 1:size(testData,1)
    if testData(i,4) <= 40,      colors_pt{i} = '#E05C6E';
    elseif testData(i,4) <= 70,  colors_pt{i} = '#E8A24A';
    else,                         colors_pt{i} = '#4CAF80';
    end
end
hold on;
for i = 1:size(testData,1)
    scatter(testData(i,4), testPred(i), 36, ...
            hex2rgb(colors_pt{i}), 'filled','MarkerFaceAlpha',0.7);
end
plot([0 100],[0 100],'k--','LineWidth',1.5,'DisplayName','Perfect Fit');
hold off;
xlabel('Actual Performance Score','FontSize',11);
ylabel('Predicted Performance Score','FontSize',11);
title(sprintf('Predicted vs Actual  |  Accuracy: %.1f%%', accuracy),...
      'FontSize',12,'FontWeight','bold');
% custom legend
h1 = plot(nan,nan,'o','Color','#E05C6E','MarkerFaceColor','#E05C6E');
h2 = plot(nan,nan,'o','Color','#E8A24A','MarkerFaceColor','#E8A24A');
h3 = plot(nan,nan,'o','Color','#4CAF80','MarkerFaceColor','#4CAF80');
legend([h1 h2 h3],{'Poor','Average','Good'},'Location','northwest','Box','off');
grid on; box off; xlim([0 100]); ylim([0 100]);

%% ── Figure 4: Confusion Matrix ──
categories = {'Poor','Average','Good'};
confMat = zeros(3,3);
catMap  = containers.Map(categories,1:3);
for i = 1:numel(trueLabels)
    r = catMap(trueLabels{i});
    c = catMap(predLabels{i});
    confMat(r,c) = confMat(r,c) + 1;
end
fig4 = figure('Name','Confusion Matrix','Color','white','Position',[720 420 500 400]);
imagesc(confMat);
colormap(flipud(gray));
colorbar;
xticks(1:3); yticks(1:3);
xticklabels(categories); yticklabels(categories);
xlabel('Predicted Label','FontSize',11);
ylabel('True Label','FontSize',11);
title('Confusion Matrix','FontSize',13,'FontWeight','bold');
for r = 1:3
    for c = 1:3
        text(c,r,num2str(confMat(r,c)),'HorizontalAlignment','center',...
             'FontSize',14,'FontWeight','bold','Color','#5B8DEE');
    end
end
box off;

%% ──────────────────────────────────────────────────────────────────
%  SECTION 6 :  SAMPLE EVALUATION
%% ──────────────────────────────────────────────────────────────────
fprintf('[5] SAMPLE EVALUATIONS\n');
fprintf('    %-12s %-18s %-12s   %-8s   %s\n',...
        'Attendance','Assignment','Test','Score','Label');
fprintf('    %s\n', repmat('-',1,62));

samples = [55 60 50; 75 70 72; 90 88 92; 40 35 30; 80 75 68];
for i = 1:size(samples,1)
    sc = evalfis(trainedFIS, samples(i,:));
    sc = min(max(sc,0),100);
    lbl = classifyPerf(sc);
    fprintf('    %-12.0f %-18.0f %-12.0f   %-8.2f   %s\n',...
            samples(i,1),samples(i,2),samples(i,3),sc,lbl{1});
end
fprintf('\n');

%% ──────────────────────────────────────────────────────────────────
%  SECTION 7 :  EXPORT FIS
%% ──────────────────────────────────────────────────────────────────
writeFIS(trainedFIS,'StudentPerformance_Trained');
fprintf('[6] Trained FIS saved → StudentPerformance_Trained.fis\n\n');
fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║               ALL TASKS COMPLETED                   ║\n');
fprintf('╚══════════════════════════════════════════════════════╝\n');

%% ── HELPERS ───────────────────────────────────────────────────────
function rgb = hex2rgb(hex)
    hex = strrep(hex,'#','');
    rgb = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] / 255;
end
