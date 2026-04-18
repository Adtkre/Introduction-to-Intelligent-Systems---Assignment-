%% =========================================================
%  WASHING MACHINE FUZZY LOGIC SYSTEM
%  Inputs  : Dirt Level, Load Size
%  Output  : Cycle Time (minutes)
%  Toolbox : MATLAB Fuzzy Logic Toolbox
%  Author  : Generated via Claude / Anthropic
%% =========================================================

clear; clc; close all;

%% ── 1. CREATE FIS ─────────────────────────────────────────
fis = mamfis('Name', 'WashingMachine');

%% ── 2. INPUT 1 : DIRT LEVEL (0–10) ───────────────────────
fis = addInput(fis, [0 10], 'Name', 'DirtLevel');

fis = addMF(fis, 'DirtLevel', 'trimf', [0  0  5],  'Name', 'Low');
fis = addMF(fis, 'DirtLevel', 'trimf', [2  5  8],  'Name', 'Medium');
fis = addMF(fis, 'DirtLevel', 'trimf', [5 10 10],  'Name', 'High');

%% ── 3. INPUT 2 : LOAD SIZE (0–10) ────────────────────────
fis = addInput(fis, [0 10], 'Name', 'LoadSize');

fis = addMF(fis, 'LoadSize', 'trimf', [0  0  5],  'Name', 'Small');
fis = addMF(fis, 'LoadSize', 'trimf', [2  5  8],  'Name', 'Medium');
fis = addMF(fis, 'LoadSize', 'trimf', [5 10 10],  'Name', 'Large');

%% ── 4. OUTPUT : CYCLE TIME (20–120 min) ──────────────────
fis = addOutput(fis, [20 120], 'Name', 'CycleTime');

fis = addMF(fis, 'CycleTime', 'trimf', [20  20  50],  'Name', 'Short');
fis = addMF(fis, 'CycleTime', 'trimf', [30  55  80],  'Name', 'Medium');
fis = addMF(fis, 'CycleTime', 'trimf', [60  90  120], 'Name', 'Long');
fis = addMF(fis, 'CycleTime', 'trimf', [90 120 120],  'Name', 'VeryLong');

%% ── 5. FUZZY RULES (min 6) ───────────────────────────────
%  [DirtLevel  LoadSize  CycleTime  Weight  Connection(1=AND,2=OR)]
ruleList = [
    1 1 1 1 1;   % R1 : Low   & Small  → Short
    1 2 1 1 1;   % R2 : Low   & Medium → Short
    1 3 2 1 1;   % R3 : Low   & Large  → Medium
    2 1 2 1 1;   % R4 : Medium& Small  → Medium
    2 2 2 1 1;   % R5 : Medium& Medium → Medium
    2 3 3 1 1;   % R6 : Medium& Large  → Long
    3 1 3 1 1;   % R7 : High  & Small  → Long
    3 2 3 1 1;   % R8 : High  & Medium → Long
    3 3 4 1 1;   % R9 : High  & Large  → VeryLong
];

fis = addRule(fis, ruleList);

%% ── 6. EVALUATE A SAMPLE POINT ───────────────────────────
dirtVal = 7;   % High dirt
loadVal = 8;   % Large load
cycleResult = evalfis(fis, [dirtVal, loadVal]);
fprintf('\n=== SAMPLE EVALUATION ===\n');
fprintf('  Dirt Level : %.1f / 10\n', dirtVal);
fprintf('  Load Size  : %.1f / 10\n', loadVal);
fprintf('  Cycle Time : %.1f minutes\n\n', cycleResult);

%% ── 7. PLOTS ──────────────────────────────────────────────
figure('Name','Membership Functions','Color','white','Position',[100 100 1200 380]);

% --- Dirt Level MFs ---
ax1 = subplot(1,3,1);
x = linspace(0,10,500);
colors = {'#1E90FF','#FF8C00','#DC143C'};
labels = {'Low','Medium','High'};
hold on;
for k = 1:3
    mfParams = fis.Inputs(1).MembershipFunctions(k).Parameters;
    y = trimfMF(x, mfParams);
    plot(x, y, 'Color', colors{k}, 'LineWidth', 2.5, 'DisplayName', labels{k});
    area(x, y, 'FaceColor', colors{k}, 'FaceAlpha', 0.08, 'EdgeColor', 'none');
end
hold off;
xlabel('Dirt Level'); ylabel('Degree of Membership');
title('Input 1 — Dirt Level','FontWeight','bold');
legend('Location','northeast','FontSize',8); grid on; ylim([0 1.1]);
set(ax1,'FontSize',10,'Box','off');

% --- Load Size MFs ---
ax2 = subplot(1,3,2);
labels2 = {'Small','Medium','Large'};
hold on;
for k = 1:3
    mfParams = fis.Inputs(2).MembershipFunctions(k).Parameters;
    y = trimfMF(x, mfParams);
    plot(x, y, 'Color', colors{k}, 'LineWidth', 2.5, 'DisplayName', labels2{k});
    area(x, y, 'FaceColor', colors{k}, 'FaceAlpha', 0.08, 'EdgeColor', 'none');
end
hold off;
xlabel('Load Size'); ylabel('Degree of Membership');
title('Input 2 — Load Size','FontWeight','bold');
legend('Location','northeast','FontSize',8); grid on; ylim([0 1.1]);
set(ax2,'FontSize',10,'Box','off');

% --- Cycle Time MFs ---
ax3 = subplot(1,3,3);
x2 = linspace(20,120,500);
colors4 = {'#1E90FF','#FF8C00','#DC143C','#8B008B'};
labels3 = {'Short','Medium','Long','VeryLong'};
hold on;
for k = 1:4
    mfParams = fis.Outputs(1).MembershipFunctions(k).Parameters;
    y = trimfMF(x2, mfParams);
    plot(x2, y, 'Color', colors4{k}, 'LineWidth', 2.5, 'DisplayName', labels3{k});
    area(x2, y, 'FaceColor', colors4{k}, 'FaceAlpha', 0.08, 'EdgeColor', 'none');
end
hold off;
xlabel('Cycle Time (min)'); ylabel('Degree of Membership');
title('Output — Cycle Time','FontWeight','bold');
legend('Location','northeast','FontSize',8); grid on; ylim([0 1.1]);
set(ax3,'FontSize',10,'Box','off');

sgtitle('Washing Machine Fuzzy Logic — Membership Functions', ...
        'FontSize',14,'FontWeight','bold');

%% ── 8. OUTPUT SURFACE ────────────────────────────────────
figure('Name','Output Surface','Color','white','Position',[100 550 700 500]);
[D, L] = meshgrid(linspace(0,10,40), linspace(0,10,40));
Z = zeros(size(D));
for i = 1:numel(D)
    Z(i) = evalfis(fis, [D(i), L(i)]);
end
surf(D, L, Z, 'EdgeColor','none','FaceAlpha',0.9);
colormap(turbo); colorbar;
xlabel('Dirt Level','FontSize',11);
ylabel('Load Size','FontSize',11);
zlabel('Cycle Time (min)','FontSize',11);
title('Fuzzy Output Surface','FontSize',14,'FontWeight','bold');
view(45, 30); grid on;

%% ── 9. RULE VIEWER (built-in GUI) ────────────────────────
ruleview(fis);

%% ── 10. EXPORT FIS FILE ──────────────────────────────────
writeFIS(fis, 'WashingMachine');
fprintf('FIS saved → WashingMachine.fis\n');

%% ── LOCAL HELPER ──────────────────────────────────────────
function y = trimfMF(x, params)
    a = params(1); b = params(2); c = params(3);
    y = max(min((x-a)/(b-a+eps), (c-x)/(c-b+eps)), 0);
end
