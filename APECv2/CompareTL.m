close all; clear all; clc;

set(0,'DefaultAxesFontName', 'LM Roman 12')
set(0,'DefaultAxesFontSize', 23)
set(0,'DefaultAxesFontWeight', 'Bold')
set(0,'DefaultTextFontname', 'LM Roman 12')
set(0,'DefaultTextFontSize', 23)
set(0,'DefaultTextFontWeight', 'Bold')

set(groot, 'defaultFigureColor', [1 1 1]); % White background for pictures
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
% It is not possible to set property defaults that apply to the colorbar label. 
% set(groot, 'defaultAxesFontSize', 12);
% set(groot, 'defaultColorbarFontSize', 12);
% set(groot, 'defaultLegendFontSize', 12);
% set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');
set(groot, 'defaultAxesZGrid', 'on');
set(groot, 'defaultAxesXMinorTick', 'on');
set(groot, 'defaultAxesYMinorTick', 'on');
set(groot, 'defaultAxesZMinorTick', 'on');
set(groot, 'defaultAxesXMinorGrid', 'on', 'defaultAxesXMinorGridMode', 'manual');
set(groot, 'defaultAxesYMinorGrid', 'on', 'defaultAxesYMinorGridMode', 'manual');
set(groot, 'defaultAxesZMinorGrid', 'on', 'defaultAxesZMinorGridMode', 'manual');
red = [200 36 35]/255;
blue = [40 120 181]/255;
a = 170;
gray = [a,a,a]/255;

% Begin Loading Data

% pred = load('C:\Users\sw0123\Downloads\pred.csv');
% meas = load('C:\Users\sw0123\Downloads\meas.csv');

fileLoad = load("test.csv");

x = fileLoad(:,1);
% y1 = fileLoad(:,4);
% y2 = fileLoad(:,5);
% 
% % mat78
y1 = fileLoad(:,2);
y2 = fileLoad(:,3);
semilogy(x,y2, 'Marker','.', 'LineWidth',3.5, 'MarkerSize',20, 'DisplayName','Reference Learning', 'Color','#D95319');
hold on;

semilogy(x,y1, 'Marker','.', 'LineWidth',3.5, 'MarkerSize',20, 'DisplayName','Transfer Learning','Color','#0072BD');

title('Framework Expansion to Material 78')

legend('FontSize',22);
legend boxoff;

xlabel('Amount of Training Data');
ylabel('Avg. Abs. Relative Error [\%]');
xlim([0 3600]);
xticks([0,500,1000,1500,2000,2500,3000,3500])
ylim([1 25]);
yticks([1,2,3,4,5,10,15,20])
% mean(err3)
set(gca, 'box', 'on')
set(gcf,'Position',[850,550,780,430])