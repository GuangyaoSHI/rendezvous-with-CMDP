paths = [6.8, 19.1;
         5.83, 16.495;
         7.35, 14.48;%outside roadnetwork
         4.44, 13.993;
         1, 13.4;
         2.69, 10.62;
         5.7, 11.45;
         10, 12;%outside roadnetwork
         10.72, 10.2;%outside roadnetwork
         12.243, 8.518;
         14.507, 6.27;
         16.076, 5.845;
         13.61, 1.23;%outside roadnetwork
         16.322, 3.0;
         17.7, 1.5;%17.5, 1.5
         10, 17;
         9.5, 15;
         7, 8.9;
         12, 4
        ];

tour = [1,2,3,4,5,6,7,18,19,13,15,14,12,11,10,9,8,17,16,1];
x = paths(tour, 1);
y = paths(tour, 2);

%figure('units','normalized','outerposition',[0 0 1 1])

plot(x, y, '.', MarkerSize=42, MarkerFaceColor='r', MarkerEdgeColor='r')
%quiver(x(1:end-1), y(1:end-1), x(2:end)-x(1:end-1), y(2:end)-y(1:end-1), 'off')
hold on

axis equal
xlim([0, 20])
ylim([0, 20])

%plot road network
filename = 'Coords.csv';
M = readmatrix(filename);
%M = Coords;
M(8,:)
X = M(8:end, 2);
Y = M(8:end, 3);

ph = plot(X, Y, 'k.', MarkerSize=12);


%plot UGV task node
X_ugv = [7.23, 6.29, 17.5];
Y_ugv = [18.76, 11.14, 1.5];
hs = scatter(X_ugv, Y_ugv, 150, 's', MarkerFaceColor='b', MarkerEdgeColor='b', MarkerFaceAlpha=0.7, MarkerEdgeAlpha=0.7);

ugv_task = {'A', 'B', 'C'};
for i=1:length(X_ugv)
    text(X_ugv(i), Y_ugv(i)-0.3, ugv_task{i}, "FontSize", 25, 'Color', 'b')
end

%annotate the UAV task node
for i=1:length(paths)
    text(x(i), y(i), num2str(i-1), "FontSize", 20)
end

l = legend('UAV task nodes', 'road network',  'UGV task nodes');
l.FontSize = 22;
set(gca,'FontSize',20)
grid on

%axis tight; 
I = imread('LargeScenarioRaw.png'); 
h = image(xlim, flip(ylim), I); 
h.AlphaData = 0.5;
uistack(h,'bottom')
%saveas(gcf,'task_scenario.pdf')
% set(gcf,'Units','inches');
% screenposition = get(gcf,'Position');
% set(gcf,...
%     'PaperPosition',[0 0 screenposition(3:4)],...
%     'PaperSize',[screenposition(3:4)]);
% print(gcf, 'task_scenario.pdf', '-dpdf', '-fillpage')

ax = gca;
exportgraphics(ax,'task_scenario.pdf')
hold off

