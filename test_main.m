filename = 'Coords.csv';
M = readmatrix(filename)
%M = Coords;
M(8,:)
X = M(8:end, 2);
Y = M(8:end, 3);
figure(1)
plot(X, Y, '*-')
axis equal

paths = [6.8, 19.1;
         5.83, 16.495;
         7.35, 14.48;%outside roadnetwork
         4.44, 13.993;
         1, 13.4;
         2.69, 10.62;
         5.7, 11.45;
         10, 12;%outside roadnetwork
         9.72, 9.2;%outside roadnetwork
         11.243, 7.518;
         12.507, 6.27;
         14.076, 4.845;
         13.61, 1.23;%outside roadnetwork
         16.322, 3.549;
         17.5, 1.5;
        ];


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
         17.5, 1.5;
         10, 17;
         9.5, 15;
         7, 8.9;
         12, 4
        ];

writematrix(paths,'UAV_task_nodes.txt','Delimiter',',')

lengths = [];

for i = 2:size(paths, 1)
    lengths = [lengths, norm(paths(i, :)-paths(i-1, :))];
end

disp(lengths)
tour = [1,2,3,4,5,6,7,18,19,13,15,14,12,11,10,9,8,17,16,1,2,3,4,5,6,7,18,19,13,15,14,12,11,10,9,8,17,16, 1];
x = paths(tour, 1);
y = paths(tour, 2);

figure(2)
%plot(x, y, '-*')
quiver(x(1:end-1), y(1:end-1), x(2:end)-x(1:end-1), y(2:end)-y(1:end-1), 'off')
hold on
for i=1:length(paths)
    text(paths(i, 1), paths(i, 2), num2str(i))
end
axis equal
hold off


%UGV task
UGV_task = [6.8, 19.1;
            5.46, 15.32;
            4.04, 13.13;
            6.29, 11.14;
            10.4, 8.35;
            14.52, 4.53;
            17.5, 1.5];

figure(6)
c = 0:0.1:1;
obj = [3264.5, 3119.2, 3001.5, 2886.2, 2770.8, 2655.5, 2540.1, 2428.3, 2317.8, 2207.3, 2096.8];
plot(c, obj)
title('threshold vs travel time')
xlabel('threshold')
ylabel('Ojbetive for LP')






t = [198.55247854079758,
 257.55201089754235,
 301.0682943578669,
 249.33840008129923,
 232.38448962959677,
 223.02420478138097,
 309.64512274669585,
 200.99751242241783,
 162.07602461530837,
 126.87773320671154];

[0.05, 0.25, 0.5]
[ 19642, 15519.5, 15482.27]

[0.03, 0.2, 0.45]
[8625,6037, 4842]