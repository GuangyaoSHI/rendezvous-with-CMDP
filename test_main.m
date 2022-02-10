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
lengths = [];

for i = 2:size(paths, 1)
    lengths = [lengths, norm(paths(i, :)-paths(i-1, :))];
end

disp(lengths)
x = paths(:, 1);
y = paths(:, 2);

figure(2)
plot(x, y, '-*')
axis equal



%UGV task
UGV_task = [6.8, 19.1;
            5.46, 15.32;
            4.04, 13.13;
            6.29, 11.14;
            10.4, 8.35;
            14.52, 4.53;, 
            17.5, 1.5];

figure(6)
c = 0:0.1:1;
obj = [3264.5, 3119.2, 3001.5, 2886.2, 2770.8, 2655.5, 2540.1, 2428.3, 2317.8, 2207.3, 2096.8];
plot(c, obj)

