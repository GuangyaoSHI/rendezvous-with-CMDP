function plot_handle = plot_grid_world(row, column)
X = []
Y = []
for i = 1:row
    for j = 1:column
        X = [X, i];
        Y = [Y, j];
    end
end
plot_handle = plot(X, Y, 'b.', 'MarkerSize', 20)
grid on
hold on 
plot([4, 4, 4], [1, 2, 3], 'r.', 'MarkerSize', 20)
text(7, 1+0.3, 'goal')
plot(7, 1, 'mx')
text(1, 1+0.3, 'start')
plot(1, 1, 'mx')
end