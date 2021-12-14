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
end