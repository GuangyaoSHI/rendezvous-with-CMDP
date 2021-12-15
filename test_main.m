for i = 1:size(paths, 1)
    plot_grid_world(7, 7);
    hold on
    plot(paths(i, 1)+1, paths(i, 2)+1, 'mo','MarkerSize', 40);
    title("t = " + string(i))
    hold off
    pause(1);
end

