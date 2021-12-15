function  [] = plot_coordinate(R, name, offset)
%R is rotation matrix
ex = [1;0;0];
ey = [0;1;0];
ez = [0;0;1];
%plot fixed/world coordinate
quiver3(0, 0, 0, 1, 0, 0, 'r')
text(1, 0, 0, 'x')
hold on
quiver3(0, 0, 0, 0, 1, 0, 'g')
text(0, 1, 0, 'y')
quiver3(0, 0, 0, 0, 0, 1, 'b')
text(0, 0, 1, 'z')
%plot new coordinate
ex_new = R*ex;
ey_new = R*ey;
ez_new = R*ez;
text(offset(1), offset(2), offset(3), name)
quiver3(offset(1), offset(2), offset(3), ex_new(1), ex_new(2), ex_new(3), 'r')
text(offset(1)+ex_new(1), offset(2)+ex_new(2), offset(3)+ex_new(3), "x"+name)

quiver3(offset(1), offset(2), offset(3), ey_new(1), ey_new(2), ey_new(3), 'g')
text(offset(1)+ey_new(1), offset(2)+ey_new(2), offset(3)+ey_new(3), "y"+name)

quiver3(offset(1), offset(2), offset(3), ez_new(1), ez_new(2), ez_new(3), 'b')
text(offset(1)+ez_new(1), offset(2)+ez_new(2), offset(3)+ez_new(3), "z"+name)
set(gca, 'ZDir','reverse')
end