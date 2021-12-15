%https://docs.modalai.com/configure-extrinsics/
%in the configuration file of voxl
%they use intrinsic XYZ representation
%rotate Z first, then Y, then X
%[X, Y, Z]
%this is different from ZYX yaw-pitch-roll convention
x_angle = 0;
Rx = rotx(x_angle/pi*180)
y_angle = pi/4;
Ry = roty(y_angle/pi*180)
z_angle = pi/2;
Rz = rotz(z_angle/pi*180)

%eul = [z_angle, y_angle, x_angle];
%rotmZYX = eul2rotm(eul)
R = Rx*Ry*Rz;
plot_coordinate(R, "camera", [1, 1, 0])


eul = [pi/2 0 0];
rotmZYX = eul2rotm(eul);
plot_coordinate(rotmZYX, "test", [2, 2, 0])

eul = [pi/2 pi/3 pi/2];
rotmZYX = eul2rotm(eul);
plot_coordinate(rotmZYX, "test", [3, 3, 0])