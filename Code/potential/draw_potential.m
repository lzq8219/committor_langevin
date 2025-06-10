xmin = -1.5;
xmax = 1.2;
ymin = -0.2;
ymax = 2;
nx = 80;
ny = 80;


a = [-0.313, 0.971, 0.218, 0.292];
h = 0;
l = 1;

x = linspace(xmin,xmax,nx+1);
y = linspace(ymin,ymax,ny+1);
[X, Y] = meshgrid(x, y);
V = zeros(size(X));
DX = zeros(size(X));
DY = zeros(size(X));

scale=1;
for i = 1:size(X,1)
    for j = 1:size(X,2)
        V(i,j) = potential_V(X(i,j), Y(i,j), a, h, l,scale);
        DX(i,j) = b1(X(i,j), Y(i,j), a, h, l,scale);
        DY(i,j) = b2(X(i,j), Y(i,j), a, h, l,scale);
    end
end

% [px, py] = gradient(V, .3, .2);

figure;
contour(X,Y,min(V,200),40); colorbar;
%hold on;
%row = 1:5:nx;
%quiver(X(row,row), Y(row,row), DX(row,row), DY(row,row),'color','k');

% colormap gray;