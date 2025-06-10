function [PotEn] = CalPote(x, y)

x0 = [1 0 -0.5 -1];
y0 = [0 0.5 1.5 1];

a = [-1 -1 -6.5 0.7];
b = [0 0 11 0.6];
c = [-10 -10 -6.5 0.7];
A = [-200 -100 -170 15];

PotEn = 0;

for k = 1:4
  a1(k) = a(k)*(x-x0(k))*(x-x0(k));
  b1(k) = b(k)*(x-x0(k))*(y-y0(k));
  c1(k) = c(k)*(y-y0(k))*(y-y0(k));
  V(k) = A(k)*exp(a1(k) + b1(k) + c1(k));

  PotEn = PotEn + V(k);
end

