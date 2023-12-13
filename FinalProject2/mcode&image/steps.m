clear; clc;
set(gca,'FontName','Times New Rome','FontSize', 18);
step = linspace(40, 90, 6);
reward = [622, 631, 657, 648, 649, 608];

hold on;
plot(step, reward, "o", LineWidth=1.5);
plot(step, reward, "-", LineWidth=1.5);
xlabel("step"); ylabel("reward");
hold off;