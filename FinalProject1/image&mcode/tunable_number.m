x=linspace(1/sqrt(2),sqrt(2),10);
reward=[0.0 0.6 0.2 -0.2 -0.2 0.2 0.2 0.4 0.8 0.0];

plot(x,reward,"r-*",LineWidth=1.5);
xlabel("Tunable number");
ylabel("Mean reward");
