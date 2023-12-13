clear;clc;
set(gca,'FontName','Times New Rome','FontSize', 18);
policy = readmatrix("choice_type_new.csv");
reward = readmatrix("reward_new.csv");
policy = policy(3: end - 1, :);
reward = reward(3: end - 1, :);

policy_1 = find(policy(: , 2) == 1);
policy_2 = find(policy(: , 2) == 2);
policy_3 = find(policy(: , 2) == 3);
policy_4 = find(policy(: , 2) == 4);

figure(1);
hold on;
plot(reward(: , 1), reward(: , 2), "b", LineWidth=1.5);
xlabel("step"); ylabel("Total reward"); title("Total reward over time")
hold off;

figure(2);
hold on;
set(gca,'FontName','Times New Rome','FontSize', 16);

plot(policy_1, ones(length(policy_1), 1), Color='#7E2F8E', Marker='*');
plot(policy_2, 2 * ones(length(policy_2), 1), Color="#EDB120", Marker='.');
plot(policy_3, 3 * ones(length(policy_3), 1), Color="#0072BD", Marker="x");
plot(policy_4, 4 * ones(length(policy_4), 1), Color="#77AC30", Marker="+", MarkerSize=0.75);

hold off;
xlabel("step"); ylabel("Policy type");
yticks([1 2 3 4]);
yticklabels({"1", "2", "3", "4"});
legend("Imitation", "Exploitation", "Exploration", "Misguide");
