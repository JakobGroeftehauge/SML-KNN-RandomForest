clear all;
clc;

%% Knn All in 
load('Data\AllIn_test.mat')
load('Data\AllIn_train.mat')

% set up
lab_train = AllIn_train(:,1);
dat_train = AllIn_train(:,2:end);

lab_test = AllIn_test(:,1);
dat_test = AllIn_test(:,2:end);
%%
k_val = 1:10;
err_rate = zeros(length(k_val),1);
err_rate_train = zeros(length(k_val),1);
for k =1:length(k_val)
    disp(k)
    [idx, d] = knnsearch(dat_train, dat_test, 'K', k_val(k), 'NSMethod', 'kdtree');
    [idx_train, d_train] = knnsearch(dat_train, dat_train, 'K', k_val(k), 'NSMethod', 'kdtree');
    
    labels_pred = mode(lab_train(idx)')';
    bool = lab_test == labels_pred;
    err_rate(k) = 1-sum(bool) / length(bool);
    
    labels_pred_train = mode(lab_train(idx_train)')';
    bool_train = lab_train == labels_pred_train;
    err_rate_train(k) = 1-sum(bool_train) / length(bool_train);
end
%%
figure(1)
plot(k_val, err_rate)
hold on
plot(k_val, err_rate_train)
title("Elbow curve - All persons in")
xlabel('k-values')
ylabel('Error rate')
legend("Test","Train");



%% Knn Disjunct 
disp("DISJUNCT!")
load('Data\Disjunct_train.mat')
load('Data\Disjunct_test.mat')

% set up
dis_lab_train = Disjunct_train(:,1);
dis_dat_train = Disjunct_train(:,2:end);

dis_lab_test = Disjunct_test(:,1);
dis_dat_test = Disjunct_test(:,2:end);
%%
k_val = 1:10;
dis_err_rate = zeros(length(k_val),1);
dis_err_rate_train = zeros(length(k_val),1);
for k =1:length(k_val)
    disp(k)
    [idx, d] = knnsearch(dis_dat_train, dis_dat_test, 'K', k_val(k), 'NSMethod', 'kdtree');
    [idx_train, d_train] = knnsearch(dis_dat_train, dis_dat_train, 'K', k_val(k), 'NSMethod', 'kdtree');
    
    labels_pred = mode(dis_lab_train(idx)')';
    dis_bool = dis_lab_test == labels_pred;
    dis_err_rate(k) = 1-sum(dis_bool) / length(dis_bool);
    
    labels_pred_train = mode(dis_lab_train(idx_train)')';
    dis_bool_train = dis_lab_train == labels_pred_train;
    dis_err_rate_train(k) = 1-sum(dis_bool_train) / length(dis_bool_train);
end
%%
figure(2)
plot(k_val, dis_err_rate)
hold on
plot(k_val, dis_err_rate_train)
title("Elbow curve - Disjunct")
xlabel('k-values')
ylabel('Error rate')
legend("Test","Train");

