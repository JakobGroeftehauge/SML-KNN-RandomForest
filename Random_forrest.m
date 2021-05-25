%% Import data - All in 
clc; clear;

load('Data\AllIn_test.mat')
load('Data\AllIn_train.mat')

% set up
lab_train_all = AllIn_train(:,1);
dat_train_all = AllIn_train(:,2:end);

lab_test_all = AllIn_test(:,1);
dat_test_all = AllIn_test(:,2:end);

% Random Forrest - All In 

N_test = 30; 

accuracy_train = zeros(N_test, 1); 
accuracy_test  = zeros(N_test, 1); 

for i = 1:N_test
    random_forrest = TreeBagger(i, dat_train_all, lab_train_all,  'Method','classification', 'OOBPrediction','On', 'MinLeafSize', 15, 'MaxNumSplits', 300, 'NumPredictorsToSample', log2(324), 'Prior','Uniform'); 
    pred_labels = predict(random_forrest, dat_test_all); 

    bool = lab_test_all == str2num(cell2mat(pred_labels));
    err_rate_test = 1-sum(bool) / length(bool);
    accuracy_test(i) = err_rate_test; 
    
    pred_labels = predict(random_forrest, dat_train_all); 
    bool = lab_train_all == str2num(cell2mat(pred_labels));
    err_rate_train = 1-sum(bool) / length(bool);
    accuracy_train(i) = err_rate_train; 
end
figure(1)
plot(accuracy_test); 
hold on 
plot(accuracy_train); 
legend('test', 'train')

hold off

%% Import Data -  Disjunct 
clear; 
load('Data\Disjunct_train.mat')
load('Data\Disjunct_test.mat')

% set up
lab_train_dis = Disjunct_train(:,1);
dat_train_dis = Disjunct_train(:,2:end);

lab_test_dis = Disjunct_test(:,1);
dat_test_dis = Disjunct_test(:,2:end);

N_test = 20; 
accuracy_train = zeros(N_test, 1); 
accuracy_test  = zeros(N_test, 1); 

for i = 1:N_test
    random_forrest = TreeBagger(i, dat_train_dis, lab_train_dis,  'Method','classification', 'CategoricalPredictors', 'all', 'OOBPrediction','On', 'MinLeafSize', 10, 'MaxNumSplits', 300, 'NumPredictorsToSample', 10, 'Prior','Uniform'); 
    
    pred_labels = predict(random_forrest, dat_test_dis); 
    bool = lab_test_dis == str2num(cell2mat(pred_labels));
    err_rate_test = 1-sum(bool) / length(bool);
    accuracy_test(i) = err_rate_test; 
    
     pred_labels = predict(random_forrest, dat_train_dis); 
     bool = lab_train_dis == str2num(cell2mat(pred_labels));
     err_rate_train = 1-sum(bool) / length(bool);
     accuracy_train(i) = err_rate_train; 
end

figure(2)
plot(accuracy_test); 
hold on 
plot(accuracy_train); 
legend('test', 'train')

hold off
