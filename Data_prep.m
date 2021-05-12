% All persons in

Disjunct_test = Cipherdata(1:200*10*3,:);
Disjunct_train = Cipherdata(200*10*3+1:end,:);
% DisJuncted
%%
AllIn_test = []; %zeros(6000,325);
AllIn_train = [];% zeros(20000,325);
idx_start =0;
idx_split = 46;
idx_end = 200;
clc;
for i =1:130
    idx_start = idx_start+1;
    idx_split = idx_start+45;
    AllIn_test=[AllIn_test; Cipherdata(idx_start:idx_split,:)];
    
    AllIn_train=[AllIn_train; Cipherdata(idx_split+1:idx_end,:)];
    
    idx_start = idx_end;
    idx_end = idx_start+200;
end