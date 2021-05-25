% All persons in

Disjunct_test = ALLCipherdata(1:200*10*8,:); % 21 %
Disjunct_train = ALLCipherdata(200*10*8+1:end,:);% 21%
% DisJuncted
%%
AllIn_test = []; %zeros(6000,325);
AllIn_train = [];% zeros(20000,325);
idx_start =0;
idx_split = 42; %
idx_end = 200;
clc;
for i =1:380
    idx_start = idx_start+1;
    idx_split = idx_start+45;
    AllIn_test=[AllIn_test; ALLCipherdata(idx_start:idx_split,:)];
    
    AllIn_train=[AllIn_train; ALLCipherdata(idx_split+1:idx_end,:)];
    
    idx_start = idx_end;
    idx_end = idx_start+200;
end