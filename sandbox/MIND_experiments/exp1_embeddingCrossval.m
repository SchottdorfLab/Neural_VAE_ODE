%% Example code to embed E65 in various dimensions, and measure R^2 with cross-validation

close all;
clear all;

load('./E65.mat') % a struct with data & behavior

rng(42);

trials_all = [9,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,53,55,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210];

randlist = rand(length(trials_all),1)>0.1;

trials_train = trials_all(randlist);
trials_test = trials_all(~randlist);

 
ROIactivities = nic_output.ROIactivities;
[T,N] = size(ROIactivities);

Datarange    = sum(ROIactivities,2)>0;
Neurons      = sum(ROIactivities,1)>0;
all_data  = ROIactivities(Datarange, Neurons);
trialn = nic_output.behavioralVariables.Trial(Datarange);

training_data = all_data(ismember(trialn, trials_train), :);
testing_data = all_data(ismember(trialn, trials_test), :);


%% Set up the mind algorithm
mindparameters.dt            = 1;               % step distance to the past
mindparameters.pca.n         = 0.95;            % keep data that explains 95% variance
mindparameters.dim_criterion = 0.95;
mindparameters.ndir          = 20;              % number of hyperplanes tested
mindparameters.min_leaf_pts  = 500;             % minimum number of leaves
mindparameters.ntrees        = 100;
mindparameters.verbose       = true;
mindparameters.lm.lmf        = 1;               % number of landmarks
mindparameters.rwd.type      = 'discrete';
mindparameters.rwd.sym       = 'avg';
mindparameters.rwd.all_geo   = true;
mindparameters.rwd.d         = 2;
mindparameters.rwd.var_scale = 0.1;
mindparameters.embed.type    = 'rwe';
mindparameters.embed.d       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
mindparameters.embed.mode    = 'mds';
mindparameters.embed.local   = false;
mindparameters.embed.opts    = statset('MaxIter',400);   % Iterations for MDS
mindparameters.learnmapping  = true;
if size(training_data,1)>100
    mindparameters.mapping.k = [5:20, 12:5:50];      % range to get optimal values
else
    mindparameters.mapping.k = [5:20, 5:3:30];       % range to get optimal values
end
mindparameters.mapping.lambda     = [10.^(-8:.5:0)]; % range to get optimal values
mindparameters.mapping.mode       = 'lle';
mindparameters.mapping.nfolds_lle = 10;               
mindparameters.prune_lm_by_time   = false;
mindparameters.lm.n  = round(sum(Datarange)/mindparameters.lm.lmf);

%% Run Mind
data = struct();
times = reshape(1:size(training_data,1), size(training_data,1),1)./15;
data.t = times;
data.f = training_data;

dat = struct();
dat.forestdat = mindAsFunction(data, mindparameters);
dat.mindparameters = mindparameters;

[~, dat.allembed] = embedAsFunction(dat.forestdat, mindparameters);
fprintf('finished running embedAsFunction\n');

%% Plot the data

figure(1)
for d_idx = 1:length(mindparameters.embed.d)
    raw_data = testing_data;
    
    y = dat.allembed(d_idx).f2m.map.transform(dat.forestdat.pca.model.transform(raw_data, mindparameters.pca.n));
    reconstructed_data = dat.forestdat.pca.model.inverse_transform(dat.allembed(d_idx).m2f.map.transform(y));
    corrcoef(raw_data(:), reconstructed_data(:))
    
    plot(mindparameters.embed.d(d_idx), 1-var(raw_data(:) - reconstructed_data(:)) / var(raw_data(:) ),'o','MarkerFaceColor','r', 'MarkerEdgeColor','r')
    hold on;
 
    for tt  = trials_test
        single_trial = all_data(ismember(trialn, tt), :);
        y = dat.allembed(d_idx).f2m.map.transform(dat.forestdat.pca.model.transform(single_trial, mindparameters.pca.n));
        reconstructed_data = dat.forestdat.pca.model.inverse_transform(dat.allembed(d_idx).m2f.map.transform(y));
        corrcoef(single_trial(:), reconstructed_data(:))
        plot(mindparameters.embed.d(d_idx), 1-var(single_trial(:) - reconstructed_data(:)) / var(single_trial(:) ),'ko')
        hold on;
    end

end

xlim([0,10])
ylim([0,1])
xlabel("Embedding dimension")
ylabel("Crossval R^2")