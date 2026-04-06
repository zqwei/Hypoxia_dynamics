close all
clear all

dirs = {'fish01-7dpf-huc-h2b-gc7f-gfap-jRGECO1b-ZTS2-O2-measurement_20221118_165508';
        'fish03-7dpf-gfap-gc6f-ZTS1-oxygen-OMR_20220728_202930';
        'fish03-gfap-gc6f-gfap-bARK-D110-ZTS1-oxygen-OMR_20220909_193049';
        'fish04-7dpf-gfap-gc6f-ZTS1-oxygen-OMR_20220728_223709';
        'fish04-gfap-gc6f-gfap-bARK-ZTS1-oxygen-OMR_20220909_221356';
        'fish05-gfap-gc6f-gfap-bARK-ZTS1-oxygen-OMR_20220910_001528'};

swim_vigor_list = [];
glia_list = [];
O2_trial_list = [];
epoch_list = [];
swim_durations_list = [];
anm_list = [];

norm_on = 5*60*3;
norm_off = 20*60*3;
hypo_on = 30*60*3;
hypo_off = 40*60*3;
pre_ = 10;
post_ = 80;



for nfile = 1:6
    load([dirs{nfile}, '/O2_ds.mat'])
    % load([dirs{nfile}, '/F.mat'])
    load([dirs{nfile}, '/ds_EMG.mat'])
    load([dirs{nfile}, '/swim_ds.mat'])
    F = readmatrix([dirs{nfile}, '/LMO_F_side_view_video.csv']);
    F = F(:, 2);
    
    disp(mean(O2_ds(norm_on:norm_off)))
    F = zscore(F);
    O2_ds = O2_ds/mean(O2_ds(norm_on:norm_off));

    swim_on = find((swim_ds(1:end-1)==0) & (swim_ds(2:end)>0));
    swim_off = find((swim_ds(1:end-1)>0) & (swim_ds(2:end)==0));    
    % swim_duration = swim_off - swim_on;
    swim_off = swim_off(swim_off>swim_on(1));
    inter_swim_interval = swim_on(2:end) - swim_off(1:end-1);
    swim_index = [inter_swim_interval', 40]>10;
    swim_ons = swim_on(swim_index);
    num_trials = length(swim_ons);
    swim_offs = zeros(num_trials, 1);
    for n = 1:num_trials-1
        swim_offs(n) = max(swim_off(swim_off<swim_ons(n+1)));
    end
    swim_offs(end) = swim_off(end);
    swim_durations = swim_offs-swim_ons;
    
    epoch = zeros(num_trials-2, 1);
    swim_vigor = zeros(num_trials-2, pre_+post_);
    glia = zeros(num_trials-2, pre_+post_);
    O2_trial = zeros(num_trials-2, pre_+post_);
    swim_durations_ = swim_durations(1:num_trials-2);
    
    for n = 1:num_trials-2
        swim_ons_ = swim_ons(n);
        if (swim_ons_>norm_on) && (swim_ons_<norm_off)
            epoch(n) = 1;
        elseif (swim_ons_>hypo_on) && (swim_ons_<hypo_off)
            epoch(n) = 2;
        end    
        swim_vigor(n, :) = swim_ds(swim_ons_-pre_:swim_ons_+post_-1);
        glia(n, :) = F(swim_ons_-pre_:swim_ons_+post_-1);
        O2_trial(n, :) = O2_ds(swim_ons_-pre_:swim_ons_+post_-1);
    end

    swim_vigor_list = [swim_vigor_list; swim_vigor]; %#ok<AGROW>
    glia_list = [glia_list; glia]; %#ok<AGROW>
    O2_trial_list = [O2_trial_list; O2_trial]; %#ok<AGROW>
    epoch_list = [epoch_list; epoch]; %#ok<AGROW>
    swim_durations_list = [swim_durations_list; swim_durations_]; %#ok<AGROW>
    anm_list = [anm_list; zeros(size(swim_durations_))+nfile]; %#ok<AGROW>

end

save('oxygen_glia.mat', "anm_list", ...
                        "swim_durations_list", ...
                        "epoch_list", ...
                        "O2_trial_list", ...
                        "glia_list", ...
                        "swim_vigor_list");


glia_ = glia_list - mean(glia_list(:, 1:pre_), 2);
O2_trial_ = O2_trial_list - mean(O2_trial_list(:, 1:pre_+3), 2);
pre_swim = sum(swim_vigor_list(:, 1:pre_), 2);

figure;
idx_ = (swim_durations_list>0) & (swim_durations_list<300) & (epoch_list==1) ; % & (pre_swim<0.3)
disp(sum(idx_))
time_ = (-pre_:post_-1)/3;
subplot(3,1,1);
hold on
plot(time_, mean(glia_(idx_, :), 1), '-k');
xlabel('Time (s)')
ylabel('Glial F')
subplot(3,1,2);
hold on
plot(time_, mean(O2_trial_(idx_, :), 1), '-k')
xlabel('Time (s)')
ylabel('\Delta O2')
subplot(3,1,3);
hold on
plot(time_, mean(swim_vigor_list(idx_, :), 1), '-k')
xlabel('Time (s)')
ylabel('Swim')


idx_ = (swim_durations_list>0) & (swim_durations_list<300) & (epoch_list==2) ; % & (pre_swim<0.3)
disp(sum(idx_))
subplot(3,1,1);
plot(time_, mean(glia_(idx_, :), 1), '-r');
xlim([-pre_/3, 25])
subplot(3,1,2);
plot(time_, mean(O2_trial_(idx_, :), 1), '-r')
xlim([-pre_/3, 25])
subplot(3,1,3);
hold on
plot(time_, mean(swim_vigor_list(idx_, :), 1), '-r')
xlim([-pre_/3, 25])
