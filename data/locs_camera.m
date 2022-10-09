%% camera location file in ephys

opts = delimitedTextImportOptions('Delimiter', ',', 'DataLines', 2, 'VariableNamesLine', 1);
table_ = readtable('datalist.csv', opts);

Fs=6000;
H = 1;
Fs_IM = 3.04;
MinPeakDistance = floor(Fs/Fs_IM)-100;

for n = 6:9
    Folder = [table_(n,:).('dir_'){1}, '/ephys/analysis/'];
    % if exist([Folder, 'locs_cam.mat'], 'file')
    %     continue
    % else
        disp(Folder)
        load([Folder, 'x3.mat']);
        [pks_cam,locs_cam] = findpeaks(abs(diff(x3)),'MinPeakHeight',H,'MinPeakProminence',H,'MinPeakDistance',MinPeakDistance,'Annotate','extents'); %find EEG_down peak
        save([Folder, 'locs_cam.mat'], 'locs_cam','-v7.3')
    % end
end