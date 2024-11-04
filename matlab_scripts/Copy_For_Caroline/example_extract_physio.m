% This is an example of Matlab script to read ECG waveform from ISMRMD *.h5
% Last modified: 01-03-2022 by Chong Chen (Chong.Chen@osumc.edu)
% The example fully sampled dataset 'fs_0005_1_5T.h5' can be downloaded here: https://ocmr.s3.us-east-2.amazonaws.com/data/fs_0005_1_5T.h5
% The example undersampled dataset 'us_0165_pt_1_5T.h5' can be downloaded here: https://ocmr.s3.us-east-2.amazonaws.com/data/us_0165_pt_1_5T.h5 

% clear
% close all;
% restoredefaultpath

[fileName,dirName,FilterIndex] = uigetfile('*.h5','MultiSelect', 'on');
for k = 1:size(cellstr(fileName),2)
    if iscell(fileName)
        filename = fileName{k};
    else
        filename = fileName;
    end
    
    %% read the data
    physio = read_physio([dirName filename]);
    acq_info = read_acqtime([dirName filename]);    
    
    ref_time_point = physio{1}.acq_time(1);
    kspace_acq_time = double(acq_info.acq_time - ref_time_point)*2.5; % unit: 2.5 ms
    
    %% show the ECG signal
    physio_id = 1;
    ECG_trigger = (physio{physio_id}.acq_time(physio{physio_id}.data(:,5) > 0) - ref_time_point)*2.5;
    figure;
    for ECG_chanel = 1:4
        ECG_signal = physio{1}.data(:,ECG_chanel);
        subplot(4,1,ECG_chanel)
        ECG_acq_time = (physio{physio_id}.acq_time - ref_time_point)*2.5; %Time of acquisition in ms (2.5ms apart)        
        plot(ECG_acq_time,ECG_signal);
        hold on; plot(ECG_trigger, mean(ECG_signal),'r*') % display the ECG trigger point
        plot([kspace_acq_time(1),kspace_acq_time(end)],[min(ECG_signal),min(ECG_signal)],'b-'); % display the k-space acq window
        title(['ECG chanel:' num2str(ECG_chanel)]);
    end
    xlabel('acq time stamp [ms]');    
    
end

