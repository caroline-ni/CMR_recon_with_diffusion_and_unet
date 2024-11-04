function physio = read_physio(file_name)
% Before running the code, download the ISMRMRD matlab library from here:
% https://github.com/ismrmrd/ismrmrd/tree/master/matlab/%2Bismrmrd
% Last modified: 06-08-2020 by Chong Chen (Chong.Chen@osumc.edu)
%
% Input:  *.h5 file name
% Output: physio   physio{id}.data        physio wave form
%                  physio{id}.acq_time    acquisition time stamp (unit: 2.5 ms)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading an existing file %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = file_name;
if exist(filename, 'file')
    dset = ismrmrd.Dataset(filename, 'dataset');
else
    error(['File ' filename ' does not exist.  Please generate it.'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read some fields from the XML header %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We need to check if optional fields exists before trying to read them
wav = dset.readWaveform;


physio = cell(1,5); %five physio waves, 1:ECG, 3:BioMatrix
idx = cell(1,5); %five physio waves, 1:ECG, 3:BioMatrix
for wav_id = 1:5
    idx{wav_id} = find(wav.head.waveform_id == wav_id-1);
end

for wav_id = 1:5
    num_of_sig = size(idx{wav_id},2);
    tmp_data = cell(num_of_sig,1);
    for i = 1:num_of_sig
    tmp_data{i} = wav.data{idx{wav_id}(i)};
    end
    physio{wav_id}.data = cell2mat(tmp_data);
end

for wav_id = 1:5
    time_stamp_begin = double(wav.head.time_stamp(idx{wav_id}(1)));
    samp_ratio = double(floor(wav.head.number_of_samples(idx{1}(1))/wav.head.number_of_samples(idx{wav_id}(1))));
    physio{wav_id}.acq_time = time_stamp_begin + samp_ratio -1 + samp_ratio*(0:size(physio{wav_id}.data,1)-1);
end

end
