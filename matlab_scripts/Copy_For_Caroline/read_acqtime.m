function acq_info = read_acqtime(file_name)
% Before running the code, download the ISMRMRD matlab library from here:
% https://github.com/ismrmrd/ismrmrd/tree/master/matlab/%2Bismrmrd
% Last modified: 06-08-2020 by Chong Chen (Chong.Chen@osumc.edu)
%
% Input:  *.h5 file name
% Output: acq_info   


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
hdr = ismrmrd.xml.deserialize(dset.readxml);

%% Discard Pilot Tone signal, if presenet
    disp('Loading k-space!')
    D = dset.readAcquisition();
    % Check noise scan
    isNoise = D.head.flagIsSet('ACQ_IS_NOISE_MEASUREMENT');
    noiseScan = find(isNoise==0,1,'first') - 1 ;
    acq_info.slice_offset = zeros(D.getNumber - noiseScan, 1);
    acq_info.ECG_trigger = zeros(D.getNumber - noiseScan, 1);
    disp('Extracting acq info!')
    for p = (noiseScan+1):D.getNumber
        acq_info.slice_offset(p-noiseScan) = D.head.position(:,p)'*D.head.slice_dir(:,p);
        acq_info.ECG_trigger(p-noiseScan) = D.head.physiology_time_stamp(1,p);
    end    
    % Discard noise scan
    acq_info.flags = D.head.flags(noiseScan+1:end);
    acq_info.slic = D.head.idx.slice(noiseScan+1:end);
    acq_info.acq_time = D.head.acquisition_time_stamp(noiseScan+1:end);
    acq_info.sequenceParameters = hdr.sequenceParameters;


end
