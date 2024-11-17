% This script processes a folder of ISMRMD *.h5 files, extracts 5D k-space data 
% with the dimensions ['kx', 'ky', 'coil', 'slice', 'phase'], and saves it as a .mat file.
% Last modified: YourName, Date

clear;
%close all;
% restoredefaultpath

% Specify the folder containing .h5 files
dataFolder = 'C:\path\to\your\h5\folder'; % Change this to your folder path
outputFolder = 'C:\Users\carol\Desktop\UGthesis_cMRIrecon\unsupervised_MRIrecon\matlab_scripts\fsr'; % Change this to your output folder path

% Get a list of all .h5 files in the folder
h5Files = dir(fullfile(dataFolder, '*.h5'));

% Loop over each .h5 file in the folder
for fileIdx = 1:length(h5Files)
    % Get the current file name and path
    fileName = h5Files(fileIdx).name;
    filePath = fullfile(dataFolder, fileName);

    % Display progress
    fprintf('Processing file %s...\n', fileName);

    % Call the function to read the data
    try
        % Read k-space data
        [kData, param, acqOrder] = read_ocmr(filePath);
        
        % Reduce kData to 5D: ['kx', 'ky', 'coil', 'slice', 'phase']
        kData5D = permute(kData(:, :, :, :, :, 1, :, :), [1, 2, 4, 7, 5]);
        % Here, permute is used to rearrange to [kx, ky, coil, slice, phase]

        % Save kData as a .mat file with the same base name as the .h5 file
        matFileName = fullfile(outputFolder, [fileName(1:end-3), '_5D.mat']);
        save(matFileName, 'kData5D', 'param', 'acqOrder');
        
        % Log successful processing
        fprintf('Saved 5D kData for %s as %s\n', fileName, matFileName);
        
        % Additional processing can be done here if needed
    catch ME
        % Log any errors encountered
        warning('Failed to process %s: %s', fileName, ME.message);
    end
end

fprintf('All files have been processed.\n');
%% 


