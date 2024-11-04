%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by: Aaron Curtis 
% 
% This function takes the raw OSU prospectively undersampled k-space data
% and rebins it into a CINE array based on the extracted ECG signal.
%
% Inputs:     data:     Raw unsorted prospectively undersampled OSU k-space 
%                       data for a SINGLE SLICE ONLY.
%             ECGtrig:  ECG trigger for the slice of interest ONLY.
%             acqOrder: Order of acquisition of each phase encode, for the
%                       slice of interest ONLY.
%             acqTime:  Time of acquisition, same units as ECGtrig. For
%                       slice of interest only.                         
%             phases:   Number of CINE phases to reconstruction
% 
% Outputs:    A sequence of images organized into a CINE movie. This is to
%             be used for the purposes of creating a training set.
% 
% Date:         April 26, 2023
% Last Updated: April 26, 2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sortedKspace,phaseNOs] = rebinOSU_ECG (data,ECGtrig,acqOrder,acqTime,phases)

sortedKspace = zeros(size(data,1),size(data,2),1,size(data,4),phases);
phaseNOs = zeros(size(data,5),1);
count = 1;

startInd = 1;

for cycle = 1:(size(ECGtrig,2)-1)
    disp("cycle:" + cycle)
    currPhase = 1;
    [~,endInd] = min(abs(acqTime-ECGtrig(cycle+1))); 

    if (acqTime(endInd)>ECGtrig(cycle+1))
        endInd = endInd-2;
    elseif acqTime(endInd)==ECGtrig(cycle+1)
        endInd = endInd-1;
    end

    phase_time = linspace(0,phases,phases+1)*((acqTime(endInd+1)-acqTime(startInd))/phases);
    phase_time = phase_time + acqTime(startInd);

    for i = startInd:endInd

        if (acqTime(i)>phase_time(currPhase+1))
            currPhase = currPhase+1;
        end

        phaseNOs(count) = currPhase;
        count = count + 1;

        try
            if sum(sum(abs(sortedKspace(:,acqOrder(1,i),1,:,currPhase)),4),1)==0
                sortedKspace(:,acqOrder(1,i),1,:,currPhase) = data(:,acqOrder(1,i),1,:,acqOrder(3,i));
            else
                sortedKspace(:,acqOrder(1,i),1,:,currPhase) = (sortedKspace(:,acqOrder(1,i),1,:,currPhase)+data(:,acqOrder(1,i),1,:,acqOrder(3,i)))./2;
            end
        catch
            
        end

    end

    startInd = endInd + 1;

end

end
