% To generate the NN training files
% The input required is the processed.mat file and the Test_Info.xlsx.
% The output is saved in the dataset folder
% Contact: Diego Serrano, ds9056@princeton.edu, Princeton University

%% Clear previous varaibles and add the paths
clear % Clear variable in the workspace
clc % Clear command window
close all % Close all open figures
addpath('Scripts') % Add the folder where the scripts are located
cd ..; % To go to the previous folder 

%%%%%%%%%%%%%%%%%%%%% PLEASE SELECT THE MATERIAL, SHAPE, DATASET TO ANALYZE
%Material = '3C92'; Shape = 'TX25-25-12'; Dataset = 1;
%Material = '3C95'; Shape = 'TX25-25-12'; Dataset = 1;
Material = 'N87'; Shape = 'R34.0X20.5X12.5'; Dataset = 5;
%Material = 'N49'; Shape = 'R16.0X9.6X6.3'; Dataset = 1;
%Material = 'N30'; Shape = '22.1X13.7X6.35'; Dataset = 1;
%Material = 'N27'; Shape = 'R20.0X10.0X7.0'; Dataset = 1;
%Material = '3F4'; Shape = 'E-32-6-20-R'; Dataset = 1;
%Material = '3C90'; Shape = 'TX-25-15-10'; Dataset = 1;
%Material = '3C94'; Shape = 'TX-20-10-7'; Dataset = 1;
%Material = '3E6'; Shape = 'TX-22-14-6.4'; Dataset = 1;
%Material = '77'; Shape = '0014'; Dataset = 1;
%Material = '78'; Shape = '0076'; Dataset = 1;

path_root = [pwd, '\', Material, '\', Shape, '\Dataset', num2str(Dataset), '\']; % Path of this file
name = [Material, ' - ', Shape, ' - Dataset ', num2str(Dataset)];
mat_name = [Material, '_', Shape, '_Data', num2str(Dataset)];

%% Read the processed .mat file
    if isfile([path_root, mat_name, '_Processed.mat'])
        load([path_root, mat_name, '_Processed.mat']); % Read the .mat file
        
        Temp = Data.Temperature; % C
        Hdc = Data.Hdc; % A/m'
        DutyP = Data.DutyP; % per unit
        DutyN = Data.DutyN; % per unit
        Freq = Data.Frequency; % Hz
        Flux = Data.Flux; % T
        Loss = Data.Volumetric_Loss; % W/m3
        B = Data.B_Field; % T
        H = Data.H_Field; % A/m
        Volt = Data.Voltage;


        Info_Setup = [Data.Date_info, ' --- ', Data.Place_info, ' --- ', Data.Trap_info, ' --- ', Data.Sine_info, ' --- ', Data.Bias_info, ' --- ', Data.Temp_info, ' --- ', Data.Meas_info, ' --- ', Data.Acquisition_info];
        Info_Core = [Data.Material, ' --- ', Data.Shape, ' --- Ae ', num2str(Data.Effective_Area), ' --- Ve ', num2str(Data.Effective_Volume), ' --- le ', num2str(Data.Effective_Length), ' --- CoreN ', num2str(Data.CoreN), ' --- N1 ', num2str(Data.Primary_Turns), ' --- N2 ', num2str(Data.Secondary_Turns), ' --- Dataset ', num2str(Data.Dataset)];
        Info_Processing = [Data.Discarding_info, ' --- ', Data.Freq_info, ' --- ', Data.Cycle_info, ' --- ', Data.Processing_info, ' --- ', Data.Date_processing];

        Ndata = length(B(:,1)); % Number of datapoints in the whole run
        Ncycle = length(B(1,:)); % Number of samples per datapoint
        disp(['Processed.mat file loaded, ', num2str(Ndata), ' datapoints with ', num2str(Ncycle), ' samples per datapoint loaded'])
    else
        disp('Cycle.mat file not been generated yet, quit execution')
        return
    end

sineNum=0;
triangular_indices = [];
trap_indices = [];
sine_indices = isnan(Data.DutyP);
indT = 1;
indTrap = 1;
for i=1:length(Temp)
    if isnan(DutyP(i))
        sineNum = sineNum+1;
    elseif abs(DutyP(i)+DutyN(i)-1) < 0.1
        triangular_indices(indT) = i;
        indT = indT+1;
    else
        trap_indices(indTrap) = i;
        indTrap = indTrap + 1;
    end
end
triangNum = numel(triangular_indices);
trapNum = numel(trap_indices);

%% augment
% Step 1: Downsample
Downsample_rate = 2^3;
B_Downsampled = B(:,1:Downsample_rate:end);
Volt_Downsampled = Volt(:,1:Downsample_rate:end);


%Step 2: Repeat Sine,Triang to balance dataset

InitialData = struct(...
    'Temperature', round(Temp),...
    'Frequency', round(Freq),...
    'Hdc', round(Hdc,1),...
    'B_Field', round(B_Downsampled,5),...
    'Voltage', round(Volt_Downsampled, 5),...
    'Volumetric_Loss', round(Loss,2));
    'Flux_Density', round(Flux,3),...
    'Duty_P', round(DutyP,1),...
    'Duty_N', round(DutyN,1),...

% for k = 1:length(InitialData.Frequency)
%     phase = round(length(InitialData.B_Field(1,:))*rand(1));
%     InitialData.B_Field(k,:) = circshift(InitialData.B_Field(k,:), phase);
% end
% 
% JSON2 = jsonencode(InitialData);
% fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_phaseOnly.json'], 'w'), JSON2); fclose('all');

%% segment sine and triang data
sineData = InitialData;
sineData.Temperature = sineData.Temperature(sine_indices);
sineData.Frequency = sineData.Frequency(sine_indices);
sineData.Hdc = sineData.Hdc(sine_indices);
sineData.Volumetric_Loss = sineData.Volumetric_Loss(sine_indices);
sineData.B_Field = sineData.B_Field(sine_indices,:);
sineData.Voltage = sineData.Voltage(sine_indicies,:);
sineData.Duty_P = sineData.Duty_P(sine_indicies,:);
sineData.Duty_N = sineData.Duty_N(sine_indicies,:);
sineData.Flux_Density = sineData.Flux_Density(sine_indicies,:);


%now triang seg
triangData = InitialData;
triangData.Temperature = triangData.Temperature(triangular_indices);
triangData.Frequency = triangData.Frequency(triangular_indices);
triangData.Hdc = triangData.Hdc(triangular_indices);
triangData.Volumetric_Loss = triangData.Volumetric_Loss(triangular_indices);
triangData.B_Field = triangData.B_Field(triangular_indices,:);
triangData.Voltage = triangData.Voltage(sine_indicies,:);
triangData.Duty_P = triangData.Duty_P(sine_indicies,:);
triangData.Duty_N = triangData.Duty_N(sine_indicies,:);
triangData.Flux_Density = triangData.Flux_Density(sine_indicies,:);

%repeat trapz just for fun!
trapzdata = InitialData;
trapzdata.Temperature = trapzdata.Temperature(trap_indices);
trapzdata.Frequency = trapzdata.Frequency(trap_indices);
trapzdata.Hdc = trapzdata.Hdc(trap_indices);
trapzdata.Volumetric_Loss = trapzdata.Volumetric_Loss(trap_indices);
trapzdata.B_Field = trapzdata.B_Field(trap_indices,:);
trapzdata.Voltage = trapzdata.Voltage(sine_indicies,:);
trapzdata.Duty_P = trapzdata.Duty_P(sine_indicies,:);
trapzdata.Duty_N = trapzdata.Duty_N(sine_indicies,:);
trapzdata.Flux_Density = trapzdata.Flux_Density(sine_indicies,:);

%% repetitions
%subtract 1 to account for retaining original dataStruct 
sineReps = round((trapNum / sineNum)); % number of times to repeat array
%sineReps = 9;

sineData.Temperature = repmat(sineData.Temperature, sineReps, 1);
sineData.Frequency = repmat(sineData.Frequency, sineReps, 1);
sineData.Hdc = repmat(sineData.Hdc, sineReps, 1);
sineData.Volumetric_Loss = repmat(sineData.Volumetric_Loss, sineReps, 1);
sineData.B_Field = repmat(sineData.B_Field, sineReps, 1);
sineData.Voltage = repmat(sineData.Voltage, sineReps, 1);
sineData.Duty_P = repmat(sineData.Duty_P, sineReps, 1);
sineData.Duty_N = repmat(sineData.Duty_N, sineReps, 1);
sineData.Flux_Density = repmat(sineData.Flux_Density, sineReps, 1);


triangReps = round((trapNum / triangNum));
%triangReps = 0;

triangData.Temperature = repmat(triangData.Temperature, triangReps, 1);
triangData.Frequency = repmat(triangData.Frequency, triangReps, 1);
triangData.Hdc = repmat(triangData.Hdc, triangReps, 1);
triangData.Volumetric_Loss = repmat(triangData.Volumetric_Loss, triangReps, 1);
triangData.B_Field = repmat(triangData.B_Field, triangReps, 1);
triangData.Voltage = repmat(triangData.Voltage, triangReps, 1);
triangData.Duty_P = repmat(triangData.Duty_P, triangReps, 1);
triangData.Duty_N = repmat(triangData.Duty_N, triangReps, 1);
triangData.Flux_Density = repmat(triangData.Flux_Density, triangReps, 1);

% Phase Each wav

for k = 1:length(sineData.Frequency)
    phase = round(length(sineData.B_Field(1,:))*rand(1));
    sineData.B_Field(k,:) = circshift(sineData.B_Field(k,:), phase);
end

for k = 1:length(triangData.Frequency)
    phase = round(length(triangData.B_Field(1,:))*rand(1));
    triangData.B_Field(k,:) = circshift(triangData.B_Field(k,:), phase);
end

for k = 1:length(trapzdata.Frequency)
    phase = round(length(trapzdata.B_Field(1,:))*rand(1));
    trapzdata.B_Field(k,:) = circshift(trapzdata.B_Field(k,:), phase);
end

% save parts
JSONS = jsonencode(sineData);
fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_sine.json'], 'w'), JSONS); fclose('all');

JSONTR = jsonencode(triangData);
fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_tria.json'], 'w'), JSONTR); fclose('all');


JSONTZ = jsonencode(trapzdata);
fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_trapz.json'], 'w'), JSONTZ); fclose('all');

%now, recombine!

DataSeq2Scalar = struct(...
    'Info_Setup', Info_Setup,...
    'Info_Core', Info_Core,...
    'Info_Processing', Info_Processing,...
    'Temperature', [sineData.Temperature; triangData.Temperature; trapzdata.Temperature],...
    'Frequency', [sineData.Frequency; triangData.Frequency; trapzdata.Frequency],...
    'Hdc', [sineData.Hdc; triangData.Hdc; trapzdata.Hdc],...
    'B_Field', [sineData.B_Field; triangData.B_Field; trapzdata.B_Field],...
    'Voltage', [sineData.Voltage; triangData.Voltage; trapzdata.Voltage],...
    'Duty_P', [sineData.Duty_P; triangData.Duty_P; trapzdata.Duty_P],...
    'Duty_N', [sineData.Duty_N; triangData.Duty_N; trapzdata.Duty_N],...
    'Flux_Density', [sineData.Flux_Density; triangData.Flux_Density; trapzdata.Flux_Density],...
    'Volumetric_Loss',[sineData.Volumetric_Loss; triangData.Volumetric_Loss; trapzdata.Volumetric_Loss]);

%% Saving the data for sequence-to-sequence NN training downsampled

finalData = length(DataSeq2Scalar.B_Field(:,1));

JSON = jsonencode(DataSeq2Scalar);
fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_Seq2Scalar_Downsampled_FullTest.json'], 'w'), JSON); fclose('all');
disp(['Seq2Seq_Downsampled.json file saved, with ', num2str(finalData), ' datapoints'])

% JSON2 = jsonencode(InitialData);
% fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_Seq2Scalar_Downsampled_noAugment.json'], 'w'), JSON2); fclose('all');


%% End   
disp(' '); disp('The script has been executed successfully'); disp(' ');

%% RePhase
for k = 1:length(DataSeq2Scalar.Frequency)
    phase = round(length(DataSeq2Scalar.B_Field(1,:))*rand(1));
    DataSeq2Scalar.B_Field(k,:) = circshift(DataSeq2Scalar.B_Field(k,:), phase);
end

JSONFix = jsonencode(DataSeq2Scalar);
fprintf(fopen([pwd, '\_Training data\_Seq2Scalar_Downsampled\', mat_name, '_Seq2Scalar_Downsampled_FullFixed.json'], 'w'), JSONFix); fclose('all');
disp(['Seq2Seq_Downsampled.json file saved, with ', num2str(finalData), ' datapoints'])

