#!/usr/bin/env python3
import os
import random
import time
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
# from pytorch_lightning.metrics.functional import r2score
import json
from collections import OrderedDict

# -------------------------------------------------------------------------------------------
# Evan Dogariu - Princeton Class of 2024, for MagNet group of Princeton Power Electronics Lab
# -------------------------------------------------------------------------------------------

MATERIALS = ['N27', 'N49', 'N87', '3C90', '3C94']  # For use transferring to other materials
WAVEFORMS = ['Sinusoidal', 'Triangle', 'SymmTrapez', 'Trapezoidal']

NN_ARCHITECTURE = [3, 15, 15, 9, 1]  # Number of neurons in each layer
INITIAL_RATE = 0.02  # Initial learning rate
DECAY_STEP = 125  # Learning rate decay stem (gamma=0.5)

CONFIG = OmegaConf.load("C:/Dropbox (Princeton)/transfer learning/fc.yaml")  # Config
# Reproducibility
random.seed(CONFIG.SEED)
np.random.seed(CONFIG.SEED)
torch.manual_seed(CONFIG.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setup CPU or GPU
if CONFIG.USE_GPU and not torch.cuda.is_available():
    raise ValueError("GPU not detected but CONFIG.USE_GPU is set to True.")
device = torch.device("cuda" if CONFIG.USE_GPU else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if CONFIG.USE_GPU else {}

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 12  # make this 20
LARGE_SIZE = 16  # make this 24
plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=LARGE_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title


# Class to represent a fully-connected NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio)
        # and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(NN_ARCHITECTURE[0], NN_ARCHITECTURE[1]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[1], NN_ARCHITECTURE[2]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[2], NN_ARCHITECTURE[3]),
            nn.ReLU(),
            nn.Linear(NN_ARCHITECTURE[3], NN_ARCHITECTURE[4])
        )

    def forward(self, x):
        return self.layers(x)

    # Returns number of trainable parameters in a network
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Returns TensorDataset to use for a NN
# Parameters: material must be something in ['N27', 'N49', 'N87', '3C90', '3C94'],
#             excitation_type must be something in ['Sinusoidal', 'Triangle', 'SymmTrapez', 'Trapezoidal', 'All']
#             if duty_ratio, frequency, or flux_density is something other than -1.0, only add points with that value
def get_dataset(material, excitation_type, duty_ratio=-1.0, frequency=-1.0, flux_density=-1.0):
    if excitation_type == 'All':
        dataset = []
        for w in WAVEFORMS:
            dataset.append(get_dataset(material, excitation_type=w, duty_ratio=duty_ratio, frequency=frequency))
        return torch.utils.data.ConcatDataset(dataset)

    # Load .json Files
    path = "C:/Dropbox (Princeton)/transfer learning/things to make n87 transfer graphs/n87 data/Data_N87_Triangle_light.json"
    with open(path, 'r') as load_f:
        data = json.load(load_f)

    freq = data['Frequency']
    flux = data['Flux_Density']
    duty = data['Duty_Ratio']
    power = data['Power_Loss']

    if duty_ratio > 0.0 or frequency > 0.0 or flux_density > 0.0:
        d_freq = []
        d_flux = []
        d_duty = []
        d_power = []
        for k in range(len(freq)):
            if (duty[k] == duty_ratio or duty_ratio < 0.0) and (freq[k] == frequency or frequency < 0.0) \
                    and (abs(flux[k] - flux_density) < 5 or flux_density < 0.0):
                d_freq.append(freq[k])
                d_flux.append(flux[k])
                d_duty.append(duty[k])
                d_power.append(power[k])
        freq = d_freq
        flux = d_flux
        duty = d_duty
        power = d_power

    # Compute labels
    # There's approximately an exponential relationship between Loss-Freq and Loss-Flux.
    # Using logarithm may help to improve the training.
    freq = np.log10(freq)
    flux = np.log10(flux)
    duty = np.array(duty)
    power = np.log10(power)

    # Reshape data
    freq = freq.reshape((-1, 1))
    flux = flux.reshape((-1, 1))
    duty = duty.reshape((-1, 1))
    temp = np.concatenate((freq, flux, duty), axis=1)

    in_tensors = torch.from_numpy(temp).view(-1, 3)
    out_tensors = torch.from_numpy(power).view(-1, 1)

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


# Trains the given neural network with the given training dataset for num_epochs epochs
# wordy=True prints training progress, show_graph graphs validation errors over training
def train(model, train_dataset, valid_dataset, num_epochs, initial_learning_rate=INITIAL_RATE,
          learning_rate_decay_step=DECAY_STEP, loss_fn='MSE',
          show_graph=False, graph_title='Validation Accuracy during Training', wordy=False):
    # Setup for training
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, **kwargs)
    if loss_fn == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss_fn == 'MAE':
        loss_fn = nn.L1Loss()
    else:
        print('No loss fn given')
        return -1
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_step, gamma=0.5)

    # Train loop
    model.train()
    valid_errors = []
    # Run valid before first epoch
    if valid_dataset is not None:
        with torch.no_grad():
            ymeas = []
            yhat = []
            for inputs, labels in valid_dataset:
                prediction = model(inputs.to(device))
                ymeas.append(10 ** labels.item())
                yhat.append(10 ** prediction.item())
            error_re = abs(np.asarray(yhat) - np.asarray(ymeas)) / abs(np.asarray(ymeas))
            error_re_avg = np.mean(error_re)
            valid_errors.append(error_re_avg)

    for epoch_i in range(1, num_epochs + 1):
        if wordy:
            print(f"Epoch {epoch_i}\n-------------------------------")
        # Training
        for batch, (inputs, labels) in enumerate(train_loader):
            # Predict
            prediction = model(inputs.to(device))
            loss = loss_fn(prediction, labels.to(device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if wordy and batch % 10 == 0:
                current = batch * len(inputs)
                print(f"Training Loss: {loss.item():>7f}  [{current:>5d}/{len(train_dataset):>5d}]")

        # Validate if given a valid_dataset
        if valid_dataset is not None:
            with torch.no_grad():
                ymeas = []
                yhat = []
                for inputs, labels in valid_dataset:
                    prediction = model(inputs.to(device))
                    ymeas.append(10 ** labels.item())
                    yhat.append(10 ** prediction.item())
                error_re = abs(np.asarray(yhat) - np.asarray(ymeas)) / abs(np.asarray(ymeas))
                error_re_avg = np.mean(error_re)
                valid_errors.append(error_re_avg)
        scheduler.step()

    # Graph
    if show_graph and len(valid_errors) > 1:
        x = np.array(range(0, num_epochs + 1))
        fig = px.line(x=x, y=valid_errors, labels={'x': 'Epoch', 'y': 'Validation Avg. Relative Error'},
                      title=graph_title)
        fig.show()

    return valid_errors


# Trains many models, combining their results into one graph. Takes lists of parameters as input.
# Usually used to train a reference and a transfer NN; you need to specify every parameter to train more models at once
def train_many(models, train_datasets, valid_datasets, num_epochs,
               initial_learning_rates=np.array([INITIAL_RATE, INITIAL_RATE]),
               learning_rate_decay_steps=np.array([DECAY_STEP, DECAY_STEP]),
               show_graph=False, graph_title='Validation Accuracies during Training',
               labels=np.array(['Reference', 'Transfer']), wordy=False):
    valid_errors = []
    x = []
    for i in range(len(models)):
        valid_errors.append(train(models[i], train_dataset=train_datasets[i], valid_dataset=valid_datasets[i],
                                  num_epochs=num_epochs[i], initial_learning_rate=initial_learning_rates[i],
                                  learning_rate_decay_step=learning_rate_decay_steps[i], show_graph=False, wordy=wordy))
        x.append(np.array(range(0, num_epochs[i] + 1)))

    if show_graph:
        fig = px.line(x=x[0], y=valid_errors[0], labels={'x': 'Epoch', 'y': 'Validation % Error'}, title=graph_title)
        for i in range(len(models)):
            fig.add_scatter(x=x[i], y=valid_errors[i], name=labels[i])
        # fig.write_html("Graphs/Training" + graph_title + ".html")  # write graph to an HTML file
        fig.show()
    return 0


# Tests a trained neural network with the given testing dataset
# Returns average and std percent error and can make Scatter or Histogram plot with graph_type='Scatter' or
# graph_type='Histogram'
def test(model, test_dataset, graph_type=None, graph_title=None, wordy=False):
    model.eval()
    ymeas = []
    yhat = []
    with torch.no_grad():
        for inputs, labels in test_dataset:
            ymeas.append(10.0 ** labels.to(device).item())
            yhat.append(10.0 ** model(inputs.to(device)).item())

    # r_squared = r2score(torch.from_numpy(np.array(ymeas)), torch.from_numpy(np.array(yhat))).item()
    abs_error = abs(np.asarray(yhat) - np.asarray(ymeas)) / abs(np.asarray(ymeas))
    avg_abs_error = np.mean(abs_error)
    std_abs_error = np.std(abs_error)
    if wordy:
        print(f"Average Absolute Relative Error: {avg_abs_error:.8f}")
        # print(f"R-Squared: {r_squared:.5f}")

    if graph_type == 'Scatter':
        # Prediction vs Target plot
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        ax.scatter(np.asarray(ymeas), np.array(yhat), label="Prediction")
        ax.plot(np.asarray(ymeas), np.array(ymeas), 'k--', label="Target")
        ax.grid(True)
        ax.legend()
        if graph_title is not None:
            ax.set_title(graph_title)
        plt.show()
        # fig.savefig('Graphs/Scatters/' + graph_title + '.png')
    elif graph_type == 'Histogram':
        num_bins = 120
        bins = 1.2 * (np.array(range(num_bins + 1)) - (num_bins / 2)) / (num_bins / 2)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        relative_error = (np.asarray(yhat) - np.asarray(ymeas)) / np.asarray(ymeas)
        ax.hist(relative_error, bins, label='Relative Error')
        ax.set_xlabel('Relative Error')
        ax.set_ylabel('Number of Points')
        ax.grid(True)
        if graph_title is not None:
            ax.set_title(graph_title)
        plt.show()
        # fig.savefig('Graphs/Histograms/' + graph_title + '.png')
    return avg_abs_error, std_abs_error


# Helper function that transforms a list of parameters to a state_dict that can be used to save/load a model's state
# This is how we do the actual 'transfer' process in transfer learning
def to_state_dict(parameters):
    # Create state dict from average parameter values
    state_dict = OrderedDict()
    n = 0
    for layer in range(len(NN_ARCHITECTURE) - 1):
        string = 'layers.' + str(layer * 2) + '.weight'
        weights = []
        for k in range(NN_ARCHITECTURE[layer + 1]):
            param = []
            for j in range(NN_ARCHITECTURE[layer]):
                param.append(parameters[n])
                n += 1
            weights.append(param)
        tensor = torch.from_numpy(np.asarray(weights))
        state_dict[string] = tensor

        string = 'layers.' + str(layer * 2) + '.bias'
        biases = []
        for k in range(NN_ARCHITECTURE[layer + 1]):
            biases.append(parameters[n])
            n += 1
        tensor = torch.from_numpy(np.asarray(biases))
        state_dict[string] = tensor
    return state_dict


# The pre-training step: gather and average model parameters on a pre-training dataset, returning the state_dict
def parameter_trials(model, train_dataset, trials, num_epochs, wordy=False):
    initial = model
    num_parameters = initial.count_parameters()
    all_parameters = []
    for k in range(num_parameters):
        all_parameters.append([])

    for trial in range(trials):
        if wordy:
            print(str(int(trial * 100 / trials)) + '%')
        count = 0
        net = initial
        train(model=net, train_dataset=train_dataset, valid_dataset=None, num_epochs=num_epochs, wordy=False,
              show_graph=False, initial_learning_rate=INITIAL_RATE, learning_rate_decay_step=10000)
        for p in net.parameters():
            if p.requires_grad:
                p = p.detach().numpy().flatten()
                for parameter in p:
                    all_parameters[count].append(parameter)
                    count += 1
    # Average each parameter over all the trials
    averages = np.mean(np.asarray(all_parameters), axis=1)
    state_dict = to_state_dict(averages)
    return state_dict


# Pretrain on all other materials, test on one material
# Parameters: test_material is the material we transfer to, pretrain_trials and pretrain_num_epochs are the number of
#             times we pretrain before averaging and how long each trial is, num_epochs and num_points are for fine
#             tuning, show_train_graph determines if we show the validation error graph during fine-tuning,
#             test_graph_type determines if we make 'Scatter' or 'Histogram' plots
# save_state_dicts works as follows:
#           if save_state_dicts='Yes', save state dicts for models at 3 points in transfer process
#                 (control after fine-tuning, transfer before and after fine-tuning)
#           if save_state_dicts='No', don't do this
#           if save_state_dicts='temp', save them with temporary names so that we can change them without overwriting
#                 (we do this in vary_data so that we can keep track of the state dict corresponding to best model)
def pretrain(test_material, pretrain_trials, pretrain_num_epochs, num_epochs, num_points, save_state_dicts,
             show_train_graph=False, test_graph_type='None', wordy=False):
    material_datasets = {'N27': None, 'N49': None, 'N87': None, '3C90': None, '3C94': None}
    for m in MATERIALS:
        # Change excitation_type based on what waveform we want to use
        material_datasets[m] = get_dataset(material=m, excitation_type='Triangle')

    # Prepare dataset for pretrain
    pretrain_datasets = []
    for m in MATERIALS:
        if m != test_material:
            pretrain_datasets.append(material_datasets[m])
    pretrain_dataset = torch.utils.data.ConcatDataset(pretrain_datasets)

    # Prepare dataset for fine-tuning and testing. We use the specified number of points for the fine-tuning dataset
    retrain, retest, _ = torch.utils.data.random_split(material_datasets[test_material],
                                                       [num_points, len(material_datasets[test_material]) -
                                                        num_points, 0])

    control = Net().double().to(device)
    transfer = Net().double().to(device)

    torch.save(control.state_dict(), 'initial.sd')  # Keep this one to vary initialization each time this is called
    control.load_state_dict(torch.load('initial.sd'))  # Keep this one to have the same initialization each pretrain
    transfer.load_state_dict(torch.load('initial.sd'))

    # Pretrain and transfer
    if wordy:
        print('Pre-training!')
    state_dict = parameter_trials(transfer, train_dataset=pretrain_dataset, num_epochs=pretrain_num_epochs,
                                  trials=pretrain_trials, wordy=False)
    transfer.load_state_dict(state_dict, strict=True)
    if save_state_dicts == 'Yes' or save_state_dicts == 'temp':
        torch.save(transfer.state_dict(), test_material + '/transferbefore.sd')

    # Train (fine-tuning)
    if wordy:
        print('Fine-tuning!')
    # ** IMPORTANT ** Here we are also using the test sets for validation ONLY BECAUSE ONLY USE THEM FOR THE GRAPH
    # IF WE WANT TO FIND HYPERPARAMETERS WE SHOULD MAKE A DIFFERENT VALIDATION SET
    train_many(models=[control, transfer], train_datasets=[retrain, retrain],
               valid_datasets=[retest, retest], num_epochs=[num_epochs, num_epochs],
               initial_learning_rates=np.array([INITIAL_RATE, INITIAL_RATE]),
               learning_rate_decay_steps=np.array([DECAY_STEP, DECAY_STEP]), show_graph=show_train_graph,
               graph_title='Validation Accuracies of Reference and Pre-trained Networks During Fine-Tuning',
               labels=np.array(['Reference', 'Transfer']), wordy=False)

    if save_state_dicts == 'Yes':
        torch.save(control.state_dict(), test_material + '/' + str(num_points) + 'control.sd')
        torch.save(transfer.state_dict(), test_material + '/' + str(num_points) + 'transferafter.sd')
    elif save_state_dicts == 'temp':
        torch.save(control.state_dict(), test_material + '/temp_' + str(num_points) + 'control.sd')
        torch.save(transfer.state_dict(), test_material + '/temp_' + str(num_points) + 'transferafter.sd')

    # Test
    if wordy:
        print('Control:\n')
    control_abs_error, control_std_error = test(control, test_dataset=retest, graph_type=test_graph_type,
                                                graph_title='Reference', wordy=wordy)
    if wordy:
        print('Transfer:\n')
    transfer_abs_error, transfer_std_error = test(transfer, test_dataset=retest, graph_type=test_graph_type,
                                                  graph_title='Transfer', wordy=wordy)

    return control_abs_error, transfer_abs_error


# Vary num_points, pretraining on all but 1 materials and testing on that 1 material and plot **TESTING ERRORS**
#           (not errors over entire dataset)
# Parameters: trials is number of trials for each number of points, nums is the list of numbers of fine-tuning points
#             to try, load_arrays determines whether to redo the trials or make the graphs with saved data,
#             std_bars determines whether to draw error bars (1.96 * SD) on the graph, and graph_title is the
#             graph title for the vary_data graph
# Every time this function is called with load_arrays=False, the state_dicts for the models of the best trial are saved
def vary_data(test_material, trials, pretrain_trials, pretrain_num_epochs, num_epochs,
              nums, graph_title='', std_bars=True, wordy=True, load_arrays=False):
    control_errors = []
    transfer_errors = []

    # Keep track of what was the best performance so far for each number of points
    min_control_errors = np.full_like(nums, float("inf"), dtype='float64')  # Initialize to maximum error
    min_transfer_errors = np.full_like(nums, float("inf"), dtype='float64')

    for t in range(1, trials + 1):
        if load_arrays:
            continue
        # Creates new initialization for each trial
        initial = Net().double().to(device)
        torch.save(initial.state_dict(), 'initial.sd')

        start = time.time()
        c_errors = []
        t_errors = []
        if wordy:
            print(test_material, 'Trial', t, 'out of ' + str(trials))
            print('0%   ------------------------------\n')

        n = 0
        while n < len(nums):
            start2 = time.time()
            errors = pretrain(test_material=test_material,
                              pretrain_trials=pretrain_trials,
                              pretrain_num_epochs=pretrain_num_epochs, num_epochs=num_epochs,
                              num_points=nums[n], save_state_dicts='temp',
                              show_train_graph=False, test_graph_type='None', wordy=False)
            c_errors.append(errors[0])
            t_errors.append(errors[1])

            # If this control model was the best out of all the trials so far, save it; otherwise, delete it
            if errors[0] < min_control_errors[n]:
                os.rename(test_material + '/temp_' + str(nums[n]) + 'control.sd',
                          test_material + '/' + str(nums[n]) + 'control.sd')
                min_control_errors[n] = errors[0]
            else:
                os.remove(test_material + '/temp_' + str(nums[n]) + 'control.sd')
            # Same thing for transfer after.
            if errors[1] < min_transfer_errors[n]:
                os.rename(test_material + '/temp_' + str(nums[n]) + 'transferafter.sd',
                          test_material + '/' + str(nums[n]) + 'transferafter.sd')
                min_transfer_errors[n] = errors[1]
            else:
                os.remove(test_material + '/temp_' + str(nums[n]) + 'transferafter.sd')

            end2 = time.time()
            if wordy:
                print('Finished ' + str(nums[n]) + 'points\n', 'Trial Time Remaining:',
                      round(float((len(nums) - n - 1) * (end2 - start2) / 60), 2), 'min\n')
            n += 1

        control_errors.append(c_errors)
        transfer_errors.append(t_errors)
        if wordy:
            end = time.time()
            tim = int(end - start)
            progress = round(t / trials, 2)
            print('Finished Trial ' + str(t) + '/' + str(trials) + ',  time=' + str(tim) + ' sec,   time_left=' +
                  str(round(float((1 - progress) * tim * trials / 60), 2)) + ' min' +
                  '---------------------------------------------------------------------------------------------\n')

    if load_arrays:
        control_avg = torch.load(test_material + '/controlavg')
        control_std = torch.load(test_material + '/controlstd')
        transfer_avg = torch.load(test_material + '/transferavg')
        transfer_std = torch.load(test_material + '/transferstd')
    else:
        # Calculate averages and standard deviations over all trials
        control_errors = np.array(control_errors)
        transfer_errors = np.array(transfer_errors)
        control_avg = np.empty((len(nums)))
        control_std = np.empty((len(nums)))
        transfer_avg = np.empty((len(nums)))
        transfer_std = np.empty((len(nums)))
        for i in range(len(nums)):
            control_avg[i] = np.mean(control_errors[:, i])
            control_std[i] = np.std(control_errors[:, i])
            transfer_avg[i] = np.mean(transfer_errors[:, i])
            transfer_std[i] = np.std(transfer_errors[:, i])
        torch.save(control_avg, test_material + '/controlavg')
        torch.save(control_std, test_material + '/controlstd')
        torch.save(transfer_avg, test_material + '/transferavg')
        torch.save(transfer_std, test_material + '/transferstd')

    plt.clf()
    plt.cla()
    plt.plot(nums, control_avg, c='g', marker='.', label='Reference')
    plt.plot(nums, transfer_avg, c='b', marker='.', label='Transfer')
    if std_bars:
        plt.errorbar(x=nums, y=control_avg, yerr=np.multiply(control_std, 1.96), fmt='none', capsize=3)
        plt.errorbar(x=nums, y=transfer_avg, yerr=np.multiply(transfer_std, 1.96), fmt='none', capsize=3)
    plt.xlabel('Amount of Training Data Used for Re-training')
    plt.legend(loc='upper right')
    plt.ylabel('Testing Avg Relative Error')
    plt.title(graph_title)
    plt.show()
    # plt.savefig(test_material + '.png')  # Can save the vary_data graph as a .pdf or .html as well


# Plots prediction vs target graph for all data points of our dataset with matching duty ratio and frequency using
# the model specified by model type and num_points, which is the number of points that the model was fine-tuned on.
# ** IMPORTANT ** The model with the right num_points must already have been trained and saved, either directly via
#                 pretrain(..., save_state_dicts=True) or indirectly via vary_data(...).
# Parameters: model_type is one of ['transferbefore', 'transferafter', 'control'], i.e. a stage in the transfer process
#             num_points must be a number of points that a model for this material was already trained and saved on
#             duty_ratios must be a list of duty ratios to make the graph with (usually [0.1, 0.2, 0.5])
#             frequency must be the frequency of the points that we want to make the prediction graph with
def plot_prediction(test_material, model_type, num_points, duty_ratios, frequency, excitation_type='Triangle'):
    model = Net().to(device).double()
    if model_type == 'transferbefore':
        model.load_state_dict(torch.load(test_material + '/' + model_type + '.sd'), strict=True)
    else:
        model.load_state_dict(torch.load(test_material + '/' + str(num_points) + model_type + '.sd'), strict=True)

    # Get datasets for graphing
    datasets = []
    for d in duty_ratios:
        datasets.append(get_dataset(test_material, excitation_type=excitation_type, duty_ratio=d, frequency=frequency))

    # Prediction vs Target plot
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'lime']
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8.5, 7.5)
    coordinates = []  # Keeps track of top right of each line. We use these coordinates to do the annotation

    model.eval()
    # Does this once for each duty ratio
    for k in range(len(datasets)):
        ymeas = []
        ypred = []
        fluxes = []
        freqs = []
        for inputs, labels in datasets[k]:
            ypred.append(10.0 ** model(inputs.to(device)).item())
            fluxes.append(10.0 ** np.array(inputs)[1])
            freqs.append(10.0 ** np.array(inputs)[0])
            ymeas.append(10.0 ** labels.to(device).item())
        if k == 0:
            ax.plot(np.asarray(fluxes), np.array(ymeas), 'k--', label="Measurement")
            ax.scatter(np.asarray(fluxes), np.array(ypred), color=colors[1], label="Prediction", marker='o', s=100)
        else:
            ax.plot(np.asarray(fluxes), np.array(ymeas), 'k--')
            ax.scatter(np.asarray(fluxes), np.array(ypred), color=colors[1], marker='o', s=100)
        coordinates.append((fluxes[-1], ymeas[-1]))

    print(np.array(ymeas).shape)
    print(np.array(ypred).shape)
    with open("C:\Dropbox (Princeton)\Training\log\\20220606/pred.csv", "a") as f:
        np.savetxt(f, np.array(ymeas)) #for single
        f.close()
    with open("C:\Dropbox (Princeton)\Training\log\\20220606/meas.csv", "a") as f:
        np.savetxt(f, np.array(ypred)) #for single
        f.close()
    with open("C:\Dropbox (Princeton)\Training\log\\20220606/flux.csv", "a") as f:
        np.savetxt(f, np.array(fluxes)) #for single
        f.close()
    with open("C:\Dropbox (Princeton)\Training\log\\20220606/freq.csv", "a") as f:
        np.savetxt(f, np.array(freqs)) #for single
        f.close()   
        
        
        
    ax.loglog()
    ax.grid(True)
    ax.legend()
    plt.legend(loc='lower right', prop={"size": 25})
    ax.text(0.24, 0.8, test_material, fontweight='bold', fontsize=48, ha='left', va='top', transform=fig.transFigure)
    ax.set_xlabel('Flux Density [T]')
    ax.set_ylabel('Core Loss [W/m\u00b3]')

    for k in range(len(duty_ratios)):
        if k == 0:
            ax.text(10.0 ** (np.log10(coordinates[k][0]) - 0.28), 10.0 ** (np.log10(coordinates[k][1]) - 0.13),
                    s='D=' + str(duty_ratios[k]), fontweight='bold', color=colors[1])
        elif k == len(duty_ratios) - 1:
            ax.text(10.0 ** (np.log10(coordinates[k][0]) - 0.10), 10.0 ** (np.log10(coordinates[k][1]) - 0.55),
                    s='D=' + str(duty_ratios[k]), fontweight='bold', color=colors[1])
        else:
            ax.text(10.0 ** (np.log10(coordinates[k][0]) - 0.21), 10.0 ** (np.log10(coordinates[k][1]) + 0.02),
                    s='D=' + str(duty_ratios[k]), fontweight='bold', color=colors[1])

    plt.show()
    # fig.savefig(test_material + '_' + str(num_points) + model_type + '.pdf')


# Helpful to visualize data of a particular flux density and duty ratio
def visualize_data(material, excitation_type='Triangle', duty_ratio=-1.0, flux_density=-1.0):
    data = get_dataset(material=material, excitation_type=excitation_type, duty_ratio=duty_ratio, frequency=-1.0,
                       flux_density=flux_density)
    freq = []
    flux = []
    duty = []
    power = []
    for inputs, labels in data:
        inp = np.array(inputs)
        freq.append(10.0 ** inp[0])
        flux.append(10.0 ** inp[1])
        duty.append(inp[2])
        power.append(10.0 ** labels.item())

    freq = np.divide(np.array(freq), 1000)
    fig = px.scatter(x=freq, y=power, color=duty,
                     data_frame={'Duty Ratio': duty}, hover_data={'Duty Ratio': True},
                     log_x=True, log_y=True,
                     color_continuous_scale=px.colors.sequential.Turbo,
                     labels={'x': 'Frequency [kHz]', 'y': 'Core Loss [kW/m\u00b3]',
                             'color': 'Duty Ratio'})
    fig.update_traces(marker=dict(size=16,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_layout(font_size=36, yaxis=dict(dtick='D1'))
    fig.update_xaxes(title_font=dict(size=48))
    fig.update_yaxes(title_font=dict(size=48))
    fig.show()


# Call main() in script part below main() (the if __name__ thing) with the arguments test_material, method, and wordy
# If using method 'Vary Data':
#           - set the nums[] to the list of numbers of points to try
#           - set trials to the number of times to do a pretrain/transfer/fine-tune cycle per element of nums[]
#           - set pretrain_trials, pretrain_num_epochs, and num_epochs to the desired values; keep in mind that
#             each run of vary_data consists of this many NN training epochs:
#                      trials * len(nums) * ((pretrain_trials * pretrain_num_epochs) + num_epochs)
#           - If you already ran vary_data with a specific list of nums[] and want to remake the graph without redoing
#             all the training, set load_arrays=True
# If using method 'Plot Prediction' to plot prediction vs target plots:
#           - Set num_points to the desired number of points
#               - ** important ** make sure that we have already run vary_data on a list of nums[] containing this
#                 num_points for this to work
#           - Set model_type to which plot you want to make:
#               - 'control' for reference model after training with num_points points of the test_material dataset
#               - 'transferbefore' for pretrained model before retraining
#               - 'transferafter' for pretrained model after retraining with num_points points
#           - Set frequency to the frequency that we want to make the pred vs target graph for; it will use all fluxes
# If using method 'Visualize Data' to plot data:
#           - set duty_ratio to the desired duty ratio (-1.0 for all), same thing with flux_density
def main(test_material, method, wordy):
    if method == 'Vary Data':
        nums = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
        vary_data(trials=4, nums=nums, test_material=test_material, pretrain_trials=8, pretrain_num_epochs=75,
                  num_epochs=1000, wordy=wordy, std_bars=True, load_arrays=False,
                  graph_title='Testing Errors over Varied Amounts of Pre-Training Data (' + test_material + ')')
    elif method == 'Plot Prediction':
        plot_prediction(test_material=test_material, model_type='control', num_points=50,
                        duty_ratios=[0.2], frequency=-1, excitation_type='Triangle')
    elif method == 'Visualize Data':
        visualize_data(material=test_material, duty_ratio=-1.0, flux_density=-1.0,
                       excitation_type='Triangle')
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(test_material='N87', method='Plot Prediction', wordy=True)