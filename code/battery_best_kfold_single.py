# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from time import time
import json

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Layer
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.utils import custom_object_scope

# Create a traditional LSTM network
def create_LSTM(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def create_LSTM_with_attention(hidden_units, dense_units, input_shape, activation):
    x=Input(shape=input_shape)
    lstm_layer = LSTM(hidden_units, return_sequences=True, activation=activation)(x)
    attention_layer = attention()(lstm_layer)
    outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
    final_outputs=Dense(1)(outputs)
    model=Model(x,final_outputs)
    model.compile(loss='mse', optimizer='adam')    
    return model

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def plot_model_comparison(model_performance, kwd, file_name):
    """
    Create robust model comparison plots
    
    Parameters:
    -----------
    model_performance : dict
        Dictionary with metric name keys for models
    kwd : string
        Keyword for figure title
    file_name : string
        File name to save image
    """
    # Models and metrics data
    models = list(model_performance.keys())
    metrics = {metric_name: [model_performance[model][metric_name] for model in model_performance] for metric_name in list(model_performance[models[0]].keys()) if not "ainable" in metric_name}
    
    # Create figure and axes
    fig, axs = plt.subplots(1, len(metrics), figsize=(15, 6), constrained_layout=True)
    
    # Colors for bars
    # 1. Get the sequential 'Blues' colormap
    cmap = plt.get_cmap('Blues')

    # 2. Sample points between 0.2 and 1.0 
    # (Starting at 0.2 avoids pure white, keeping all colors visible)
    vals = np.linspace(0.2, 1.0, len(models))
    colors_rgba = cmap(vals)

    # 3. Convert RGBA colors to HEX format
    colors = [mcolors.to_hex(c) for c in colors_rgba]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        # Create bar plot
        axsi = axs
        if len(metrics) > 1:
            axsi = axs[i]
    
        bars = axsi.bar(models, values, color=colors)
        axsi.set_title(f'Model Comparison - {metric_name}', fontsize=14)
        axsi.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust y-axis limits to ensure labels are visible
        y_max = max(values)
        y_min = min(values)
        y_range = y_max - y_min
        new_min = y_min - 0.3 * y_range
        if y_min > 0:
            new_min = max(0, new_min)
        axsi.set_ylim(new_min, y_max + 0.3 * y_range)

        # Add value labels on top of bars
        for j, v in enumerate(values):
            v2 = max(0, v)
            if 'Parameters' != kwd and 'Size' not in metric_name:
                axsi.text(j, v2 + y_range * 0.01, f"{v2:.6f}", 
                            ha='center', va='bottom',
                            fontweight='bold', rotation=90)
            else:
                axsi.text(j, v2 + y_range * 0.01, f"{v2:.0f}", 
                            ha='center', va='bottom',
                            fontweight='bold', rotation=90)
            
        # Make plot more readable
        axsi.spines['top'].set_visible(False)
        axsi.spines['right'].set_visible(False)
        axsi.tick_params(axis='y', which='major', labelsize=12)
        axsi.tick_params(axis='x', which='major', labelsize=12, rotation=90)

    # Add overall title
    fig.suptitle('Model ' + kwd + ' Comparison', fontsize=16, fontweight='bold', y=1.05)
    
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

def create_model(input_shape, start_size):
    """Create LSTM model for sequence prediction"""
    model = Sequential([
        LSTM(start_size, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(int(start_size / 2), activation='relu'),
        Dropout(0.2),
        Dense(int(start_size / 4), activation='relu'),
        Dense(1)  # Output layer for SOH prediction
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def return_callbacks(model_file, metric):
    callbacks = [
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor=metric, mode="min"
        ),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    ]
    return callbacks

def create_seq_model(input_shape, conv1_filters = 5, conv2_filters = 5, conv_kernel_size = 6, num_cells = 64, dropout = 0.1):
    model_input = Input(shape = input_shape, name = "input_1")
    
    if conv1_filters > 0:
        final_output_layer = Conv1D(conv1_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = "conv1d_1")(model_input)
        
        if conv2_filters > 0:
            final_output_layer = Conv1D(conv2_filters, conv_kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = "conv1d_2")(final_output_layer)
        final_output_layer = Bidirectional(LSTM(num_cells, name = "bi_lstm"), name = "bidirectional_lstm")(final_output_layer)
    else:
        final_output_layer = Bidirectional(LSTM(num_cells, name = "bi_lstm"), name = "bidirectional_lstm")(model_input)

    if dropout > 0:
        final_output_layer = Dropout(dropout, name = "dropout")(final_output_layer)

    final_output_layer = Dense(1, activation = 'sigmoid', name = "output_dense")(final_output_layer)
    
    model = Model(inputs = model_input, outputs = final_output_layer)

    return model

# XGBoost
import xgboost as xgb

# Set random seeds for reproducibility
n_splits = 5
random_state = 42
start_sizes = [32, 64, 128, 256, 512]
kernel_sizes = [4, 6, 8, 10]
test_bi = False
test_conv = False
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Load data for models
with open('../models/data.pkl', 'rb') as f:
    data = pickle.load(f)
    
X = data['X']
y_soh = data['y_soh']
features = data['features']
metadata = data['metadata']
y_battery = data['metadata']['battery_id']

print(f"Data shape: {X.shape}")

plot_history = False
model_is_min = dict()
model_is_max = dict()
performance_dict = dict()
size_time_dict = dict()
parameters_dict = dict()
for ii in range(n_splits):
    for jj in range(n_splits):
        # Load models
        models_dir = '../models/nested_n_splits_' + str(n_splits) + '/seed_' + str(random_state) + '/test_fold_' + str(ii + 1) + '/validation_fold_' + str(jj + 1)

        if plot_history:
            for start_size in start_sizes:
                # Load models in subdirectories
                cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
                # Load history
                with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    history = pickle.load(f)
            
                # Plot training history
                plt.figure(figsize=(5, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('LSTM Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + '\nTest fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            for start_size in start_sizes:
                # Load models in subdirectories
                cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
                custom_objects = {'attention': attention}
                # Load history
                with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    with custom_object_scope(custom_objects):
                        history = pickle.load(f)
            
                # Plot training history
                plt.figure(figsize=(5, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('LSTM Att Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + '\nTest fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
                plt.close()

            for start_size in start_sizes:
                # Load models in subdirectories
                cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
                # Load history
                with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    history = pickle.load(f)
            
                # Plot training history
                plt.figure(figsize=(5, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('LSTM SELU Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + '\nTest fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MSE)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
                plt.close()

        model_performance = dict()
        model_size_time = dict()
        model_parameters = dict()
        y_pred = dict()
        y_true = dict()

        # Load XGBoost model performance
        with open(os.path.join(models_dir, 'xgb_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
            model_performance['XGBoost'] = pickle.load(f)
        # Load XGBoost model size and time
        with open(os.path.join(models_dir, 'xgb_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
            model_size_time['XGBoost'] = pickle.load(f)
        # Load XGBoost model parameters
        with open(os.path.join(models_dir, 'xgb_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
            model_parameters['XGBoost'] = pickle.load(f)
        # Load y_test_xgb_np
        with open(os.path.join(models_dir, 'y_test_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
            y_true['XGBoost'] = pickle.load(f)
        # Load y_pred_xgb_np
        with open(os.path.join(models_dir, 'y_pred_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
            y_pred['XGBoost'] = pickle.load(f)

        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load LSTM model performance
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_performance['LSTM Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM model size and time
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_size_time['LSTM Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM model parameters
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_parameters['LSTM Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_test_lstm_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_true['LSTM Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_pred_lstm_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_pred['LSTM Cells: ' + str(start_size)] = pickle.load(f)
                
        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load LSTM SELU model performance
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_performance['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM SELU model size and time
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_size_time['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM SELU model parameters
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_parameters['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_test_lstm_selu_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_true['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_pred_lstm_selu_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_pred['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)

        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load LSTM Att model performance
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_performance['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM Att model size and time
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_size_time['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
            # Load LSTM Att model parameters
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                model_parameters['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_test_lstm_att_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_true['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_pred_lstm_att_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                y_pred['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)

        if test_bi:
            for start_size in start_sizes:
                # Load models in subdirectories
                cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
                # Load Bi model performance
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    model_performance['Bi Cells: ' + str(start_size)] = pickle.load(f)
                # Load Bi model size and time
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    model_size_time['Bi Cells: ' + str(start_size)] = pickle.load(f)
                # Load Bi model parameters
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    model_parameters['Bi Cells: ' + str(start_size)] = pickle.load(f)
                # Load y_test_bi_np
                with open(os.path.join(cells_subdir, 'y_test_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    y_true['Bi Cells: ' + str(start_size)] = pickle.load(f)
                # Load y_pred_bi_np
                with open(os.path.join(cells_subdir, 'y_pred_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                    y_pred['Bi Cells: ' + str(start_size)] = pickle.load(f)

        if test_conv:
            for start_size in start_sizes:
                # Load models in subdirectories
                cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
                for kernel_size in kernel_sizes:
                    # Load the Conv model in subdirectories
                    kernel_subdir = os.path.join(cells_subdir, 'kernel_' + str(kernel_size))
                    # Load Conv model performance
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                        model_performance['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                    # Load Conv model size and time
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                        model_size_time['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                    # Load Conv model parameters
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                        model_parameters['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                    # Load y_test_conv_np
                    with open(os.path.join(kernel_subdir, 'y_test_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                        y_true['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                    # Load y_pred_conv_np
                    with open(os.path.join(kernel_subdir, 'y_pred_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'rb') as f:
                        y_pred['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)

        file_name_performance = os.path.join(models_dir, 'model_performance_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png')
        file_name_size_time = os.path.join(models_dir, 'model_size_time_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png')
        file_name_parameters = os.path.join(models_dir, 'model_parameters_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png')
        
        model_types = set(sorted([model.split(" Cells")[0] for model in model_performance]))
        max_for_metric = dict()
        min_for_metric = dict()
        
        max_for_metric["All"] = dict()
        min_for_metric["All"] = dict()
        for model in model_performance:
            for metric_name in model_performance[model]:
                if metric_name not in max_for_metric["All"] or model_performance[model][metric_name] > max_for_metric["All"][metric_name]:
                    max_for_metric["All"][metric_name] = model_performance[model][metric_name]
                if metric_name not in min_for_metric["All"] or model_performance[model][metric_name] < min_for_metric["All"][metric_name]:
                    min_for_metric["All"][metric_name] = model_performance[model][metric_name]
        for model_type in model_types:
            max_for_metric[model_type] = dict()
            min_for_metric[model_type] = dict()
            for model in model_performance:
                if model_type == model.split(" Cells")[0]:
                    for metric_name in model_performance[model]:
                        if metric_name not in max_for_metric[model_type] or model_performance[model][metric_name] > max_for_metric[model_type][metric_name]:
                            max_for_metric[model_type][metric_name] = model_performance[model][metric_name]
                        if metric_name not in min_for_metric[model_type] or model_performance[model][metric_name] < min_for_metric[model_type][metric_name]:
                            min_for_metric[model_type][metric_name] = model_performance[model][metric_name]
        for model_type in max_for_metric:
            for metric_name in max_for_metric[model_type]:
                for model in model_performance:
                    if model_type not in model_is_max:
                        model_is_max[model_type] = dict()
                    if metric_name not in model_is_max[model_type]:
                        model_is_max[model_type][metric_name] = dict()
                    if model not in model_is_max[model_type][metric_name]:
                        model_is_max[model_type][metric_name][model] = 0
                    model_is_max[model_type][metric_name][model] += 1 * model_performance[model][metric_name] == max_for_metric[model_type][metric_name]
                    if model_type not in model_is_min:
                        model_is_min[model_type] = dict()
                    if metric_name not in model_is_min[model_type]:
                        model_is_min[model_type][metric_name] = dict()
                    if model not in model_is_min[model_type][metric_name]:
                        model_is_min[model_type][metric_name][model] = 0
                    model_is_min[model_type][metric_name][model] += 1 * model_performance[model][metric_name] == min_for_metric[model_type][metric_name]
        for model in model_performance:
            for metric_name in model_performance[model]:
                if metric_name not in performance_dict:
                    performance_dict[metric_name] = dict()
                if ii not in performance_dict[metric_name]:
                    performance_dict[metric_name][ii] = dict()
                if jj not in performance_dict[metric_name][ii]:
                    performance_dict[metric_name][ii][jj] = dict()
                if model not in performance_dict[metric_name][ii][jj]:
                    performance_dict[metric_name][ii][jj][model] = dict()
                performance_dict[metric_name][ii][jj][model] = model_performance[model][metric_name]
        for model in model_size_time:
            for metric_name in model_size_time[model]:
                if metric_name not in size_time_dict:
                    size_time_dict[metric_name] = dict()
                if ii not in size_time_dict[metric_name]:
                    size_time_dict[metric_name][ii] = dict()
                if jj not in size_time_dict[metric_name][ii]:
                    size_time_dict[metric_name][ii][jj] = dict()
                if model not in size_time_dict[metric_name][ii][jj]:
                    size_time_dict[metric_name][ii][jj][model] = dict()
                size_time_dict[metric_name][ii][jj][model] = model_size_time[model][metric_name]
        for model in model_parameters:
            for metric_name in model_parameters[model]:
                if metric_name not in parameters_dict:
                    parameters_dict[metric_name] = dict()
                if ii not in parameters_dict[metric_name]:
                    parameters_dict[metric_name][ii] = dict()
                if jj not in parameters_dict[metric_name][ii]:
                    parameters_dict[metric_name][ii][jj] = dict()
                if model not in parameters_dict[metric_name][ii][jj]:
                    parameters_dict[metric_name][ii][jj][model] = dict()
                parameters_dict[metric_name][ii][jj][model] = model_parameters[model][metric_name]
        sorted_models = sorted(list(model_performance.keys()))
mul_factor = {
    "RMSE": 1,
    "MAE": 1,
    "MSE": 1,
    "R²": 1,
    "Training Time (s)": 1,
    "Testing Time (s)": 1,
    "Size (B)": 1,
    "Trainable Parameters": 1,
    "Non-trainable Parameters": 1,
    "Total Parameters": 1
}
unit = {
    "RMSE": "",
    "MAE": "",
    "MSE": "",
    "R²": "",
    "Training Time (s)": "",
    "Testing Time (s)": "",
    "Size (B)": "",
    "Trainable Parameters": "",
    "Non-trainable Parameters": "",
    "Total Parameters": ""
}
for metric_name in performance_dict:
    print(metric_name.split(" (")[0])
    print("Test & Val & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in performance_dict[metric_name]:
        for jj in performance_dict[metric_name][ii]:
            model_list = [(performance_dict[metric_name][ii][jj][model] * mul_factor[metric_name]) for model in sorted_models]
            new_model_list = []
            for s in model_list:
                if (s == min(model_list) and metric_name[0] != "R") or (s == max(model_list) and metric_name[0] == "R"):
                    new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                    continue
                if (s == max(model_list) and metric_name[0] != "R") or (s == min(model_list) and metric_name[0] == "R"):
                    new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                    continue
                new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
            print("$" + str(ii + 1) + "$ & $" + str(jj + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")
for metric_name in size_time_dict:
    print(metric_name.split(" (")[0])
    print("Test & Val & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in size_time_dict[metric_name]:
        for jj in size_time_dict[metric_name][ii]:
            model_list = [(size_time_dict[metric_name][ii][jj][model] * mul_factor[metric_name]) for model in sorted_models]
            new_model_list = []
            for s in model_list:
                if s == min(model_list):
                    new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                    continue
                if s == max(model_list):
                    new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                    continue
                new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
            print("$" + str(ii + 1) + "$ & $" + str(jj + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")
for metric_name in parameters_dict:
    print(metric_name.split(" (")[0])
    print("Test & Val & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in parameters_dict[metric_name]:
        for jj in parameters_dict[metric_name][ii]:
            model_list = [(parameters_dict[metric_name][ii][jj][model] * mul_factor[metric_name]) for model in sorted_models]
            new_model_list = []
            for s in model_list:
                if s == min(model_list):
                    new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                    continue
                if s == max(model_list):
                    new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                    continue
                new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
            print("$" + str(ii + 1) + "$ & $" + str(jj + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")
lstm_cells = 0
bi_cells = 0
conv_cells = 0
conv_kernel = 0
lstm_selu_cells = 0
lstm_att_cells = 0
for metric_name in model_is_max["XGBoost"]:
    print(metric_name)
    for model_type in model_is_max:
        list_for_metric_max = []
        list_for_metric_min = []
        for model in model_is_max[model_type][metric_name]:
            list_for_metric_max.append((model_is_max[model_type][metric_name][model], model))
        for model in model_is_min[model_type][metric_name]:
            list_for_metric_min.append((model_is_min[model_type][metric_name][model], model))
        if "LSTM" in model_type:
            if metric_name[0] != "R":
                print(model_type + " & " + (" & ").join([["$" + str(lmm[0]) + "$" for lmm in list_for_metric_min if "Cells: " + str(start_size) in lmm[1] and model_type + " Cells: " in lmm[1]][0] for start_size in start_sizes]) + " \\\\ \\hline")
            else:
                print(model_type + " & " + (" & ").join([["$" + str(lmm[0]) + "$" for lmm in list_for_metric_max if "Cells: " + str(start_size) in lmm[1] and model_type + " Cells: " in lmm[1]][0] for start_size in start_sizes]) + " \\\\ \\hline")
for model_type in model_is_max:
    for metric_name in model_is_max[model_type]:
        list_for_metric_max = []
        list_for_metric_min = []
        for model in model_is_max[model_type][metric_name]:
            list_for_metric_max.append((model_is_max[model_type][metric_name][model], model))
        for model in model_is_min[model_type][metric_name]:
            list_for_metric_min.append((model_is_min[model_type][metric_name][model], model))
        if "All" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
        if "LSTM" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
            if metric_name == "MAE":
                lstm_cells = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-1])
        if "LSTM SELU" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
            if metric_name == "MAE":
                lstm_selu_cells = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-1])
        if "LSTM Att" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
            if metric_name == "MAE":
                lstm_att_cells = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-1])
        if "Bi" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
            if metric_name == "MAE":
                bi_cells = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-1])
        if "Conv" == model_type:
            if metric_name[0] != "R":
                print(model_type, metric_name, "min", sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True))
            else:
                print(model_type, metric_name, "max", sorted([lmm for lmm in list_for_metric_max if lmm[0]], reverse = True))
            if metric_name == "MAE":
                conv_cells = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-3])
                conv_kernel = int(sorted([lmm for lmm in list_for_metric_min if lmm[0]], reverse = True)[0][1].split(" ")[-1])

performance_dict_short = dict()
size_time_dict_short = dict()
parameters_dict_short = dict()
for ii in range(n_splits):
    # Load models
    models_dir = '../models/single_n_splits_' + str(n_splits) + '/seed_' + str(random_state) + '/test_fold_' + str(ii + 1)
    
    if plot_history:
        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load history
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                history = pickle.load(f)
        
            # Plot training history
            plt.figure(figsize=(5, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.title('LSTM Model Training Cells: ' + str(start_size) + '\nNumber of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()

        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            custom_objects = {'attention': attention}
            # Load history
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                with custom_object_scope(custom_objects):
                    history = pickle.load(f)
        
            # Plot training history
            plt.figure(figsize=(5, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.title('LSTM Att Model Training Cells: ' + str(start_size) + '\nNumber of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()

        for start_size in start_sizes:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load history
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                history = pickle.load(f)
        
            # Plot training history
            plt.figure(figsize=(5, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.title('LSTM SELU Model Training Cells: ' + str(start_size) + '\nNumber of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_training_narrow_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()

    model_performance = dict()
    model_size_time = dict()
    model_parameters = dict()
    y_pred = dict()
    y_true = dict()

    # Load XGBoost model performance
    with open(os.path.join(models_dir, 'xgb_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
        model_performance['XGBoost'] = pickle.load(f)
    # Load XGBoost model size and time
    with open(os.path.join(models_dir, 'xgb_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
        model_size_time['XGBoost'] = pickle.load(f)
    # Load XGBoost model parameters
    with open(os.path.join(models_dir, 'xgb_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
        model_parameters['XGBoost'] = pickle.load(f)
    # Load y_test_xgb_np
    with open(os.path.join(models_dir, 'y_test_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
        y_true['XGBoost'] = pickle.load(f)
    # Load y_pred_xgb_np
    with open(os.path.join(models_dir, 'y_pred_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
        y_pred['XGBoost'] = pickle.load(f)

    for start_size in [lstm_cells]:
        # Load models in subdirectories
        cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
        # Load LSTM model performance
        with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_performance['LSTM Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM model size and time
        with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_size_time['LSTM Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM model parameters
        with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_parameters['LSTM Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_test_lstm_np
        with open(os.path.join(cells_subdir, 'y_test_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_true['LSTM Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_pred_lstm_np
        with open(os.path.join(cells_subdir, 'y_pred_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_pred['LSTM Cells: ' + str(start_size)] = pickle.load(f)

    for start_size in [lstm_selu_cells]:
        # Load models in subdirectories
        cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
        # Load LSTM SELU model performance
        with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_performance['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM SELU model size and time
        with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_size_time['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM SELU model parameters
        with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_parameters['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_test_lstm_selu_np
        with open(os.path.join(cells_subdir, 'y_test_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_true['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_pred_lstm_selu_np
        with open(os.path.join(cells_subdir, 'y_pred_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_pred['LSTM SELU Cells: ' + str(start_size)] = pickle.load(f)

    for start_size in [lstm_att_cells]:
        # Load models in subdirectories
        cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
        # Load LSTM Att model performance
        with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_performance['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM Att model size and time
        with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_size_time['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
        # Load LSTM Att model parameters
        with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            model_parameters['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_test_lstm_att_np
        with open(os.path.join(cells_subdir, 'y_test_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_true['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)
        # Load y_pred_lstm_att_np
        with open(os.path.join(cells_subdir, 'y_pred_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
            y_pred['LSTM Att Cells: ' + str(start_size)] = pickle.load(f)

    if test_bi:
        for start_size in [bi_cells]:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            # Load Bi model performance
            with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                model_performance['Bi Cells: ' + str(start_size)] = pickle.load(f)
            # Load Bi model size and time
            with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                model_size_time['Bi Cells: ' + str(start_size)] = pickle.load(f)
            # Load Bi model parameters
            with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                model_parameters['Bi Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_test_bi_np
            with open(os.path.join(cells_subdir, 'y_test_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                y_true['Bi Cells: ' + str(start_size)] = pickle.load(f)
            # Load y_pred_bi_np
            with open(os.path.join(cells_subdir, 'y_pred_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                y_pred['Bi Cells: ' + str(start_size)] = pickle.load(f)

    if test_conv:
        for start_size in [conv_cells]:
            # Load models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            for kernel_size in kernel_sizes:
                # Load the Conv model in subdirectories
                kernel_subdir = os.path.join(cells_subdir, 'kernel_' + str(kernel_size))
                # Load Conv model performance
                with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                    model_performance['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                # Load Conv model size and time
                with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                    model_size_time['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                # Load Conv model parameters
                with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                    model_parameters['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                # Load y_test_conv_np
                with open(os.path.join(kernel_subdir, 'y_test_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                    y_true['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)
                # Load y_pred_conv_np
                with open(os.path.join(kernel_subdir, 'y_pred_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.pkl'), 'rb') as f:
                    y_pred['Conv Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size)] = pickle.load(f)

    replot_two = True
    if replot_two:
        file_name_performance = os.path.join(models_dir, 'model_performance_short_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png')
        file_name_size_time = os.path.join(models_dir, 'model_size_time_short_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png')
        file_name_parameters = os.path.join(models_dir, 'model_parameters_short_comparison_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png')

        # Generate the comparison plots
        plot_model_comparison(model_performance, 'Performance', file_name_performance)
        plot_model_comparison(model_size_time, 'Size and Time', file_name_size_time)
        plot_model_comparison(model_parameters, 'Parameters', file_name_parameters)

        # Visualize predictions for all models
        plt.figure(figsize=(6, 12))
        nrows = int(np.sqrt(len(y_pred)))
        ncols = int(np.ceil(len(y_pred) / int(np.sqrt(len(y_pred)))))
        nrows = len(y_pred)
        ncols = 1
        mindex = 1
        for model in list(y_pred.keys()):
            # Plot model predictions
            plt.subplot(nrows, ncols, mindex)
            plt.scatter(y_true[model], y_pred[model], alpha=0.5)
            vv = [min(y_true[model]), max(y_true[model])]
            plt.plot(vv, vv, 'r--')
            plt.xlabel('Actual SOH')
            plt.ylabel('Predicted SOH')
            plt.title(model)
            plt.grid(True)
            mindex += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'model_short_predictions_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '.png'), dpi=300, bbox_inches='tight')
        plt.close()
    for model in model_performance:
        for metric_name in model_performance[model]:
            if metric_name not in performance_dict_short:
                performance_dict_short[metric_name] = dict()
            if ii not in performance_dict_short[metric_name]:
                performance_dict_short[metric_name][ii] = dict()
            performance_dict_short[metric_name][ii][model] = model_performance[model][metric_name]
    for model in model_size_time:
        for metric_name in model_size_time[model]:
            if metric_name not in size_time_dict_short:
                size_time_dict_short[metric_name] = dict()
            if ii not in size_time_dict_short[metric_name]:
                size_time_dict_short[metric_name][ii] = dict()
            size_time_dict_short[metric_name][ii][model] = model_size_time[model][metric_name]
    for model in model_parameters:
        for metric_name in model_parameters[model]:
            if metric_name not in parameters_dict_short:
                parameters_dict_short[metric_name] = dict()
            if ii not in parameters_dict_short[metric_name]:
                parameters_dict_short[metric_name][ii] = dict()
            parameters_dict_short[metric_name][ii][model] = model_parameters[model][metric_name]
    sorted_models = sorted(list(model_performance.keys()))
for metric_name in performance_dict_short:
    print(metric_name.split(" (")[0])
    print("Test & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in performance_dict_short[metric_name]:
        model_list = [(performance_dict_short[metric_name][ii][model] * mul_factor[metric_name]) for model in sorted_models]
        new_model_list = []
        for s in model_list:
            if (s == min(model_list) and metric_name[0] != "R") or (s == max(model_list) and metric_name[0] == "R"):
                new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                continue
            if (s == max(model_list) and metric_name[0] != "R") or (s == min(model_list) and metric_name[0] == "R"):
                new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                continue
            new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
        print("$" + str(ii + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")
for metric_name in size_time_dict_short:
    print(metric_name.split(" (")[0])
    print("Test & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in size_time_dict_short[metric_name]:
        model_list = [(size_time_dict_short[metric_name][ii][model] * mul_factor[metric_name]) for model in sorted_models]
        new_model_list = []
        for s in model_list:
            if s == min(model_list):
                new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                continue
            if s == max(model_list):
                new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                continue
            new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
        print("$" + str(ii + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")
for metric_name in parameters_dict_short:
    print(metric_name.split(" (")[0])
    print("Test & " + (" & ").join(sorted_models) + " \\\\ \\hline")
    for ii in parameters_dict_short[metric_name]:
        model_list = [(parameters_dict_short[metric_name][ii][model] * mul_factor[metric_name]) for model in sorted_models]
        new_model_list = []
        for s in model_list:
            if s == min(model_list):
                new_model_list.append("$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}")
                continue
            if s == max(model_list):
                new_model_list.append("\\underline{$\\mathbf{" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "}$ \\textbf{" + unit[metric_name] + "}}")
                continue
            new_model_list.append("$" + ("{valu:.6f}".format(valu = s)).replace(".000000", "") + "$ " + unit[metric_name])
        print("$" + str(ii + 1) + "$ & " + (" & ").join(new_model_list).replace(" \\textbf{}", "").replace("  ", " ") + " \\\\ \\hline")