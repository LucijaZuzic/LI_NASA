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

skf1 = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
# Split into train/test
for ii, (train_index, test_index) in enumerate(skf1.split(y_soh, y_battery)):
    print(f"Test fold {ii + 1}:")
    # Split into train/val
    X_new = (X.loc[train_index]).reset_index(drop = True)
    y_soh_new = (y_soh.loc[train_index]).reset_index(drop = True)
    y_battery_new = (y_battery.loc[train_index]).reset_index(drop = True)
    skf2 = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for jj, (train_no_val_index, val_index) in enumerate(skf2.split(y_soh_new, y_battery_new)):
        # Save models
        models_dir = '../models/nested_n_splits_' + str(n_splits) + '/seed_' + str(random_state) + '/test_fold_' + str(ii + 1) + '/validation_fold_' + str(jj + 1)
        os.makedirs(models_dir, exist_ok=True)

        print(f"Validation fold {jj + 1}:")

        scaler = MinMaxScaler()
        X_train_no_val = scaler.fit_transform(X_new.loc[train_no_val_index])
        X_train = scaler.transform(X_new)
        X_test = scaler.transform(X.loc[test_index])
        X_val = scaler.transform(X_new.loc[val_index])

        # Reshape to 3D
        X_train_no_val_lstm = X_train_no_val.reshape((X_train_no_val.shape[0], 1, X_train_no_val.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        y_train_no_val = y_soh_new.loc[train_no_val_index]
        y_train = y_soh_new
        y_test = y_soh.loc[test_index]
        y_val = y_soh_new.loc[val_index]

        print(f"Training data shape: {X_train_no_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Validation data shape: {X_val.shape}")

        # Train XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state
        )

        # Create evaluation sets (similar to original)
        eval_set = [(X_train, y_train), (X_test, y_test)]

        # Basic fit with evaluation
        xgb_time_train_start = time()
        xgb_model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        xgb_time_train_end = time()
        xgb_time_train = xgb_time_train_end - xgb_time_train_start

        # Save XGBoost model
        xgb_model_file_name = os.path.join(models_dir, 'xgboost_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.json')
        xgb_model.save_model(xgb_model_file_name)

        pickle.dump(xgb_model, open(xgb_model_file_name.replace('.json', '.pkl'), 'wb'))
        clf2 = pickle.load(open(xgb_model_file_name.replace('.json', '.pkl'), 'rb'))

        # If you want to implement early stopping manually, you can do:
        best_iteration = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else None
        if best_iteration:
            print(f"Best iteration: {best_iteration}")

        # Evaluate XGBoost model
        xgb_time_test_start = time()
        y_pred_xgb = xgb_model.predict(X_test)
        xgb_time_test_end = time()
        xgb_time_test = xgb_time_test_end - xgb_time_test_start

        xgb_mse = mean_squared_error(y_test, y_pred_xgb)
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        xgb_model_file_size = os.path.getsize(xgb_model_file_name)

        xgb_trainable_count = len(xgb_model.get_booster().trees_to_dataframe())
        xgb_non_trainable_count = 0
        xgb_total_count = xgb_trainable_count + xgb_non_trainable_count

        print(f"XGBoost Model Performance:")
        print(f"MSE: {xgb_mse:.10f}")
        print(f"MAE: {xgb_mae:.10f}")
        print(f"R²: {xgb_r2:.10f}")
        print(f"XGBoost Model Size and Time:")
        print(f"Size (B): {xgb_model_file_size:.10f}")
        print(f"Training Time (s): {xgb_time_train:.10f}")
        print(f"Testing Time (s): {xgb_time_test:.10f}")
        print(f"XGBoost Model Parameters:")
        print(f"Trainable Parameters: {xgb_trainable_count:.10f}")
        print(f"Non-trainable Parameters: {xgb_non_trainable_count:.10f}")
        print(f"Total Parameters: {xgb_total_count:.10f}")

        # Save XGBoost model performance
        with open(os.path.join(models_dir, 'xgb_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump({"MSE": xgb_mse, "MAE": xgb_mae, "R²": xgb_r2}, f)

        # Save XGBoost model size and time
        with open(os.path.join(models_dir, 'xgb_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump({"Size (B)": xgb_model_file_size, "Training Time (s)": xgb_time_train, "Testing Time (s)": xgb_time_test}, f)

        # Save XGBoost model parameters
        with open(os.path.join(models_dir, 'xgb_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump({"Trainable Parameters": xgb_trainable_count, "Non-trainable Parameters": xgb_non_trainable_count, "Total Parameters": xgb_total_count}, f)

        # Plot feature importance (as in original)
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(xgb_model, max_num_features=10)
        plt.title('XGBoost Feature Importance Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'feature_importance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # For XGBoost: Convert pandas Series to numpy array if needed
        if hasattr(y_test, 'values'):
            y_test_xgb_np = y_test.values
            y_pred_xgb_np = y_pred_xgb
        else:
            y_test_xgb_np = y_test
            y_pred_xgb_np = y_pred_xgb

        # Save y_test_xgb_np
        with open(os.path.join(models_dir, 'y_test_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump(y_test_xgb_np, f)
        # Save y_pred_xgb_np
        with open(os.path.join(models_dir, 'y_pred_xgb_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump(y_pred_xgb_np, f)

        # Save scaler
        with open(os.path.join(models_dir, 'scaler_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        for start_size in start_sizes:
            # Save models in subdirectories
            cells_subdir = os.path.join(models_dir, 'num_cells_' + str(start_size))
            os.makedirs(cells_subdir, exist_ok=True)

            # Create and train LSTM model
            input_shape = (X_train_no_val_lstm.shape[1], X_train_no_val_lstm.shape[2])
            lstm_model = create_model(input_shape, start_size)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # Print model summary
            print(lstm_model.summary())
            lstm_time_train_start = time()
            history = lstm_model.fit(
                X_train_no_val_lstm, y_train_no_val,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_lstm, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            lstm_time_train_end = time()
            lstm_time_train = lstm_time_train_end - lstm_time_train_start

            # Save LSTM model
            lstm_model_file_name = os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.h5')
            lstm_model.save(lstm_model_file_name)

            # Evaluate LSTM model
            lstm_time_test_start = time()
            y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
            lstm_time_test_end = time()
            lstm_time_test = lstm_time_test_end - lstm_time_test_start

            lstm_mse = mean_squared_error(y_test, y_pred_lstm)
            lstm_mae = mean_absolute_error(y_test, y_pred_lstm)
            lstm_r2 = r2_score(y_test, y_pred_lstm)
            lstm_model_file_size = os.path.getsize(lstm_model_file_name)
            lstm_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_model.trainable_weights]))
            lstm_non_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_model.non_trainable_weights]))
            lstm_total_count = lstm_trainable_count + lstm_non_trainable_count

            print(f"LSTM Model Performance:")
            print(f"MSE: {lstm_mse:.10f}")
            print(f"MAE: {lstm_mae:.10f}")
            print(f"R²: {lstm_r2:.10f}")
            print(f"LSTM Model Size and Time:")
            print(f"Size (B): {lstm_model_file_size:.10f}")
            print(f"Training Time (s): {lstm_time_train:.10f}")
            print(f"Testing Time (s): {lstm_time_test:.10f}")
            print(f"LSTM Model Parameters:")
            print(f"Trainable Parameters: {lstm_trainable_count:.10f}")
            print(f"Non-trainable Parameters: {lstm_non_trainable_count:.10f}")
            print(f"Total Parameters: {lstm_total_count:.10f}")

            # Save LSTM model performance
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"MSE": lstm_mse, "MAE": lstm_mae, "R²": lstm_r2}, f)

            # Save LSTM model size and time
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Size (B)": lstm_model_file_size, "Training Time (s)": lstm_time_train, "Testing Time (s)": lstm_time_test}, f)

            # Save LSTM model parameters
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Trainable Parameters": lstm_trainable_count, "Non-trainable Parameters": lstm_non_trainable_count, "Total Parameters": lstm_total_count}, f)

            # Plot training history
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_training_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save history
            with open(os.path.join(cells_subdir, 'lstm_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(history, f)

            # For LSTM: Convert pandas Series to numpy array if needed
            if hasattr(y_test, 'values'):
                y_test_lstm_np = y_test.values
                y_pred_lstm_np = y_pred_lstm
            else:
                y_test_lstm_np = y_test
                y_pred_lstm_np = y_pred_lstm

            # Save y_test_lstm_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_test_lstm_np, f)
            # Save y_pred_lstm_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_pred_lstm_np, f)

            # Create and train LSTM SELU model
            input_shape = (X_train_no_val_lstm.shape[1], X_train_no_val_lstm.shape[2])
            lstm_selu_model = create_LSTM(start_size, start_size / 2, input_shape, ['selu', 'selu'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # Print model summary
            print(lstm_selu_model.summary())
            lstm_selu_time_train_start = time()
            history = lstm_selu_model.fit(
                X_train_no_val_lstm, y_train_no_val,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_lstm, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            lstm_selu_time_train_end = time()
            lstm_selu_time_train = lstm_selu_time_train_end - lstm_selu_time_train_start

            # Save LSTM SELU model
            lstm_selu_model_file_name = os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.h5')
            lstm_selu_model.save(lstm_selu_model_file_name)

            # Evaluate LSTM SELU model
            lstm_selu_time_test_start = time()
            y_pred_lstm_selu = lstm_selu_model.predict(X_test_lstm).flatten()
            lstm_selu_time_test_end = time()
            lstm_selu_time_test = lstm_selu_time_test_end - lstm_selu_time_test_start

            lstm_selu_mse = mean_squared_error(y_test, y_pred_lstm_selu)
            lstm_selu_mae = mean_absolute_error(y_test, y_pred_lstm_selu)
            lstm_selu_r2 = r2_score(y_test, y_pred_lstm_selu)
            lstm_selu_model_file_size = os.path.getsize(lstm_selu_model_file_name)
            lstm_selu_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_selu_model.trainable_weights]))
            lstm_selu_non_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_selu_model.non_trainable_weights]))
            lstm_selu_total_count = lstm_selu_trainable_count + lstm_selu_non_trainable_count

            print(f"LSTM SELU Model Performance:")
            print(f"MSE: {lstm_selu_mse:.10f}")
            print(f"MAE: {lstm_selu_mae:.10f}")
            print(f"R²: {lstm_selu_r2:.10f}")
            print(f"LSTM SELU Model Size and Time:")
            print(f"Size (B): {lstm_selu_model_file_size:.10f}")
            print(f"Training Time (s): {lstm_selu_time_train:.10f}")
            print(f"Testing Time (s): {lstm_selu_time_test:.10f}")
            print(f"LSTM SELU Model Parameters:")
            print(f"Trainable Parameters: {lstm_selu_trainable_count:.10f}")
            print(f"Non-trainable Parameters: {lstm_selu_non_trainable_count:.10f}")
            print(f"Total Parameters: {lstm_selu_total_count:.10f}")

            # Save LSTM SELU model performance
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"MSE": lstm_selu_mse, "MAE": lstm_selu_mae, "R²": lstm_selu_r2}, f)

            # Save LSTM SELU model size and time
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Size (B)": lstm_selu_model_file_size, "Training Time (s)": lstm_selu_time_train, "Testing Time (s)": lstm_selu_time_test}, f)

            # Save LSTM SELU model parameters
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Trainable Parameters": lstm_selu_trainable_count, "Non-trainable Parameters": lstm_selu_non_trainable_count, "Total Parameters": lstm_selu_total_count}, f)

            # Plot training history
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM SELU Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_training_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save history
            with open(os.path.join(cells_subdir, 'lstm_selu_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(history, f)

            # For LSTM SELU: Convert pandas Series to numpy array if needed
            if hasattr(y_test, 'values'):
                y_test_lstm_selu_np = y_test.values
                y_pred_lstm_selu_np = y_pred_lstm_selu
            else:
                y_test_lstm_selu_np = y_test
                y_pred_lstm_selu_np = y_pred_lstm_selu

            # Save y_test_lstm_selu_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_test_lstm_selu_np, f)
            # Save y_pred_lstm_selu_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_selu_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_pred_lstm_selu_np, f)
            
            # Create and train LSTM Att model
            input_shape = (X_train_no_val_lstm.shape[1], X_train_no_val_lstm.shape[2])
            lstm_att_model = create_LSTM_with_attention(start_size, start_size / 2, input_shape, 'selu')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # Print model summary
            print(lstm_att_model.summary())
            lstm_att_time_train_start = time()
            history = lstm_att_model.fit(
                X_train_no_val_lstm, y_train_no_val,
                epochs=100,
                batch_size=32,
                validation_data=(X_val_lstm, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            lstm_att_time_train_end = time()
            lstm_att_time_train = lstm_att_time_train_end - lstm_att_time_train_start

            # Save LSTM Att model
            lstm_att_model_file_name = os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.h5')
            lstm_att_model.save(lstm_att_model_file_name)

            # Evaluate LSTM Att model
            lstm_att_time_test_start = time()
            y_pred_lstm_att = lstm_att_model.predict(X_test_lstm).flatten()
            lstm_att_time_test_end = time()
            lstm_att_time_test = lstm_att_time_test_end - lstm_att_time_test_start

            lstm_att_mse = mean_squared_error(y_test, y_pred_lstm_att)
            lstm_att_mae = mean_absolute_error(y_test, y_pred_lstm_att)
            lstm_att_r2 = r2_score(y_test, y_pred_lstm_att)
            lstm_att_model_file_size = os.path.getsize(lstm_att_model_file_name)
            lstm_att_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_att_model.trainable_weights]))
            lstm_att_non_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in lstm_att_model.non_trainable_weights]))
            lstm_att_total_count = lstm_att_trainable_count + lstm_att_non_trainable_count

            print(f"LSTM Att Model Performance:")
            print(f"MSE: {lstm_att_mse:.10f}")
            print(f"MAE: {lstm_att_mae:.10f}")
            print(f"R²: {lstm_att_r2:.10f}")
            print(f"LSTM Att Model Size and Time:")
            print(f"Size (B): {lstm_att_model_file_size:.10f}")
            print(f"Training Time (s): {lstm_att_time_train:.10f}")
            print(f"Testing Time (s): {lstm_att_time_test:.10f}")
            print(f"LSTM Att Model Parameters:")
            print(f"Trainable Parameters: {lstm_att_trainable_count:.10f}")
            print(f"Non-trainable Parameters: {lstm_att_non_trainable_count:.10f}")
            print(f"Total Parameters: {lstm_att_total_count:.10f}")

            # Save LSTM Att model performance
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"MSE": lstm_att_mse, "MAE": lstm_att_mae, "R²": lstm_att_r2}, f)

            # Save LSTM Att model size and time
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Size (B)": lstm_att_model_file_size, "Training Time (s)": lstm_att_time_train, "Testing Time (s)": lstm_att_time_test}, f)

            # Save LSTM Att model parameters
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump({"Trainable Parameters": lstm_att_trainable_count, "Non-trainable Parameters": lstm_att_non_trainable_count, "Total Parameters": lstm_att_total_count}, f)

            # Plot training history
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Att Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_training_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save history
            with open(os.path.join(cells_subdir, 'lstm_att_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(history, f)

            # For LSTM Att: Convert pandas Series to numpy array if needed
            if hasattr(y_test, 'values'):
                y_test_lstm_att_np = y_test.values
                y_pred_lstm_att_np = y_pred_lstm_att
            else:
                y_test_lstm_att_np = y_test
                y_pred_lstm_att_np = y_pred_lstm_att

            # Save y_test_lstm_att_np
            with open(os.path.join(cells_subdir, 'y_test_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_test_lstm_att_np, f)
            # Save y_pred_lstm_att_np
            with open(os.path.join(cells_subdir, 'y_pred_lstm_att_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                pickle.dump(y_pred_lstm_att_np, f)

            if test_bi:
                bi_model = create_seq_model(input_shape, conv1_filters = 0, conv2_filters = 0, conv_kernel_size = 0, num_cells = start_size, dropout = 0.1)
                bi_model_file_name = os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1)) + '.h5'
                callbacks = return_callbacks(bi_model_file_name, 'val_loss')
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                bi_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                # Print model summary
                print(bi_model.summary())
                bi_time_train_start = time()
                history_bi = bi_model.fit(
                    X_train_no_val_lstm, y_train_no_val,
                    validation_data=(X_val_lstm, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                )
                bi_time_train_end = time()
                bi_time_train = bi_time_train_end - bi_time_train_start
                
                # Evaluate Bi model
                bi_time_test_start = time()
                y_pred_bi = bi_model.predict(X_test_lstm).flatten()
                bi_time_test_end = time()
                bi_time_test = bi_time_test_end - bi_time_test_start

                bi_mse = mean_squared_error(y_test, y_pred_bi)
                bi_mae = mean_absolute_error(y_test, y_pred_bi)
                bi_r2 = r2_score(y_test, y_pred_bi)
                bi_model_file_size = os.path.getsize(bi_model_file_name)
                bi_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in bi_model.trainable_weights]))
                bi_non_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in bi_model.non_trainable_weights]))
                bi_total_count = bi_trainable_count + bi_non_trainable_count

                print(f"Bi Model Performance:")
                print(f"MSE: {bi_mse:.10f}")
                print(f"MAE: {bi_mae:.10f}")
                print(f"R²: {bi_r2:.10f}")
                print(f"Bi Model Size and Time:")
                print(f"Size (B): {bi_model_file_size:.10f}")
                print(f"Training Time (s): {bi_time_train:.10f}")
                print(f"Testing Time (s): {bi_time_test:.10f}")
                print(f"Bi Model Parameters:")
                print(f"Trainable Parameters: {bi_trainable_count:.10f}")
                print(f"Non-trainable Parameters: {bi_non_trainable_count:.10f}")
                print(f"Total Parameters: {bi_total_count:.10f}")

                # Save Bi model performance
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump({"MSE": bi_mse, "MAE": bi_mae, "R²": bi_r2}, f)

                # Save Bi model size and time
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump({"Size (B)": bi_model_file_size, "Training Time (s)": bi_time_train, "Testing Time (s)": bi_time_test}, f)

                # Save Bi model parameters
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump({"Trainable Parameters": bi_trainable_count, "Non-trainable Parameters": bi_non_trainable_count, "Total Parameters": bi_total_count}, f)

                # Plot training history
                plt.figure(figsize=(10, 4))
                plt.plot(history_bi.history['loss'], label='Training Loss')
                plt.plot(history_bi.history['val_loss'], label='Validation Loss')
                plt.title('Bi Model Training Cells: ' + str(start_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
                plt.xlabel('Epoch')
                plt.ylabel('Loss (Binary Crossentropy)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_training_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save Bi history
                with open(os.path.join(cells_subdir, 'bi_num_cells_' + str(start_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump(history_bi, f)

                # For Bi: Convert pandas Series to numpy array if needed
                if hasattr(y_test, 'values'):
                    y_test_bi_np = y_test.values
                    y_pred_bi_np = y_pred_bi
                else:
                    y_test_bi_np = y_test
                    y_pred_bi_np = y_pred_bi

                # Save y_test_bi_np
                with open(os.path.join(cells_subdir, 'y_test_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump(y_test_bi_np, f)
                # Save y_pred_bi_np
                with open(os.path.join(cells_subdir, 'y_pred_bi_num_cells_' + str(start_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                    pickle.dump(y_pred_bi_np, f)

            if test_conv:
                for kernel_size in kernel_sizes:
                    # Save the Conv model in subdirectories
                    kernel_subdir = os.path.join(cells_subdir, 'kernel_' + str(kernel_size))
                    os.makedirs(kernel_subdir, exist_ok=True)
                    
                    conv_model = create_seq_model(input_shape, conv1_filters = 5, conv2_filters = 5, conv_kernel_size = 8, num_cells = start_size, dropout = 0.1)
                    conv_model_file_name = os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_soh_model_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1)) + '.h5'
                    callbacks = return_callbacks(conv_model_file_name, 'val_loss')
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                    conv_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                    # Print model summary.
                    print(conv_model.summary())
                    conv_time_train_start = time()
                    history_conv = conv_model.fit(
                        X_train_no_val_lstm, y_train_no_val,
                        validation_data=(X_val_lstm, y_val),
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=1,
                    )
                    conv_time_train_end = time()
                    conv_time_train = conv_time_train_end - conv_time_train_start
                
                    # Evaluate Conv model
                    conv_time_test_start = time()
                    y_pred_conv = conv_model.predict(X_test_lstm).flatten()
                    conv_time_test_end = time()
                    conv_time_test = conv_time_test_end - conv_time_test_start

                    conv_mse = mean_squared_error(y_test, y_pred_conv)
                    conv_mae = mean_absolute_error(y_test, y_pred_conv)
                    conv_r2 = r2_score(y_test, y_pred_conv)
                    conv_model_file_size = os.path.getsize(conv_model_file_name)
                    conv_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in conv_model.trainable_weights]))
                    conv_non_trainable_count = int(np.sum([np.prod(K.get_value(w).shape) for w in conv_model.non_trainable_weights]))
                    conv_total_count = conv_trainable_count + conv_non_trainable_count

                    print(f"Conv Model Performance:")
                    print(f"MSE: {conv_mse:.10f}")
                    print(f"MAE: {conv_mae:.10f}")
                    print(f"R²: {conv_r2:.10f}")
                    print(f"Conv Model Size and Time:")
                    print(f"Size (B): {conv_model_file_size:.10f}")
                    print(f"Training Time (s): {conv_time_train:.10f}")
                    print(f"Testing Time (s): {conv_time_test:.10f}")
                    print(f"Conv Model Parameters:")
                    print(f"Trainable Parameters: {conv_trainable_count:.10f}")
                    print(f"Non-trainable Parameters: {conv_non_trainable_count:.10f}")
                    print(f"Total Parameters: {conv_total_count:.10f}")

                    # Save Conv model performance
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_performance_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump({"MSE": conv_mse, "MAE": conv_mae, "R²": conv_r2}, f)

                    # Save Conv model size and time
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_size_time_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump({"Size (B)": conv_model_file_size, "Training Time (s)": conv_time_train, "Testing Time (s)": conv_time_test}, f)

                    # Save Conv model parameters
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_parameters_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump({"Trainable Parameters": conv_trainable_count, "Non-trainable Parameters": conv_non_trainable_count, "Total Parameters": conv_total_count}, f)

                    # Plot training history
                    plt.figure(figsize=(10, 4))
                    plt.plot(history_conv.history['loss'], label='Training Loss')
                    plt.plot(history_conv.history['val_loss'], label='Validation Loss')
                    plt.title('Conv Model Training Cells: ' + str(start_size) + ' Kernel: ' + str(kernel_size) + ' Number of splits: ' + str(n_splits) + ' Seed: ' + str(random_state) + ' Test fold: ' + str(ii + 1) + ' Validation fold: ' + str(jj + 1))
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss (Binary Crossentropy)')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_training_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
                    plt.close()

                    # Save Conv history
                    with open(os.path.join(kernel_subdir, 'conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_history_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump(history_conv, f)

                    # For Conv: Convert pandas Series to numpy array if needed
                    if hasattr(y_test, 'values'):
                        y_test_conv_np = y_test.values
                        y_pred_conv_np = y_pred_conv
                    else:
                        y_test_conv_np = y_test
                        y_pred_conv_np = y_pred_conv

                    # Save y_test_conv_np
                    with open(os.path.join(kernel_subdir, 'y_test_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump(y_test_conv_np, f)
                    # Save y_pred_conv_np
                    with open(os.path.join(kernel_subdir, 'y_pred_conv_num_cells_' + str(start_size) + '_kernel_size_' + str(kernel_size) + '_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.pkl'), 'wb') as f:
                        pickle.dump(y_pred_conv_np, f)
            
        print("Models saved successfully.")

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
        
        # Generate the comparison plots
        plot_model_comparison(model_performance, 'Performance', file_name_performance)
        plot_model_comparison(model_size_time, 'Size and Time', file_name_size_time)
        plot_model_comparison(model_parameters, 'Parameters', file_name_parameters)

        # Visualize predictions for all models
        plt.figure(figsize=(12, 12))
        nrows = int(np.sqrt(len(y_pred)))
        ncols = int(np.ceil(len(y_pred) / int(np.sqrt(len(y_pred)))))
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
        plt.savefig(os.path.join(models_dir, 'model_predictions_n_splits_' + str(n_splits) + '_seed_' + str(random_state) + '_test_fold_' + str(ii + 1) + '_validation_fold_' + str(jj + 1) + '.png'), dpi=300, bbox_inches='tight')
        plt.close()