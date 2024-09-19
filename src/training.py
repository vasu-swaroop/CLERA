import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from autoencoder import full_network, define_loss
import pickle
import time 
import os

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    autoencoder_network = full_network(params)
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    train_op_model_refinement = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    validation_dict = create_feed_dictionary(val_data, params, idxs=None)
    #Calculate norms to be used in the relative losses 
    x_norm = np.mean(val_data['x']**2)
    assert params['model_order'] == 1, 'Only model order 1 supported for now'
    if params['model_order'] == 1:
        sindy_predict_norm_x = np.mean(val_data['dx']**2)

    #Lists to keep track of the training and validation losses. Used to plot the training and validation losses.
    validation_losses = []
    training_losses=[]
    ratios=[]
    ref_validation_losses = []
    ref_training_losses=[]
    ref_ratios=[]

    sindy_model_terms = [np.sum(params['coefficient_mask'])]
    thresh_terms=params['terms']
    loss_feature_names = ['Combined loss', 'Reconstruction loss','SINDy_z loss', 'SINDy_x loss', 'Sindy Regularisation- L1 Norm', 'Autoencoder weights- L1 Norm', 'Classification Loss']

    print('TRAINING')
    if(params['print_progress']):
        print("Legend")
        print(loss_feature_names)
    with tf.Session() as sess:
        # Initialize TensorFlow session
        sess.run(tf.global_variables_initializer())

        # Training loop: iterate through epochs
        for i in range(params['max_epochs']):
            
            # Loop through batches within each epoch
            for j in range(params['epoch_size'] // params['batch_size']):
                
                # Determine batch indices
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                
                # Create feed dictionary for training data
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op, feed_dict=train_dict)
 
            # Calculate the validation predictions for z
            z_validation = sess.run(autoencoder_network['dz'], feed_dict=validation_dict)
            z_norm = np.mean(z_validation ** 2)
            # Check if printing progress is required and perform necessary operations
            if params['print_progress'] and (i % params['print_frequency'] == 0):
                train_loss, val_loss, ratio = (
                    print_progress(
                        sess,
                        i,
                        loss,
                        losses,
                        train_dict,
                        validation_dict,
                        x_norm,
                        sindy_predict_norm_x,
                        z_norm
                    )
                )
                
                # Append losses and ratio values for tracking
                validation_losses.append(val_loss)
                training_losses.append(train_loss)
                ratios.append(ratio)
            
            # Check if sequential thresholding is enabled and perform it
            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 1):
                params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['sindy_coefficients'])) > params['coefficient_threshold']
                validation_dict['coefficient_mask:0'] = params['coefficient_mask']
                terms=np.sum(params['coefficient_mask'])
                # Print the number of active coefficients after thresholding
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
                
                # Update sindy_model_terms for tracking
                sindy_model_terms.append(np.sum(params['coefficient_mask']))
                if thresh_terms is not None and terms<thresh_terms:
                    break
        
        # Refinement phase
        print('REFINEMENT')
        for i_refinement in range(params['refinement_epochs']):
            
            # Loop through batches within each refinement epoch
            for j in range(params['epoch_size'] // params['batch_size']):
                if(math.isnan(sess.run(losses['decoder'], feed_dict=validation_dict))):
                    break
                
                # Determine batch indices
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                
                # Create feed dictionary for training data
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                
                # Perform a refinement training step
                sess.run(train_op_model_refinement, feed_dict=train_dict)
            
            # Check if printing progress is required for refinement phase and perform necessary operations
            if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
                train_loss, val_loss, ratio = (
                    print_progress(
                        sess,
                        i_refinement,
                        loss_refinement,
                        losses,
                        train_dict,
                        validation_dict,
                        x_norm,
                        sindy_predict_norm_x,
                        z_norm
                    )
                )
                
                # Append losses and ratio values for tracking
                ref_validation_losses.append(val_loss)
                ref_training_losses.append(train_loss)
                ref_ratios.append(ratio)


        # Save the trained model and parameters
        saver.save(sess, params['save_folder'] + params['save_name'])

        # Save the parameters to a pickle file
        pickle.dump(params, open(params['save_folder']+ params['save_name'] + '_params.pkl', 'wb'))

        # Calculate the final losses for different components
        final_losses = sess.run(
            (
                losses['decoder'],      # Reconstruction loss
                losses['sindy_x'],      # SINDy x loss
                losses['sindy_z'],      # SINDy z loss
                losses['sindy_regularization'],  # SINDy regularization loss
                losses['autoencoder_regularization'], # Autoencoder regularization loss
                losses['class'] # Classification loss
            ),
            feed_dict=validation_dict
        )

        # Calculate the mean squared value of the time derivatives of input for z predictions
        if params['model_order'] == 1:
            sindy_predict_norm_z = np.mean(
                sess.run(autoencoder_network['dz'], feed_dict=validation_dict) ** 2
            )

        # Get the learned SINDy coefficients
        sindy_coefficients = sess.run(autoencoder_network['sindy_coefficients'], feed_dict={})


        # Create a dictionary to store the results of the training process
        results_dict = {}

        # Store various metrics and information in the results dictionary
        results_dict['num_epochs'] = i
        results_dict['x_norm'] = x_norm
        results_dict['sindy_predict_norm_x'] = sindy_predict_norm_x
        results_dict['sindy_predict_norm_z'] = sindy_predict_norm_z
        results_dict['sindy_coefficients'] = sindy_coefficients
        results_dict['loss_decoder'] = final_losses[0]
        results_dict['loss_decoder_sindy'] = final_losses[1]
        results_dict['loss_sindy'] = final_losses[2]
        results_dict['loss_sindy_regularization'] = final_losses[3]
        results_dict['loss_autoencoder_regularization'] = final_losses[4]
        results_dict['loss_classification']=final_losses[5]
        results_dict['training_losses'] = np.array(training_losses)
        results_dict['validation_losses'] = np.array(validation_losses)
        results_dict['ratios'] = np.array(ratios)
        results_dict['ref_training_losses'] = np.array(ref_training_losses)
        results_dict['ref_validation_losses'] = np.array(ref_validation_losses)
        results_dict['ref_ratios'] = np.array(ref_ratios)
        results_dict['sindy_model_terms'] = np.array(sindy_model_terms)
        
        start=0
        # Adjust arrays to start from a specific epoch based on print frequency
        training_array = np.array(results_dict['training_losses'])
        validation_array = np.array(results_dict['validation_losses'])
        ratios_array = np.array(results_dict['ratios'])

        ref_training_array = np.array(results_dict['ref_training_losses'])
        ref_validation_array = np.array(results_dict['ref_validation_losses'])
        ref_ratios_array = np.array(results_dict['ref_ratios'])


        # Define names and labels for various features and axes
        loss_feature_names = ['Combined loss', 'Reconstruction loss','SINDy_z loss', 'SINDy_x loss', 'Sindy Regularisation- L1 Norm', 'Autoencoder weights- L1 Norm', 'Classifier Loss']
        ratio_feature_names = ['Reconstruction Ratio', 'SINDy Z ratio', 'SINDY X ratio']
        y_axis_loss = 'Error'
        y_axis_ratios = 'Loss Ratios'
        title_params = 'Training before refinement'
        title_params_ref = 'Refinement'
        
        # Plot training and validation curves for losses and ratios
        plot_curves(start, title=params['save_name'] + 'errors', axis_name=y_axis_loss, training_array=training_array, validation_array=validation_array, feature_names=loss_feature_names, params=title_params, validation_exists=True, scale=params['print_frequency'])
        plot_curves(start, title=params['save_name'] + 'errors_ref', axis_name=y_axis_loss, training_array=ref_training_array, validation_array=ref_validation_array, feature_names=loss_feature_names, params=title_params_ref, validation_exists=True, scale=params['print_frequency'])
        plot_curves(start, params['save_name'] + 'ratios', axis_name=y_axis_ratios, training_array=ratios_array, feature_names=ratio_feature_names, params=title_params, validation_exists=False,scale=params['print_frequency'])
        plot_curves(start, params['save_name'] + 'ratios_ref', axis_name=y_axis_ratios, training_array=ref_ratios_array, feature_names=ratio_feature_names, params=title_params_ref, validation_exists=False,scale=params['print_frequency'])


        return results_dict
    

def plot_curves(start,title, feature_names, params, axis_name, training_array, validation_exists=False, validation_array=None, scale=1):
    """
    Plot training and validation curves for various features.
    
    Arguments:
        start - The starting epoch index for plotting.
        title - Title for the plot.
        feature_names - List of names for the features to be plotted.
        params - Dictionary containing parameters used for plotting.
        axis_name - Name for the y-axis.
        training_array - Array containing training data for the features.
        validation_exists - Boolean indicating if validation data exists for plotting.
        validation_array - Array containing validation data for the features.
    """
    num_features = training_array.shape[1]  # Number of features
    num_epochs = training_array.shape[0]     # Number of samples

    # Prepare subplots
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 8*num_features))

    # Loop through each feature
    for i in range(num_features):
        training_feature_values = training_array[:, i]
        
        # Multiply epochs by 50 to scale the x-axis values
        scaled_epochs = np.arange(0, num_epochs)*scale
        
        # Plot the training curve
        axes[i].plot(scaled_epochs, training_feature_values, label='Training')
        
        # Plot the validation curve as dashed
        if validation_exists:
            validation_feature_values = validation_array[:, i]
            axes[i].plot(scaled_epochs, validation_feature_values, '--', label='Validation')
        
        # Set plot title
        axes[i].set_title(feature_names[i])
        
        # Set plot labels
        axes[i].set_xlabel('Epochs')  # Adjusted x-axis label
        axes[i].set_ylabel(axis_name)
        
        # Add legend
        axes[i].legend()

    # Set the main plot title
    fig.suptitle(title)
    
    # Save the plot as an image
    fig.savefig(title + '.png')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the plots
    plt.show()
    time.sleep(1)


def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm, z_norm):
    """
    Print loss function values to keep track of the training progress.
    Arguments:
        sess - the tensorflow session
        i - the training iteration
        loss - tensorflow object representing the total loss function used in training
        losses - tuple of the individual losses that make up the total loss
        train_dict - feed dictionary of training data
        validation_dict - feed dictionary of validation data
        x_norm - float, the mean square value of the input
        sindy_predict_norm - float, the mean square value of the time derivatives of the input.
        Can be first or second order time derivatives depending on the model order.
    Returns:
        Tuple of losses calculated on the validation set.
    """
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    
    print("Epoch %d" % i)
    print("Training loss {0}, {1}".format(training_loss_vals[0],
                                            training_loss_vals[1:]))
    print("Validation loss {0}, {1}".format(validation_loss_vals[0],
                                                validation_loss_vals[1:]))
    decoder_losses = sess.run((losses['decoder'], losses['sindy_x'],losses['sindy_z']), feed_dict=validation_dict)

    loss_ratios = (decoder_losses[0]/x_norm, decoder_losses[1]/sindy_predict_norm,decoder_losses[2]/z_norm)
    print("decoder loss ratio: %f, decoder SINDy loss  ratio: %f, SINDy z loss ratio: %f"% loss_ratios)
    
    return training_loss_vals, validation_loss_vals, loss_ratios


def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow.

    Arguments:
        data - Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params - Dictionary object containing model and training parameters. The relevant
        parameters are model_order (which determines whether the SINDy model predicts first or
        second order time derivatives), sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the SINDy
        model), and learning rate (float that determines the learning rate).
        idxs - Optional array of indices that selects which examples from the dataset are passed
        in to tensorflow. If None, all examples are used.

    Returns:
        feed_dict - Dictionary object containing the relevant data to pass to tensorflow.
    """
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])
    feed_dict = {}
    feed_dict['x:0'] = data['x'][idxs]
    feed_dict['dx:0'] = data['dx'][idxs]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
    feed_dict['learning_rate:0'] = params['learning_rate']
    feed_dict['classes:0']=data['classes'][idxs]
    return feed_dict