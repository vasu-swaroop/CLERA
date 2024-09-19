import tensorflow as tf

def l1_regularizer(weights_list):
    l1_reg = 0.0
    for weights in weights_list:
        l1_reg += tf.reduce_sum(tf.abs(weights))
    return l1_reg

def full_network(params):
    """
    Define the full network architecture.

    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.

    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    if 'include_tan' in params.keys():
        include_tan = params['include_tan']
    else:
        include_tan = False
    if 'include_log' in params.keys():
        include_log = params['include_log']
    else:
        include_log = False
    if 'include_exp' in params.keys():
        include_exp = params['include_exp']
    else:
        include_exp = False
    if 'include_reciprocal_func' in params.keys():
        include_reciprocal=params['include_reciprocal_func']
    else:
        include_reciprocal= False
    library_dim = params['library_dim']
    model_order = params['model_order']
    assert model_order == 1, 'Only model order 1 supported for now'
    network = {}

    # Define placeholder tensors
    x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
    dx = tf.placeholder(tf.float32, shape=[None, input_dim], name='dx')

    if(params['classify']):
        class_labels=tf.placeholder(tf.float32, shape=[None, params['num_classes']],name='classes')
    # Construct encoder and decoder based on activation type
    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, input_dim, latent_dim)
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, input_dim, latent_dim, params['widths'],  params=params, activation=activation)
    
    # Compute derivatives and construct library based on model order and optional terms
    if model_order == 1:
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf(z, latent_dim, poly_order, include_sine=include_sine, include_tan=include_tan, include_log=include_log, include_exp=include_exp, include_reciprocal_func=include_reciprocal)
    # Initialize sindy_coefficients based on specified initialization method
    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.contrib.layers.xavier_initializer())
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.constant_initializer(0.3))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim,latent_dim], initializer=tf.initializers.random_normal())

    
    # Handle sequential thresholding if applicable
    if params['sequential_thresholding']:
        coefficient_mask = tf.placeholder(tf.float32, shape=[library_dim,latent_dim], name='coefficient_mask')
      
        sindy_predict = tf.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

    if params['classify']==True:
        class_score, classifier_weights, classifier_biases=classifier(z, params['latent_dim'],params['num_classes'],params['classifier_widths'],params,activation)
    # Reconstruct derivatives 
    if model_order == 1:
        dx_decode = z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=activation)
    #Populate the network dictionary
    network['x'] = x
    network['dx'] = dx
    network['z'] = z
    network['dz'] = dz
    network['x_decode'] = x_decode
    network['dx_decode'] = dx_decode
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients
    network['class_labels']=class_labels
    network['class_score']=class_score
    network['classifier_weights']=classifier_weights
    network['classifier_biases']=classifier_biases

    if model_order == 1:
        network['dz_predict'] = sindy_predict

    return network


def define_loss(network, params):
    """
    Create the loss functions for the SINDy model.

    This function computes the various loss components and the overall loss function used for training.

    Arguments:
        network (dict): Dictionary containing the elements of the network architecture.
            This dictionary is usually the output of the full_network() function.
        params (dict): Dictionary containing training parameters.

    Returns:
        loss (Tensor): The overall loss function computed based on the provided loss components and weights.
        losses (dict): Dictionary containing individual loss components.
        loss_refinement (Tensor): Loss function used for refinement training.
        loss_decoder (Tensor): Loss function associated with the decoder part of the SINDy model.
    """
    # Extract necessary components from the network and parameters
    x = network['x']
    x_decode = network['x_decode']
    if(params['classify']):
        class_score=network['class_score']
        class_labels=network['class_labels']
        class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=class_score, labels=class_labels))
    else:
        class_loss=0
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']

    sindy_coefficients = params['coefficient_mask'] * network['sindy_coefficients']
    network_weights= network['encoder_weights']+network['decoder_weights']+network['classifier_weights']
    # Compute individual loss components
    losses = {}
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)
    if params['model_order'] == 1:
        losses['sindy_z'] = tf.reduce_mean((dz - dz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((dx - dx_decode)**2)
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients))
    losses['autoencoder_regularization'] = l1_regularizer(network_weights)
    losses['class']=class_loss
    # Compute overall loss function based on loss components and their respective weights
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']\
           +losses['autoencoder_regularization']*params['autoencoder_regularization']\
           +params['loss_class']*losses['class']
    # Create loss function for refinement training
    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x'] \
                      +losses['autoencoder_regularization']*params['autoencoder_regularization']\
                      +params['loss_class']*losses['class']
    # Create loss function specifically associated with the decoder
   
    return loss, losses, loss_refinement

def classifier(z, input_dim, num_classes, classifier_widths,params,activation='elu'):
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    class_score, classifier_weights, classifier_biases = build_network_layers(z, input_dim, num_classes, classifier_widths, activation_function, 'classifier',params['classifier_weights'])
    return class_score, classifier_weights, classifier_biases

def linear_autoencoder(x, input_dim, latent_dim):
    """
    Construct a linear autoencoder.
    Arguments:
        x (Tensor): Input tensor.
        input_dim (int): Dimension of the input.
        latent_dim (int): Dimension of the latent space.
    Returns:
        z (Tensor): Latent representation tensor.
        x_decode (Tensor): Decoded tensor.
        encoder_weights (List[Tensor]): List of TensorFlow arrays containing the encoder weights.
        encoder_biases (List[Tensor]): List of TensorFlow arrays containing the encoder biases.
        decoder_weights (List[Tensor]): List of TensorFlow arrays containing the decoder weights.
        decoder_biases (List[Tensor]): List of TensorFlow arrays containing the decoder biases.
    """
    z, encoder_weights, encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder')
    x_decode, decoder_weights, decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder')

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def nonlinear_autoencoder(x, input_dim, latent_dim, widths,params, activation='elu'):
    """
    Construct a nonlinear autoencoder.
    Arguments:
        x (Tensor): Input tensor.
        input_dim (int): Dimension of the input.
        latent_dim (int): Dimension of the latent space.
        widths (List[int]): List of widths for hidden layers.
        activation (str): Activation function name ('relu', 'elu', or 'sigmoid').
    Returns:
        z (Tensor): Latent representation tensor.
        x_decode (Tensor): Decoded tensor.
        encoder_weights (List[Tensor]): List of TensorFlow arrays containing the encoder weights.
        encoder_biases (List[Tensor]): List of TensorFlow arrays containing the encoder biases.
        decoder_weights (List[Tensor]): List of TensorFlow arrays containing the decoder weights.
        decoder_biases (List[Tensor]): List of TensorFlow arrays containing the decoder biases.
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('invalid activation function')
    z, encoder_weights, encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder',params['encoder_weights'])
    x_decode, decoder_weights, decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder',params['decoder_weights'])

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name, network_weights):
    """
    Construct one portion of the network (either encoder or decoder).
    Arguments:
        input (Tensor): 2D TensorFlow array, input to the network (shape is [?, input_dim]).
        input_dim (int): Integer, number of state variables in the input to the first layer.
        output_dim (int): Integer, number of state variables to output from the final layer.
        widths (List[int]): List of integers representing how many units are in each network layer.
        activation (function): TensorFlow activation function to be used at each layer.
        name (str): Prefix to be used in naming the TensorFlow variables.
    Returns:
        output (Tensor): TensorFlow array, output of the network layers (shape is [?, output_dim]).
        weights (List[Tensor]): List of TensorFlow arrays containing the network weights.
        biases (List[Tensor]): List of TensorFlow arrays containing the network biases.
    """
    weights = []
    biases = []
    last_width = input_dim
    for i, n_units in enumerate(widths):
        W = tf.get_variable(name + '_W' + str(i),initializer=network_weights[i])
        b = tf.get_variable(name + '_b' + str(i), shape=[n_units],initializer=tf.constant_initializer(0.0))
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)
        last_width = n_units
        weights.append(W)
        biases.append(b)
    
    W = tf.get_variable(name + '_W' + str(len(widths)), initializer=network_weights[len(widths)])
    b = tf.get_variable(name + '_b' + str(len(widths)), shape=[output_dim], initializer=tf.constant_initializer(0.0))
    output = tf.matmul(input, W) + b
    weights.append(W)
    biases.append(b)
    
    return output, weights, biases

def sindy_library_tf(z, latent_dim, poly_order, include_sine=False, include_tan=False, include_log=False, include_exp=False, include_reciprocal_func=False):
    """
    Build the SINDy library.
    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.
    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])
    
    if include_reciprocal_func:
        
        for i in range(latent_dim):
            library.append(1/(1+z[:,i]*z[:,i]))

    if include_tan:
        for i in range(latent_dim):
            library.append(tf.tan(z[:,i]))  

    if include_log:
        for i in range(latent_dim):
            library.append(tf.log(z[:,i]))      

    if include_exp:
        for i in range(latent_dim):
            library.append(tf.exp(z[:,i]))

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)

def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.minimum(tf.exp(input),1.0),
                                  tf.matmul(dz, weights[i]))
            input = tf.nn.elu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            dz = tf.multiply(tf.to_float(input>0), tf.matmul(dz, weights[i]))
            input = tf.nn.relu(input)
        dz = tf.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz = tf.multiply(tf.multiply(input, 1-input), tf.matmul(dz, weights[i]))
        dz = tf.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
    return dz
