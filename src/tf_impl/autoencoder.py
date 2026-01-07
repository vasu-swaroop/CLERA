import tensorflow as tf
# Enable TF 1.x behavior in TF 2.x for backward compatibility
tf.compat.v1.disable_v2_behavior()

from typing import Dict, List, Tuple, Any, Optional, Callable


def l1_regularizer(weights_list: List[tf.Tensor]) -> tf.Tensor:
    """Calculates the L1 regularization term for a list of weight tensors."""
    l1_reg = 0.0
    for weights in weights_list:
        l1_reg += tf.reduce_sum(tf.abs(weights))
    return l1_reg


def full_network(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define the full network architecture.

    Args:
        params: Dictionary containing the training parameters.

    Returns:
        network: Dictionary containing the tensorflow objects of the network.
    """
    input_dim = params["input_dim"]
    latent_dim = params["latent_dim"]
    activation = params["activation"]
    poly_order = params["poly_order"]

    include_sine = params.get("include_sine", False)
    include_tan = params.get("include_tan", False)
    include_log = params.get("include_log", False)
    include_exp = params.get("include_exp", False)
    include_reciprocal = params.get("include_reciprocal_func", False)

    library_dim = params["library_dim"]
    model_order = params["model_order"]
    assert model_order == 1, "Only model order 1 supported for now"
    
    network = {}

    # Define placeholder tensors
    x = tf.placeholder(tf.float32, shape=[None, input_dim], name="x")
    dx = tf.placeholder(tf.float32, shape=[None, input_dim], name="dx")

    class_labels = None
    if params["classify"]:
        class_labels = tf.placeholder(
            tf.float32, shape=[None, params["num_classes"]], name="classes"
        )

    # Construct encoder and decoder based on activation type
    if activation == "linear":
        (
            z,
            x_decode,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
        ) = linear_autoencoder(x, input_dim, latent_dim)
    else:
        (
            z,
            x_decode,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
        ) = nonlinear_autoencoder(
            x,
            input_dim,
            latent_dim,
            params["widths"],
            params=params,
            activation=activation,
        )

    # Compute derivatives and construct library
    dz = None
    Theta = None
    if model_order == 1:
        dz = z_derivative(x, dx, encoder_weights, encoder_biases, activation=activation)
        Theta = sindy_library_tf(
            z,
            latent_dim,
            poly_order,
            include_sine=include_sine,
            include_tan=include_tan,
            include_log=include_log,
            include_exp=include_exp,
            include_reciprocal_func=include_reciprocal,
        )

    # Initialize sindy_coefficients
    if params["coefficient_initialization"] == "xavier":
        sindy_coefficients = tf.get_variable(
            "sindy_coefficients",
            shape=[library_dim, latent_dim],
            initializer=tf.compat.v1.initializers.glorot_uniform(),
        )
    elif params["coefficient_initialization"] == "specified":
        sindy_coefficients = tf.get_variable(
            "sindy_coefficients", initializer=params["init_coefficients"]
        )
    elif params["coefficient_initialization"] == "constant":
        sindy_coefficients = tf.get_variable(
            "sindy_coefficients",
            shape=[library_dim, latent_dim],
            initializer=tf.constant_initializer(0.3),
        )
    elif params["coefficient_initialization"] == "normal":
        sindy_coefficients = tf.get_variable(
            "sindy_coefficients",
            shape=[library_dim, latent_dim],
            initializer=tf.initializers.random_normal(),
        )
    else:
        raise ValueError(
            f"Unknown coefficient initialization: {params['coefficient_initialization']}"
        )

    # Handle sequential thresholding
    sindy_predict = None
    if params["sequential_thresholding"]:
        coefficient_mask = tf.placeholder(
            tf.float32, shape=[library_dim, latent_dim], name="coefficient_mask"
        )
        sindy_predict = tf.matmul(Theta, coefficient_mask * sindy_coefficients)
        network["coefficient_mask"] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

    class_score = None
    classifier_weights = None
    classifier_biases = None
    
    if params["classify"]:
        class_score, classifier_weights, classifier_biases = classifier(
            z,
            params["latent_dim"],
            params["num_classes"],
            params["classifier_widths"],
            params,
            activation,
        )

    # Reconstruct derivatives
    dx_decode = None
    if model_order == 1:
        dx_decode = z_derivative(
            z, sindy_predict, decoder_weights, decoder_biases, activation=activation
        )

    # Populate the network dictionary
    network["x"] = x
    network["dx"] = dx
    network["z"] = z
    network["dz"] = dz
    network["x_decode"] = x_decode
    network["dx_decode"] = dx_decode
    network["encoder_weights"] = encoder_weights
    network["encoder_biases"] = encoder_biases
    network["decoder_weights"] = decoder_weights
    network["decoder_biases"] = decoder_biases
    network["Theta"] = Theta
    network["sindy_coefficients"] = sindy_coefficients
    network["class_labels"] = class_labels
    
    if params["classify"]:
        network["class_score"] = class_score
        network["classifier_weights"] = classifier_weights
        network["classifier_biases"] = classifier_biases

    if model_order == 1:
        network["dz_predict"] = sindy_predict

    return network


def define_loss(
    network: Dict[str, Any], params: Dict[str, Any]
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], tf.Tensor]:
    """
    Create the loss functions for the SINDy model.

    Args:
        network: Dictionary containing the elements of the network architecture.
        params: Dictionary containing training parameters.

    Returns:
        loss: The overall loss function.
        losses: Dictionary containing individual loss components.
        loss_refinement: Loss function used for refinement training.
    """
    x = network["x"]
    x_decode = network["x_decode"]
    
    class_loss = 0.0
    if params["classify"]:
        class_score = network["class_score"]
        class_labels = network["class_labels"]
        class_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=class_score, labels=class_labels
            )
        )

    dz = None
    dz_predict = None
    dx = None
    dx_decode = None

    if params["model_order"] == 1:
        dz = network["dz"]
        dz_predict = network["dz_predict"]
        dx = network["dx"]
        dx_decode = network["dx_decode"]

    sindy_coefficients = params["coefficient_mask"] * network["sindy_coefficients"]
    
    network_weights = network["encoder_weights"] + network["decoder_weights"]
    if params["classify"]:
         network_weights += network["classifier_weights"]
         
    # Compute individual loss components
    losses = {}
    losses["decoder"] = tf.reduce_mean((x - x_decode) ** 2)
    
    if params["model_order"] == 1:
        losses["sindy_z"] = tf.reduce_mean((dz - dz_predict) ** 2)
        losses["sindy_x"] = tf.reduce_mean((dx - dx_decode) ** 2)
        
    losses["sindy_regularization"] = tf.reduce_mean(tf.abs(sindy_coefficients))
    losses["autoencoder_regularization"] = l1_regularizer(network_weights)
    losses["class"] = class_loss

    # Compute overall loss function
    loss = (
        params["loss_weight_decoder"] * losses["decoder"]
        + params["loss_weight_sindy_z"] * losses["sindy_z"]
        + params["loss_weight_sindy_x"] * losses["sindy_x"]
        + params["loss_weight_sindy_regularization"] * losses["sindy_regularization"]
        + losses["autoencoder_regularization"] * params["autoencoder_regularization"]
        + params["loss_class"] * losses["class"]
    )

    # Create loss function for refinement training
    loss_refinement = (
        params["loss_weight_decoder"] * losses["decoder"]
        + params["loss_weight_sindy_z"] * losses["sindy_z"]
        + params["loss_weight_sindy_x"] * losses["sindy_x"]
        + losses["autoencoder_regularization"] * params["autoencoder_regularization"]
        + params["loss_class"] * losses["class"]
    )

    return loss, losses, loss_refinement


def classifier(
    z: tf.Tensor,
    input_dim: int,
    num_classes: int,
    classifier_widths: List[int],
    params: Dict[str, Any],
    activation: str = "elu",
) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
    
    if activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "elu":
        activation_function = tf.nn.elu
    elif activation == "sigmoid":
        activation_function = tf.sigmoid
    else:
        raise ValueError("invalid activation function")
        
    class_score, classifier_weights, classifier_biases = build_network_layers(
        z,
        input_dim,
        num_classes,
        classifier_widths,
        activation_function,
        "classifier",
        params["classifier_weights"],
    )
    return class_score, classifier_weights, classifier_biases


def linear_autoencoder(
    x: tf.Tensor, input_dim: int, latent_dim: int
) -> Tuple[
    tf.Tensor,
    tf.Tensor,
    List[tf.Tensor],
    List[tf.Tensor],
    List[tf.Tensor],
    List[tf.Tensor],
]:
    """
    Construct a linear autoencoder.
    """
    z, encoder_weights, encoder_biases = build_network_layers(
        x, input_dim, latent_dim, [], None, "encoder"
    )
    x_decode, decoder_weights, decoder_biases = build_network_layers(
        z, latent_dim, input_dim, [], None, "decoder"
    )

    return (
        z,
        x_decode,
        encoder_weights,
        encoder_biases,
        decoder_weights,
        decoder_biases,
    )


def nonlinear_autoencoder(
    x: tf.Tensor,
    input_dim: int,
    latent_dim: int,
    widths: List[int],
    params: Dict[str, Any],
    activation: str = "elu",
) -> Tuple[
    tf.Tensor,
    tf.Tensor,
    List[tf.Tensor],
    List[tf.Tensor],
    List[tf.Tensor],
    List[tf.Tensor],
]:
    """
    Construct a nonlinear autoencoder.
    """
    if activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "elu":
        activation_function = tf.nn.elu
    elif activation == "sigmoid":
        activation_function = tf.sigmoid
    else:
        raise ValueError("invalid activation function")

    z, encoder_weights, encoder_biases = build_network_layers(
        x,
        input_dim,
        latent_dim,
        widths,
        activation_function,
        "encoder",
        params["encoder_weights"],
    )
    x_decode, decoder_weights, decoder_biases = build_network_layers(
        z,
        latent_dim,
        input_dim,
        widths[::-1],
        activation_function,
        "decoder",
        params["decoder_weights"],
    )

    return (
        z,
        x_decode,
        encoder_weights,
        encoder_biases,
        decoder_weights,
        decoder_biases,
    )


def build_network_layers(
    input_tensor: tf.Tensor,
    input_dim: int,
    output_dim: int,
    widths: List[int],
    activation: Optional[Callable],
    name: str,
    network_weights: Optional[List[Any]] = None,
) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
    """
    Construct one portion of the network (either encoder or decoder).
    """
    weights = []
    biases = []
    
    # Allow passing explicit weights, otherwise None implies initialization logic elsewhere 
    # (though original code assumed network_weights is present)
    if network_weights is None:
         # Fallback or handle error if required by logic. 
         # Original code crashed if network_weights index out of bounds?
         # Assuming passed correctly for now as per legacy logic.
         pass

    current_input = input_tensor
    
    for i, n_units in enumerate(widths):
        W = tf.get_variable(
            name + "_W" + str(i), initializer=network_weights[i]
        )
        b = tf.get_variable(
            name + "_b" + str(i),
            shape=[n_units],
            initializer=tf.constant_initializer(0.0),
        )
        current_input = tf.matmul(current_input, W) + b
        if activation is not None:
            current_input = activation(current_input)
        
        weights.append(W)
        biases.append(b)

    W = tf.get_variable(
        name + "_W" + str(len(widths)), initializer=network_weights[len(widths)]
    )
    b = tf.get_variable(
        name + "_b" + str(len(widths)),
        shape=[output_dim],
        initializer=tf.constant_initializer(0.0),
    )
    output = tf.matmul(current_input, W) + b
    weights.append(W)
    biases.append(b)

    return output, weights, biases


def sindy_library_tf(
    z: tf.Tensor,
    latent_dim: int,
    poly_order: int,
    include_sine: bool = False,
    include_tan: bool = False,
    include_log: bool = False,
    include_exp: bool = False,
    include_reciprocal_func: bool = False,
) -> tf.Tensor:
    """
    Build the SINDy library.
    """
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:, i])

    if include_reciprocal_func:
        for i in range(latent_dim):
            library.append(1 / (1 + z[:, i] * z[:, i]))

    if include_tan:
        for i in range(latent_dim):
            library.append(tf.tan(z[:, i]))

    if include_log:
        for i in range(latent_dim):
            library.append(tf.log(z[:, i]))

    if include_exp:
        for i in range(latent_dim):
            library.append(tf.exp(z[:, i]))

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(tf.multiply(z[:, i], z[:, j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    library.append(z[:, i] * z[:, j] * z[:, k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        library.append(z[:, i] * z[:, j] * z[:, k] * z[:, p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j, latent_dim):
                    for p in range(k, latent_dim):
                        for q in range(p, latent_dim):
                            library.append(
                                z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]
                            )

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:, i]))

    return tf.stack(library, axis=1)


def z_derivative(
    input_tensor: tf.Tensor,
    dx: tf.Tensor,
    weights: List[tf.Tensor],
    biases: List[tf.Tensor],
    activation: str = "elu",
) -> tf.Tensor:
    """
    Compute the first order time derivatives by propagating through the network.
    """
    dz = dx
    current_input = input_tensor
    
    if activation == "elu":
        for i in range(len(weights) - 1):
            current_input = tf.matmul(current_input, weights[i]) + biases[i]
            dz = tf.multiply(
                tf.minimum(tf.exp(current_input), 1.0), tf.matmul(dz, weights[i])
            )
            current_input = tf.nn.elu(current_input)
        dz = tf.matmul(dz, weights[-1])
        
    elif activation == "relu":
        for i in range(len(weights) - 1):
            current_input = tf.matmul(current_input, weights[i]) + biases[i]
            # Use tf.cast for float conversion in recent TF versions, but maintaining compatibility
            dz = tf.multiply(tf.cast(current_input > 0, tf.float32), tf.matmul(dz, weights[i]))
            current_input = tf.nn.relu(current_input)
        dz = tf.matmul(dz, weights[-1])
        
    elif activation == "sigmoid":
        for i in range(len(weights) - 1):
            current_input = tf.matmul(current_input, weights[i]) + biases[i]
            current_input = tf.sigmoid(current_input)
            dz = tf.multiply(
                tf.multiply(current_input, 1 - current_input), tf.matmul(dz, weights[i])
            )
        dz = tf.matmul(dz, weights[-1])
        
    else:
        for i in range(len(weights) - 1):
            dz = tf.matmul(dz, weights[i])
        dz = tf.matmul(dz, weights[-1])
        
    return dz

