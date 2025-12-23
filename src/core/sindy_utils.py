import numpy as np
from scipy.special import binom
from scipy.integrate import odeint,solve_ivp


def library_size(n, poly_order, use_sine=True, include_constant=True, use_tan=False, use_log=False, use_exp=True, use_reciprocal=True):
    """
    Calculate the size of a library of features for a given input dimension and polynomial order.
       
    Arguments:
        n (int): The input dimension or number of variables.
        poly_order (int): The polynomial order for polynomial features.
        use_sine (bool, optional): Whether to include sine terms.
        include_constant (bool, optional): Whether to include the constant term.
        use_tan (bool, optional): Whether to include tangent terms.
        use_log (bool, optional): Whether to include logarithmic terms.
        use_exp (bool, optional): Whether to include exponential terms.
        use_reciprocal (bool, optional): Whether to inclide reciprocal functions
    Returns:
        l (int): The total number of terms in the library.
    """
    l = 0
 
    for k in range(poly_order+1):
        l+=(int(binom(n+k-1,k)))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    if use_reciprocal:    
        l+=n
    if use_tan:
        l += n
    if use_log:
        l += n
    if use_exp:
        l += n
    return l


def sindy_library(X, poly_order, include_sine=True, include_constant=True, use_tan=True, use_log=False, use_exp=True, use_reciprocal=False):
    """
    Construct the library of features for sparse identification of nonlinear dynamics (SINDy). This can be used to 
    simulate the behavior of the nonlinear dynamics of a system using solve_ivp/odeint.
        
    Arguments:
        X (numpy.ndarray): The input data matrix with samples in rows and features in columns.
        poly_order (int): The polynomial order for polynomial features.
        include_sine (bool, optional): Whether to include sine terms.
        use_tan (bool, optional): Whether to include tangent terms.
        use_log (bool, optional): Whether to include logarithmic terms.
        use_exp (bool, optional): Whether to include exponential terms.
        
    Returns:
        library (numpy.ndarray): The constructed library of features.
    """
    n = X.shape[1]  # Number of features (columns in X)

    # Calculate the size of the library
    l = library_size(n, poly_order, include_sine, True, use_tan, use_log, use_exp)
   

    # Initialize the library matrix with ones
    library = np.ones((X.shape[0], l))
    index = 1

    # Add linear terms
    for i in range(n):
        library[:, index] = X[:, i]
        index += 1
    
    if use_reciprocal:
        # Add reciprocal terms
        '''for i in range(n):
            library[:, index] = 1 / (1 + X[:, i])
            index += 1'''
        
        # Add reciprocal squared terms
        for i in range(n):
            library[:, index] = 1 / (1 + X[:, i] * X[:, i])
            index += 1

    # Add tangent terms
    if use_tan:
        for i in range(n):
            library[:, index] = np.tan(X[:, i])
            index += 1
    
    # Add logarithmic terms
    if use_log:
        for i in range(n):
            library[:, index] = np.log(X[:, i])
            index += 1

    if use_exp:
        for i in range(n):
            library[:, index] = np.exp(X[:, i])
            index += 1

    # Add polynomial terms with order > 1
    if poly_order > 1:
        for i in range(n):
            for j in range(i, n):
                library[:, index] = X[:, i] * X[:, j]
                index += 1

    # Add polynomial terms with order > 2
    if poly_order > 2:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    library[:, index] = X[:, i] * X[:, j] * X[:, k]
                    index += 1

    # Add polynomial terms with order > 3
    if poly_order > 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        library[:, index] = X[:, i] * X[:, j] * X[:, k] * X[:, q]
                        index += 1
                    
    # Add polynomial terms with order > 4
    if poly_order > 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for q in range(k, n):
                        for r in range(q, n):
                            library[:, index] = X[:, i] * X[:, j] * X[:, k] * X[:, q] * X[:, r]
                            index += 1

    # Add sine terms
    if include_sine:
        for i in range(n):
            library[:, index] = np.sin(X[:, i])
            index += 1

    return library


def sindy_simulate(x0, t, Xi,  poly_order, include_sine,method='RK45',t_eval=None,use_reciprocal=True, use_tan=False, use_log=False, use_exp=True):
    n = x0.size
    #print(sindy_library(np.array(x0).reshape((1,n)), poly_order, include_sine,use_tan, use_log, use_exp).shape)
    def f(t, x):
        # print('1')
        # Reshape x to a 1D array
        x_reshaped = x.reshape((1,n))
        # Compute the SINDy library term
        sindy_lib_term = sindy_library(x_reshaped, poly_order, include_sine, True, use_tan=use_tan, use_log=use_log, use_exp=use_exp, use_reciprocal=use_reciprocal)
        
        # Compute the derivative using the SINDy library term and Xi
        dxdt = np.dot(sindy_lib_term, Xi)

        
        return dxdt
    x = solve_ivp(f,t,x0,method=method, t_eval=t_eval)
    return x

def sindy_simulate_odeint(x0, t, Xi,  poly_order, include_sine,method='RK45',use_reciprocal=False, use_tan=True, use_log=False, use_exp=True):
    n = x0.size
    #print(sindy_library(np.array(x0).reshape((1,n)), poly_order, include_sine,use_tan, use_log, use_exp).shape)
    def f(t, x):
        print('1')
        # Reshape x to a 1D array
        x_reshaped = x.reshape((1,n))


        # Compute the SINDy library term
        sindy_lib_term = sindy_library(x_reshaped, poly_order, include_sine, True, use_tan=True, use_log=False, use_exp=True, use_reciprocal=False)
        #print(sindy_lib_term.shape)
        # Compute the derivative using the SINDy library term and Xi
        dxdt = np.dot(sindy_lib_term, Xi)
        
        dxdt=dxdt.reshape((n))
        
        return dxdt
    x = odeint(f,x0,t,tfirst=True, full_output=1, printmessg=True)
    return x

def sindy_library_equations(z, latent_dim, poly_order, include_sine=False, include_constant=False, include_tan=False, include_log=False, include_exp=False, include_reciprocal=False):
    """
    Generate a dictionary of library equations for the given input variables. The pattern matches the one used to initialize the SINDy library. This is done to get a string representation of the library equations.

    This function generates a dictionary of library equations based on the input variables and specified options.

    Arguments:
        z (list): List of input variable names.
        latent_dim (int): The number of latent dimensions.
        poly_order (int): The polynomial order for feature creation.
        include_sine (bool): Whether to include sine terms in the library.
        include_constant (bool): Whether to include the constant term in the library.
        include_tan (bool): Whether to include tangent terms in the library.
        include_log (bool): Whether to include logarithmic terms in the library.
        include_exp (bool): Whether to include exponential terms in the library.

    Returns:
        library (dict): A dictionary containing library equations indexed by their corresponding coefficients.
        active_var (dict): A dictionary containing active variables for each coefficient index.
        rename_range_util (list of lists): A list of lists representing the range of coefficients.
    """
    n = len(z)
    l = library_size(n, poly_order, include_constant=include_constant, use_sine=include_sine, use_tan=include_tan, use_log=include_log, use_exp=include_exp, use_reciprocal=include_reciprocal)
    library = {}
    active_var = {i: [] for i in range(l)}  # Initialize active_var with empty sets for each index

    index = 0
    rename_range_util = []

    # Helper function to add a range to rename_range_util
    def add_range(start, end):
        rename_range_util.append([start, end])

    if include_constant:
        library[index] = '1'
        add_range(index, index + 1)
        index += 1

    add_range(index, index + n)
    for i in range(n):
        library[index] = z[i]
        active_var[index].append(i)
        index += 1

    if include_reciprocal:
        add_range(index, index + n)
        for i in range(n):
            library[index] = '1/(1+' + z[i] + '^2)'
            active_var[index].append(i)
            index += 1

    if include_tan:
        add_range(index, index + n)
        for i in range(n):
            library[index] = 'tan(' + z[i] + ')'
            active_var[index].append(i)
            index += 1

    if include_log:
        add_range(index, index + n)
        for i in range(n):
            library[index] = 'log(' + z[i] + ')'
            active_var[index].append(i)
            index += 1

    if include_exp:
        add_range(index, index + n)
        for i in range(n):
            library[index] = 'exp(' + z[i] + ')'
            active_var[index].append(i)
            index += 1

    if poly_order > 1:
        add_range(index, index + n * (n + 1) // 2)
        for i in range(n):
            for j in range(i, n):
                library[index] = z[i] + '*' + z[j]
                active_var[index].append(i)
                active_var[index].append(j)
                index += 1

    if poly_order > 2:
        add_range(index, index + n * (n + 1) * (n + 2) // 6)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    library[index] = z[i] + '*' + z[j] + '*' + z[k]
                    active_var[index].append(i)
                    active_var[index].append(j)
                    active_var[index].append(k)
                    index += 1

    if poly_order > 3:
        add_range(index, index + n * (n + 1) * (n + 2) * (n + 3) // 24)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for p in range(k, n):
                        library[index] = z[i] + '*' + z[j] + '*' + z[k] + '*' + z[p]
                        active_var[index].append(i)
                        active_var[index].append(j)
                        active_var[index].append(k)
                        active_var[index].append(p)
                        index += 1

    if poly_order > 4:
        add_range(index, index + n * (n + 1) * (n + 2) * (n + 3) * (n + 4) // 120)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for p in range(k, n):
                        for q in range(p, n):
                            library[index] = z[i] + '*' + z[j] + '*' + z[k] + '*' + z[p] + '*' + z[q]
                            active_var[index].append(i)
                            active_var[index].append(j)
                            active_var[index].append(k)
                            active_var[index].append(p)
                            active_var[index].append(q)
                            index += 1

    if include_sine:
        add_range(index, index + n)
        for i in range(n):
            library[index] = 'sin(' + z[i] + ')'
            active_var[index].append(i)
            index += 1

    return library, active_var, rename_range_util

def build_equations(coeff_matrix, resulting_library, coeff_thresh, active_terms):
    """
    Build equations based on coefficient matrix and library.

    This function constructs equations for each row of the coefficient matrix using the provided library and coefficient matrix.

    Arguments:
        coeff_matrix (numpy.ndarray): The coefficient matrix from the SINDy model.
        resulting_library (dict): The resulting library of equations generated by `sindy_library_equations`.
        coeff_thresh (float): The threshold below which coefficients are considered negligible.

    Returns:
        equations (list): A list of equations representing the relationships between variables.
    """
    latent_dim = coeff_matrix.shape[0]  # Number of latent dimensions

    # Check if the dimensions match
    if coeff_matrix.shape[1] != len(resulting_library):
        print('error: coeff_matrix.shape[1] != len(resulting_library)')

    equations = []
    active_variable={}
    # Iterate through each row of the coefficient matrix
    for row_index in range(0, latent_dim):
        equation_parts=[]
        active_variable[row_index] = set()
        for col_index, value in resulting_library.items():
            if np.abs(coeff_matrix[row_index][col_index]) > coeff_thresh:
                equation_parts.append(f"{coeff_matrix[row_index][col_index]:.5f}*{value}")
                active_variable[row_index].update(list(active_terms[col_index]))
        # Combine equation parts and create the equation
        equation = f"dz{row_index} = " + ' + '.join(equation_parts)
        equations.append(equation)
    return equations, active_variable


