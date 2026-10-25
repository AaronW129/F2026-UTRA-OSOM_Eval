from osom_packages import *

def bias(model : np.array, obs : np.array) -> float:
    '''
    This computes the Bias between
    - model : the model values
    - obs : observed values
    '''
    result = np.average(model - obs)
    return result

def rmse(model : np.array, obs: np.array) -> float: 
    '''
    This computes the Root Mean Squared Error (RMSE) between the 
    - model : the model values
    - obs : observed values
    '''
    result = np.sqrt(np.average((model - obs)**2))
    return result

# KL-divergence given in the "scipy" library: sp.special.kl_div()
# Same inputs as above