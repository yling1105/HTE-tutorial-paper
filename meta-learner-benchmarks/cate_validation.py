import pandas as pd
import numpy as np

def influence_func(t_hat, df_val):
    """
    t_hat: (np.array) Estimated cate
    df_val: (pd.DataFrame) With column 'adrd', 'treatment', 0:248(covariates)
    
    """
    
    t_hat = y_hat.flatten()
    y_pred1 = plug_in_model1.predict(x)
    y_pred0 = plug_in_model0.predict(x)
    
    t_plugin = y_pred1 -y_pred0
    plug_in = (t_plugin - t_hat)**2
    
    a = w - pi
    ident = np.array([1] * len(w))
    c = ps * (ident - ps)
    b = np.array([2]*len(w))*w*(w-pi) / c
    
    l_de = (ident-b)*t_plugin**2 + b*y_val*(t_plugin-t_hat) + (-a*(t_plugin-t_hat)**2 + t_hat**2)
    
    return np.sum(l_de) + np.sum(plug_in)