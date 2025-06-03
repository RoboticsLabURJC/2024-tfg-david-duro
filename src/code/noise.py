def add_origin_perturbation(origin, magnitude=0.001): 
    """Add noise to origin""" 
    perturbation = np.random.uniform(-magnitude, magnitude, size=2) 
    return origin + np.array([perturbation[0], perturbation[1], 0]) 
