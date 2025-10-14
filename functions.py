import numpy as np

# Definišemo rečnik koji će mapirati imena funkcija sa samim funkcijama i njihovim gradijentima
def get_function_by_name(name):
    """Vraća funkciju, njen gradijent i preporučeni opseg za crtanje."""
    if name == 'quadratic':
        func = lambda x, y: x**2 + y**2
        grad = lambda x, y: np.array([2*x, 2*y])
        bounds = [-10, 10, -10, 10] # [xmin, xmax, ymin, ymax]
        return func, grad, bounds
    
    elif name == 'booth':
        # Booth funkcija, minimum u (1,3)
        func = lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2
        grad = lambda x, y: np.array([2*(x + 2*y - 7) + 2*(2*x + y - 5)*2, 
                                      2*(x + 2*y - 7)*2 + 2*(2*x + y - 5)])
        bounds = [-10, 10, -10, 10]
        return func, grad, bounds
        
    elif name == 'beale':
        # Beale funkcija, kompleksnija, minimum u (3, 0.5)
        func = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        grad = lambda x, y: np.array([
            2*(1.5 - x + x*y)*(-1 + y) + 2*(2.25 - x + x*y**2)*(-1 + y**2) + 2*(2.625 - x + x*y**3)*(-1 + y**3),
            2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + 2*(2.625 - x + x*y**3)*(3*x*y**2)
        ])
        bounds = [-4.5, 4.5, -4.5, 4.5]
        return func, grad, bounds
        
    else:
        raise ValueError(f"Funkcija '{name}' nije definisana.")