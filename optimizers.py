# optimizers.py
import numpy as np

class BaseOptimizer:
    """Osnovna klasa za sve optimizatore."""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.params_history = []
        self.params = None

    def register_parameters(self, params):
        """Prima početne parametre i čuva ih u istoriji."""
        self.params = np.array(params, dtype=float)
        self.params_history.append(self.params.copy())

    def step(self, gradient):
        """Izvršava jedan korak optimizacije. Mora biti implementirano u podklasama."""
        raise NotImplementedError

    def get_history(self):
        """Vraća istoriju kretanja parametara."""
        return np.array(self.params_history)

# --- Implementacije konkretnih optimizatora ---

class SGD(BaseOptimizer):
    def step(self, gradient):
        self.params -= self.lr * gradient
        self.params_history.append(self.params.copy())

class Adagrad(BaseOptimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.g_squared = None

    def register_parameters(self, params):
        super().register_parameters(params)
        self.g_squared = np.zeros_like(self.params)

    def step(self, gradient):
        self.g_squared += gradient**2
        self.params -= self.lr * gradient / (np.sqrt(self.g_squared) + self.epsilon)
        self.params_history.append(self.params.copy())

class Adadelta(BaseOptimizer):
    def __init__(self, learning_rate=1.0, rho=0.9, epsilon=1e-6):
        super().__init__(learning_rate) # Note: Adadelta nema eksplicitni learning rate, ali ga zadržavamo radi konzistentnosti
        self.rho = rho
        self.epsilon = epsilon
        self.avg_sq_grad = None
        self.avg_sq_update = None

    def register_parameters(self, params):
        super().register_parameters(params)
        self.avg_sq_grad = np.zeros_like(self.params)
        self.avg_sq_update = np.zeros_like(self.params)

    def step(self, gradient):
        self.avg_sq_grad = self.rho * self.avg_sq_grad + (1 - self.rho) * gradient**2
        
        update = - (np.sqrt(self.avg_sq_update + self.epsilon) / np.sqrt(self.avg_sq_grad + self.epsilon)) * gradient
        
        self.avg_sq_update = self.rho * self.avg_sq_update + (1 - self.rho) * update**2
        
        self.params += self.lr * update # lr je ovde obično 1.0
        self.params_history.append(self.params.copy())

class RMSprop(BaseOptimizer):
    def __init__(self, learning_rate=0.01, alpha=0.99, epsilon=1e-8):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
        self.avg_sq_grad = None

    def register_parameters(self, params):
        super().register_parameters(params)
        self.avg_sq_grad = np.zeros_like(self.params)

    def step(self, gradient):
        self.avg_sq_grad = self.alpha * self.avg_sq_grad + (1 - self.alpha) * gradient**2
        self.params -= self.lr * gradient / (np.sqrt(self.avg_sq_grad) + self.epsilon)
        self.params_history.append(self.params.copy())

class Adam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0

    def register_parameters(self, params):
        super().register_parameters(params)
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)

    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        self.params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.params_history.append(self.params.copy())

class AdamW(Adam):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.weight_decay = weight_decay

    def step(self, gradient):
        # Prvo primenjuje regularni Adam korak
        super().step(gradient)
        # Zatim primenjuje weight decay direktno na parametre
        self.params -= self.lr * self.weight_decay * self.params
        # Ažuriramo poslednji unos u istoriji da reflektuje decay
        self.params_history[-1] = self.params.copy()

class Adamax(BaseOptimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.u, self.t = None, None, 0

    def register_parameters(self, params):
        super().register_parameters(params)
        self.m = np.zeros_like(self.params)
        self.u = np.zeros_like(self.params)

    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.u = np.maximum(self.beta2 * self.u, np.abs(gradient))
        
        alpha = self.lr / (1 - self.beta1**self.t)
        self.params -= alpha * self.m / (self.u + self.epsilon)
        self.params_history.append(self.params.copy())

class Nadam(Adam):
    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Nesterov momentum deo
        m_nesterov = (self.beta1 * m_hat + (1 - self.beta1) * gradient / (1 - self.beta1**self.t))
        
        self.params -= self.lr * m_nesterov / (np.sqrt(v_hat) + self.epsilon)
        self.params_history.append(self.params.copy())

class RAdam(Adam):
    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        
        rho_inf = 2 / (1 - self.beta2) - 1
        rho_t = rho_inf - 2 * self.t * (self.beta2**self.t) / (1 - self.beta2**self.t)

        if rho_t > 5.0: # Prag za rektifikaciju
            v_hat = self.v / (1 - self.beta2**self.t)
            r_t = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            self.params -= self.lr * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            self.params -= self.lr * m_hat
        
        self.params_history.append(self.params.copy())

class ASGD(BaseOptimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.ax = None # Averaged parameters
        self.t = 0

    def register_parameters(self, params):
        super().register_parameters(params)
        self.ax = self.params.copy()
        # Za ASGD, istorija prati prosečne parametre
        self.params_history = [self.ax.copy()] 

    def step(self, gradient):
        # Klasičan SGD korak na originalnim parametrima
        self.params -= self.lr * gradient
        self.t += 1
        # Ažuriranje proseka (running average)
        self.ax = (self.ax * (self.t - 1) + self.params) / self.t
        # U istoriju beležimo putanju prosečnih parametara
        self.params_history.append(self.ax.copy())


def get_optimizer_by_name(name, learning_rate, **kwargs):
    """Fabrika funkcija za kreiranje optimizatora na osnovu imena."""
    optimizers = {
        'sgd': SGD,
        'asgd': ASGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'rmsprop': RMSprop,
        'adam': Adam,
        'adamw': AdamW,
        'adamax': Adamax,
        'nadam': Nadam,
        'radam': RAdam
    }
    optimizer_class = optimizers.get(name.lower())
    if optimizer_class:
        # Prosleđuje samo relevantne kwargs za dati optimizator
        import inspect
        sig = inspect.signature(optimizer_class.__init__)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return optimizer_class(learning_rate=learning_rate, **valid_kwargs)
    else:
        raise ValueError(f"Optimizator '{name}' nije definisan.")