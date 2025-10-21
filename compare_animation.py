import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

# Uvozimo kod iz našeg postojećeg projekta
from functions import get_function_by_name
from optimizers import get_optimizer_by_name

# --- PODESIVI PARAMETRI EKSPERIMENTA ---
OPTIMIZERS_TO_COMPARE = ['sgd', 'nadam']
FUNCTION_NAME = 'beale'
INITIAL_POINT = [1.0, 1.0]
LEARNING_RATE = 0.002
ITERATIONS = 600
OUTPUT_FILE = 'results/uporedna_animacija_sgd_vs_nadam.gif'
# -----------------------------------------

def main():
    print("--- Priprema podataka za uporednu animaciju ---")
    
    # 1. Dohvatamo funkciju i granice za crtanje
    func, grad, bounds = get_function_by_name(FUNCTION_NAME)
    
    # 2. Pokrećemo optimizaciju za oba optimizatora da bismo dobili njihove kompletne putanje
    paths = {}
    for name in OPTIMIZERS_TO_COMPARE:
        print(f"Simuliram putanju za: {name.upper()}")
        optimizer = get_optimizer_by_name(name, learning_rate=LEARNING_RATE)
        optimizer.register_parameters(INITIAL_POINT)
        for _ in range(ITERATIONS):
            params = optimizer.params
            gradient = grad(params[0], params[1])
            optimizer.step(gradient)
        paths[name] = optimizer.get_history()

    sgd_path = paths[OPTIMIZERS_TO_COMPARE[0]]
    nadam_path = paths[OPTIMIZERS_TO_COMPARE[1]]

    print("--- Kreiranje animacije ---")
    
    # 3. Priprema matplotlib figure sa DVA subplot-a (jedan pored drugog)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Uporedni Prikaz - Iteracija: 0', fontsize=16)

    # Priprema pozadine (konturnog grafika) za oba subplota
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, 200)
    y = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    for ax, name in zip([ax1, ax2], OPTIMIZERS_TO_COMPARE):
        ax.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis', norm=LogNorm())
        ax.set_title(name.upper(), fontsize=14)
        ax.set_xlabel('Parametar 1 (x)')
        ax.set_ylabel('Parametar 2 (y)')
        ax.plot(INITIAL_POINT[0], INITIAL_POINT[1], 'rx', markersize=10) # Početna tačka

    # 4. Inicijalizacija linija koje će se animirati
    line1, = ax1.plot([], [], 'o-', color='red', markersize=4, linewidth=2)
    line2, = ax2.plot([], [], 'o-', color='orange', markersize=4, linewidth=2)

    def init():
        """Inicijalizacija animacije (prazni frejmovi)."""
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        """Funkcija koja se poziva za svaki frejm animacije."""
        # Ažuriramo putanju za SGD (levi grafik)
        line1.set_data(sgd_path[:i+1, 0], sgd_path[:i+1, 1])
        
        # Ažuriramo putanju za Nadam (desni grafik)
        line2.set_data(nadam_path[:i+1, 0], nadam_path[:i+1, 1])
        
        # Ažuriramo glavni naslov sa brojem iteracije
        fig.suptitle(f'Uporedni Prikaz - Iteracija: {i}', fontsize=16)
        return line1, line2

    # 5. Kreiranje i čuvanje animacije
    anim = FuncAnimation(fig, animate, init_func=init,
                           frames=ITERATIONS, interval=50, blit=False)
    
    # Sprečava preklapanje naslova
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        anim.save(OUTPUT_FILE, writer='imagemagick', fps=15)
        print(f"\nUporedna animacija uspešno sačuvana u '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"\nGreška pri čuvanju animacije: {e}")
        print("Proverite da li je ImageMagick ispravno instaliran i dostupan u PATH-u.")

if __name__ == '__main__':
    main()