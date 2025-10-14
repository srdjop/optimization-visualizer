# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

def plot_optimization_path(func, paths, bounds, title, output_file):
    """
    Crta konturni grafik funkcije i putanje jednog ili više optimizatora.

    Args:
        func (callable): Funkcija cilja koja prihvata x i y.
        paths (dict): Rečnik gde je ključ ime optimizatora, a vrednost je niz 
                      koordinata (istorija parametara).
        bounds (list): Opseg za crtanje [xmin, xmax, ymin, ymax].
        title (str): Naslov grafika.
        output_file (str): Putanja za čuvanje slike (npr. 'output.png').
    """
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Korišćenje LogNorm skale može pomoći da se konture bolje vide ako su vrednosti jako zbijene
    contour = ax.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis', norm=LogNorm())
    fig.colorbar(contour, label='Vrednost funkcije cilja (log skala)')

    for name, path in paths.items():
        # Crtamo putanju optimizatora
        ax.plot(path[:, 0], path[:, 1], 'o-', label=name, markersize=3, linewidth=1.5)
        # Označavamo početnu tačku
        ax.plot(path[0, 0], path[0, 1], 'x', color='red', markersize=10, label=f'_start {name}')

    ax.set_title(title)
    ax.set_xlabel('Parametar 1 (x)')
    ax.set_ylabel('Parametar 2 (y)')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(output_file, dpi=150)
    print(f"Slika sačuvana u '{output_file}'")
    plt.close()


def create_animation(func, path, bounds, title, output_file):
    """
    Kreira animaciju putanje jednog optimizatora.

    Args:
        func (callable): Funkcija cilja.
        path (np.array): Niz koordinata (istorija parametara) za jedan optimizator.
        bounds (list): Opseg za crtanje [xmin, xmax, ymin, ymax].
        title (str): Naslov grafika.
        output_file (str): Putanja za čuvanje animacije (npr. 'anim.gif' ili 'anim.mp4').
    """
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis', norm=LogNorm())
    fig.colorbar(contour)

    ax.set_title(title)
    ax.set_xlabel('Parametar 1 (x)')
    ax.set_ylabel('Parametar 2 (y)')
    
    # Elementi koji će se animirati
    line, = ax.plot([], [], 'o-', color='red', markersize=4, linewidth=2)
    iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

    def init():
        """Inicijalizacija animacije."""
        line.set_data([], [])
        iteration_text.set_text('')
        return line, iteration_text

    def animate(i):
        """Funkcija koja se poziva za svaki frejm."""
        # Prikazuje putanju od početka do trenutne iteracije i
        line.set_data(path[:i+1, 0], path[:i+1, 1])
        iteration_text.set_text(f'Iteracija: {i}')
        return line, iteration_text

    # Kreiranje animacije
    anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(path), interval=50, blit=True)

    try:
        # Čuvanje animacije
        anim.save(output_file, writer='imagemagick', fps=15)
        print(f"Animacija sačuvana u '{output_file}'")
    except Exception as e:
        print(f"Greška pri čuvanju animacije: {e}")
        print("Da li je 'imagemagick' (za GIF) ili 'ffmpeg' (za MP4) instaliran i dostupan u PATH-u?")
    
    plt.close()