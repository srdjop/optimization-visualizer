import argparse
import numpy as np
import os  

from functions import get_function_by_name
from optimizers import get_optimizer_by_name
from visualizations import plot_optimization_path, create_animation

def main():
    """Glavna funkcija koja pokreće ceo proces."""
    
    parser = argparse.ArgumentParser(
        description="Vizuelizacija metoda optimizacije za 2D funkcije.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--optimizers', nargs='+', required=True, help="Lista imena optimizatora za poređenje (npr. sgd adam rmsprop).")
    parser.add_argument('--function', type=str, required=True, choices=['quadratic', 'booth', 'beale'], help="Ime funkcije cilja koju treba optimizovati.")
    parser.add_argument('--initial_point', type=float, nargs=2, required=True, metavar=('X', 'Y'), help="Početna tačka za optimizaciju (npr. 8 8).")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Stopa učenja (learning rate) za optimizatore.")
    parser.add_argument('--iterations', type=int, default=100, help="Broj iteracija za optimizaciju.")
    parser.add_argument('--output_file', type=str, default='output.png', help="Ime izlaznog fajla (bez putanje). Ekstenzija (.png, .gif) određuje tip izlaza.")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    args = parser.parse_args()

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True) # Kreira folder 'results' ako ne postoji

    final_output_path = os.path.join(RESULTS_DIR, os.path.basename(args.output_file))

    func, grad, bounds = get_function_by_name(args.function)
    
    paths = {}

    print("--- Pokretanje optimizacije ---")
    optimizer_kwargs = {'beta1': args.beta1, 'beta2': args.beta2}

    for optimizer_name in args.optimizers:
        print(f"Izvršavam {optimizer_name.upper()}...")
        optimizer = get_optimizer_by_name(optimizer_name, args.learning_rate, **optimizer_kwargs)
        optimizer.register_parameters(args.initial_point)
        
        for i in range(args.iterations):
            current_params = optimizer.params
            gradient = grad(current_params[0], current_params[1])
            optimizer.step(gradient)
            
        paths[optimizer_name] = optimizer.get_history()

    print("--- Optimizacija završena ---")
    
    title = f"Optimizatori na funkciji '{args.function}'\nLR={args.learning_rate}, Iteracije={args.iterations}"
    
    if final_output_path.lower().endswith('.png'):
        print(f"Generišem statičnu sliku: {final_output_path}")
        plot_optimization_path(func, paths, bounds, title, final_output_path)
        
    elif final_output_path.lower().endswith(('.gif', '.mp4')):
        if len(args.optimizers) > 1:
            print("Upozorenje: Animacija se može generisati samo za jednog optimizatora.")
            print(f"Koristim prvi navedeni: {args.optimizers[0].upper()}")
        
        first_optimizer_name = args.optimizers[0]
        path_to_animate = paths[first_optimizer_name]
        
        anim_title = f"{first_optimizer_name.upper()} na funkciji '{args.function}'\nLR={args.learning_rate}, Iteracije={args.iterations}"
        
        print(f"Generišem animaciju: {final_output_path}")
        create_animation(func, path_to_animate, bounds, anim_title, final_output_path)
        
    else:
        print("Greška: Podržane ekstenzije za izlazni fajl su .png, .gif, .mp4")

if __name__ == "__main__":
    main()