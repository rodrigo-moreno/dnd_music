import tkinter as tk
from tkinter import messagebox
from test import main as test_model

genre_embd = {
    'epic': 0,
    'festival': 1,
    'fight': 2,
    'mysterious': 3,
    'romance': 4,
    'sad': 5,
    'tavern': 6,
    'town': 7,
}

def select_genre(genre, root):
    genre_number = genre_embd[genre]
    messagebox.showinfo("Selection", f"You selected {genre}")
    root.selected_genre_number = genre_number
    root.quit()


def gui():
    # Crear la ventana principal
    root = tk.Tk()
    root.title("Select Ambiance")
    root.geometry("300x400")
    root.config(bg="#2c3e50")

    # Crear una etiqueta de bienvenida
    welcome_label = tk.Label(root, text="Choose your ambiance:", font=("Helvetica", 16), bg="#2c3e50", fg="white")
    welcome_label.pack(pady=20)

    # Crear botones para cada opción de ambiente
    for genre in genre_embd:
        button = tk.Button(root, text=genre.capitalize(), font=("Helvetica", 14), width=20, bg="#3498db", fg="black",
                           activebackground="#2980b9", activeforeground="white",
                           command=lambda g=genre: select_genre(g, root))
        button.pack(pady=10)

    # Inicializar el atributo para almacenar el resultado
    root.selected_genre_number = None

    # Ejecutar la interfaz gráfica
    root.mainloop()
    # Obtener el número de género seleccionado y destruir la ventana
    genre_number = root.selected_genre_number
    root.destroy()

    return genre_number


def main():
    genre = gui()  # Llama a la GUI y espera a que se cierre
    test_model(genre)  # Llama a test_model con el género seleccionado


if __name__ == "__main__":
    main()
