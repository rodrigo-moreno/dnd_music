"""
Script to generate a simple GUI for better UX.
"""
import tkinter as tk
from tkinter import messagebox

# Genre embedding dictionary maps genre names to numerical identifiers.
genre_embd = {
    "epic": 0,
    "festival": 1,
    "fight": 2,
    "mysterious": 3,
    "romance": 4,
    "sad": 5,
    "tavern": 6,
    "town": 7,
}


def select_genre(genre, root):
    """
    Handle genre selection and display a confirmation message.

    Args:
        genre (str): The genre name selected.
        root (tk.Tk): The root window of the application.
    """
    genre_number = genre_embd[genre]
    messagebox.showinfo("Selection", f"You selected {genre}")
    root.selected_genre_number = genre_number
    root.selected_genre = genre
    root.quit()


def gui():
    """
    Create and run the GUI for genre selection.
    """
    # Set up the main window
    root = tk.Tk()
    root.title("Select Ambiance")
    root.geometry("300x400")
    root.config(bg="#2c3e50")

    # Create a welcome label
    welcome_label = tk.Label(
        root,
        text="Choose your ambiance:",
        font=("Helvetica", 16),
        bg="#2c3e50",
        fg="white",
    )
    welcome_label.pack(pady=20)

    # Create buttons for each genre
    for genre in genre_embd:
        button = tk.Button(
            root,
            text=genre.capitalize(),
            font=("Helvetica", 14),
            width=20,
            bg="#3498db",
            fg="black",
            activebackground="#2980b9",
            activeforeground="white",
            command=lambda g=genre: select_genre(g, root),
        )
        button.pack(pady=10)

    # Initialize attributes to store the selected genre
    root.selected_genre_number = None
    root.selected_genre = None

    # Run the GUI
    root.mainloop()

    # Write the selected genre number to a file and clean up
    if root.selected_genre_number is not None:
        with open("genre_embd.txt", "w") as genre_embd_file:
            genre_embd_file.write(str(root.selected_genre_number))

    # Destroy the root window to ensure clean exit
    root.destroy()


def main():
    """
    Main function to invoke the GUI.
    """
    gui()


if __name__ == "__main__":
    main()
