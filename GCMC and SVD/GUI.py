import tkinter as tk
from tkinter import messagebox, filedialog
import json
import os

# Load recommendation data
def load_recommendations():
    try:
        with open("recommendations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load recommendations.json\n\n{e}")
        return []

# Show recommendations for a specific user
def show_recommendations():
    user_id_str = user_id_entry.get()
    if not user_id_str.isdigit():
        messagebox.showwarning("Invalid Input", "Please enter a valid user ID (positive integer).")
        return

    user_id = int(user_id_str)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f" Recommendations for User {user_id}:\n\n")

    data = load_recommendations()
    found = False
    for r in data:
        if r["user"] == user_id:
            for rec in r["recommendations"]:
                result_text.insert(tk.END, f" {rec['title']}\n")
            found = True
            break

    if not found:
        result_text.insert(tk.END, " No recommendations found for this user.\n")

# Export recommendations to a .txt file
def export_to_txt():
    content = result_text.get(1.0, tk.END).strip()
    if not content:
        messagebox.showinfo("Info", "No recommendation content to export. Please generate results first.")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")],
        title="Save Recommendations"
    )
    if filepath:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Success", f"Recommendations have been saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

# Create the window
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("720x500")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Left control panel
left_frame = tk.Frame(main_frame, width=180, bg="#f0f0f0")
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# User ID input field
tk.Label(left_frame, text="User ID:", bg="#f0f0f0").pack(pady=(20, 5))
user_id_entry = tk.Entry(left_frame, width=10)
user_id_entry.pack(pady=(0, 10))

# Model selection dropdown
tk.Label(left_frame, text="Select Model:", bg="#f0f0f0").pack(pady=(10, 5))
model_var = tk.StringVar(value="gcmc")  # default value
model_options = ["gcmc"]
model_menu = tk.OptionMenu(left_frame, model_var, *model_options)
model_menu.config(width=15)
model_menu.pack(pady=(0, 10))

# Get Recommendations button
recommend_button = tk.Button(left_frame, text="Get Recommendations", command=show_recommendations, width=18, height=2)
recommend_button.pack(pady=10)

# Export button
export_button = tk.Button(left_frame, text="Export to .txt", command=export_to_txt, width=18, height=2)
export_button.pack(pady=10)

# Right output display area
right_frame = tk.Frame(main_frame, bg="white")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

result_text = tk.Text(right_frame, font=("Helvetica", 12), wrap=tk.WORD)
result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()