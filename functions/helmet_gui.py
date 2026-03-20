import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import csv

class HelmetNumberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hjelmnummer Import")
        self.root.geometry("400x550")
        self.root.configure(bg="#1e1e2e")  # Mørkt moderne tema

        self.helmet_numbers = []

        # Stil-konfigurasjon
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Tilpassede stiler
        self.style.configure("TButton",
                            padding=10,
                            relief="flat",
                            background="#45475a",
                            foreground="white",
                            font=('Segoe UI', 10, 'bold'))
        self.style.map("TButton",
                      background=[('active', '#585b70')],
                      foreground=[('active', '#f5e0dc')])

        self.setup_ui()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#313244", height=70)
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(header_frame,
                               text="Hjelmnummer Importør",
                               bg="#313244",
                               fg="#cdd6f4",
                               font=('Segoe UI', 16, 'bold'))
        header_label.pack(pady=20)

        # Hovedområde
        main_frame = tk.Frame(self.root, bg="#1e1e2e", padx=30, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Import Knapp
        self.import_btn = ttk.Button(main_frame, text="Velg CSV Fil", command=self.import_csv)
        self.import_btn.pack(fill=tk.X, pady=(0, 20))

        # Liste-overskrift
        list_label = tk.Label(main_frame,
                             text="Importerte nummere:",
                             bg="#1e1e2e",
                             fg="#a6adc8",
                             font=('Segoe UI', 10))
        list_label.pack(anchor=tk.W, pady=(0, 5))

        # Liste-container med scroll
        list_container = tk.Frame(main_frame, bg="#313244", bd=1, relief="flat")
        list_container.pack(expand=True, fill=tk.BOTH)

        self.scrollbar = tk.Scrollbar(list_container, bg="#313244")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_container,
            bg="#313244",
            fg="#cdd6f4",
            borderwidth=0,
            highlightthickness=0,
            font=('Consolas', 11),
            selectbackground="#585b70",
            yscrollcommand=self.scrollbar.set
        )
        self.listbox.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.scrollbar.config(command=self.listbox.yview)

        # Status info
        self.status_label = tk.Label(main_frame,
                                    text="Ingen filer valgt",
                                    bg="#1e1e2e",
                                    fg="#f38ba8",
                                    font=('Segoe UI', 9, 'italic'))
        self.status_label.pack(pady=10)

        # Footer / Start Knapp
        footer_frame = tk.Frame(self.root, bg="#1e1e2e", pady=20)
        footer_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(footer_frame, text="Start Program ", command=self.close_and_start)
        self.start_btn.pack(pady=10, padx=50, fill=tk.X)
        self.start_btn.state(['disabled'])

    def import_csv(self):
        file_path = filedialog.askopenfilename(
            title="Velg CSV fil med hjelmnummer",
            filetypes=[("CSV filer", "*.csv"), ("Tekstfiler", "*.txt"), ("Alle filer", "*.*")]
        )

        if not file_path:
            return

        try:
            temp_numbers = []
            # Bruker utf-8-sig for å håndtere BOM fra Excel-eksporterte CSV-er
            with open(file_path, mode='r', encoding='utf-8-sig') as file:
                # Prøver å detektere skilletegn automatisk
                content = file.read(1024)
                file.seek(0)
                dialect = csv.Sniffer().sniff(content) if content else None
                reader = csv.reader(file, dialect if dialect else 'excel')

                for row in reader:
                    if row:
                        # Tar første kolonne, fjerner whitespace
                        num = str(row[0]).strip()
                        if num:
                            temp_numbers.append(num)

            if temp_numbers:
                self.helmet_numbers = temp_numbers
                self.update_listbox()
                self.start_btn.state(['!disabled'])
                self.status_label.config(text=f" {len(self.helmet_numbers)} nummere lastet inn", fg="# a6e3a1")
            else:
                messagebox.showwarning("Tom fil", "CSV-filen inneholder ingen gjenkjennbare nummere.")

        except Exception as e:
            messagebox.showerror("Feil ved innlesing", f"Kunne ikke lese filen:\n{str(e)}")

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, num in enumerate(self.helmet_numbers, 1):
            self.listbox.insert(tk.END, f" {i:02d} |  {num}")

    def close_and_start(self):
        if not self.helmet_numbers:
            messagebox.showwarning("Mangler data", "Vennligst importer hjelmnummer før du starter.")
            return
        self.root.quit()

def hent_hjelmnummer():
    """
    Kall denne funksjonen for å åpne GUI-en.
    Returnerer: En liste (array) med hjelmnummer som strings.
    """
    root = tk.Tk()
    # Hindre at hovedvinduet vises før det er ferdig tegnet
    root.withdraw()

    gui = HelmetNumberGUI(root)

    # Sentrer vinduet på skjermen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    root.deiconify()
    root.mainloop()

    # Hent ut data før vi ødelegger objektet
    resultat = gui.helmet_numbers
    root.destroy()
    return resultat

if __name__ == "__main__":
    # Test-kjøring hvis man kjører denne filen direkte
    nummere = hent_hjelmnummer()
    print(f"Du importerte følgende liste: {nummere}")
