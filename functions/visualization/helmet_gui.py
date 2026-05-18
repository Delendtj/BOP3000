from __future__ import annotations

import csv
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk


@dataclass
class RaceSetup:
    helmet_numbers: list[str]
    total_laps: int


class HelmetNumberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Race Setup")
        self.root.geometry("420x650")
        self.root.configure(bg="#1e1e2e")

        self.helmet_numbers: list[str] = []
        self.total_laps_var = tk.StringVar(value="3")
        self.selected_file_var = tk.StringVar(value="Ingen fil valgt")
        self.status_var = tk.StringVar(value="Importer hjelmnummer og velg antall runder.")
        self.result: RaceSetup | None = None

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "TButton",
            padding=10,
            relief="flat",
            background="#45475a",
            foreground="white",
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map(
            "TButton",
            background=[("active", "#585b70")],
            foreground=[("active", "#f5e0dc")],
        )
        self.style.configure(
            "Setup.TEntry",
            fieldbackground="#313244",
            foreground="#cdd6f4",
            insertcolor="#cdd6f4",
            borderwidth=1,
        )

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self.setup_ui()
        self.total_laps_var.trace_add("write", self._on_total_laps_changed)
        self._refresh_start_state()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg="#313244", height=70)
        header_frame.pack(fill=tk.X)
        header_label = tk.Label(
            header_frame,
            text="Race Setup",
            bg="#313244",
            fg="#cdd6f4",
            font=("Segoe UI", 16, "bold"),
        )
        header_label.pack(pady=20)

        main_frame = tk.Frame(self.root, bg="#1e1e2e", padx=30, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        import_label = tk.Label(
            main_frame,
            text="Startliste",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Segoe UI", 11, "bold"),
        )
        import_label.pack(anchor=tk.W, pady=(0, 8))

        self.import_btn = ttk.Button(main_frame, text="Velg CSV Fil", command=self.import_csv)
        self.import_btn.pack(fill=tk.X, pady=(0, 10))

        file_label = tk.Label(
            main_frame,
            textvariable=self.selected_file_var,
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Segoe UI", 9, "italic"),
            anchor="w",
        )
        file_label.pack(fill=tk.X, pady=(0, 18))

        laps_label = tk.Label(
            main_frame,
            text="Antall runder",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Segoe UI", 11, "bold"),
        )
        laps_label.pack(anchor=tk.W, pady=(0, 8))

        laps_entry = ttk.Entry(main_frame, textvariable=self.total_laps_var, style="Setup.TEntry")
        laps_entry.pack(fill=tk.X, pady=(0, 6))

        laps_hint = tk.Label(
            main_frame,
            text="Skriv et heltall storre enn 0.",
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Segoe UI", 9),
        )
        laps_hint.pack(anchor=tk.W, pady=(0, 20))

        list_label = tk.Label(
            main_frame,
            text="Importerte numre:",
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Segoe UI", 10),
        )
        list_label.pack(anchor=tk.W, pady=(0, 5))

        list_container = tk.Frame(main_frame, bg="#313244", bd=1, relief="flat")
        list_container.pack(expand=True, fill=tk.BOTH)

        scrollbar = tk.Scrollbar(list_container, bg="#313244")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            list_container,
            bg="#313244",
            fg="#cdd6f4",
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 11),
            selectbackground="#585b70",
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        scrollbar.config(command=self.listbox.yview)

        self.status_label = tk.Label(
            main_frame,
            textvariable=self.status_var,
            bg="#1e1e2e",
            fg="#f38ba8",
            font=("Segoe UI", 9, "italic"),
            wraplength=340,
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.X, pady=10)

        footer_frame = tk.Frame(self.root, bg="#1e1e2e", pady=20)
        footer_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(footer_frame, text="Start Program", command=self.close_and_start)
        self.start_btn.pack(pady=10, padx=50, fill=tk.X)

    def import_csv(self):
        file_path = filedialog.askopenfilename(
            title="Velg CSV fil med hjelmnummer",
            filetypes=[("CSV filer", "*.csv"), ("Tekstfiler", "*.txt"), ("Alle filer", "*.*")],
        )
        if not file_path:
            return

        try:
            temp_numbers: list[str] = []
            with open(file_path, mode="r", encoding="utf-8-sig") as file:
                content = file.read(1024)
                file.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(content) if content else None
                except csv.Error:
                    dialect = None
                reader = csv.reader(file, dialect if dialect else "excel")

                for row in reader:
                    if not row:
                        continue
                    number = str(row[0]).strip()
                    if number:
                        temp_numbers.append(number)

            if not temp_numbers:
                messagebox.showwarning("Tom fil", "CSV-filen inneholder ingen gjenkjennbare nummer.")
                return

            self.helmet_numbers = temp_numbers
            self.selected_file_var.set(file_path)
            self.status_var.set(f"{len(self.helmet_numbers)} hjelmnummer lastet inn.")
            self.status_label.config(fg="#a6e3a1")
            self.update_listbox()
            self._refresh_start_state()
        except Exception as exc:
            messagebox.showerror("Feil ved innlesing", f"Kunne ikke lese filen:\n{exc}")

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for index, number in enumerate(self.helmet_numbers, start=1):
            self.listbox.insert(tk.END, f"{index:02d} | {number}")

    def close_and_start(self):
        total_laps = self._parse_total_laps()
        if not self.helmet_numbers:
            messagebox.showwarning("Mangler data", "Importer hjelmnummer for du starter.")
            return
        if total_laps is None:
            messagebox.showwarning("Ugyldig rundeantall", "Skriv inn et gyldig antall runder.")
            return

        self.result = RaceSetup(
            helmet_numbers=list(self.helmet_numbers),
            total_laps=total_laps,
        )
        self.root.quit()

    def cancel(self):
        self.result = None
        self.root.quit()

    def _parse_total_laps(self):
        raw_value = self.total_laps_var.get().strip()
        if not raw_value:
            return None
        try:
            total_laps = int(raw_value)
        except ValueError:
            return None
        return total_laps if total_laps > 0 else None

    def _on_total_laps_changed(self, *_args):
        if self._parse_total_laps() is None:
            self.status_label.config(fg="#f38ba8")
            if self.helmet_numbers:
                self.status_var.set("Skriv inn et gyldig antall runder for a starte.")
        else:
            if self.helmet_numbers:
                self.status_label.config(fg="#a6e3a1")
                self.status_var.set(f"{len(self.helmet_numbers)} hjelmnummer lastet inn.")
        self._refresh_start_state()

    def _refresh_start_state(self):
        start_ready = bool(self.helmet_numbers) and self._parse_total_laps() is not None
        if start_ready:
            self.start_btn.state(["!disabled"])
        else:
            self.start_btn.state(["disabled"])


def prompt_race_setup():
    root = tk.Tk()
    root.withdraw()

    gui = HelmetNumberGUI(root)

    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.deiconify()
    root.mainloop()

    result = gui.result
    root.destroy()
    return result


def hent_hjelmnummer():
    setup = prompt_race_setup()
    return setup.helmet_numbers if setup is not None else []


if __name__ == "__main__":
    setup_result = prompt_race_setup()
    print(setup_result)
