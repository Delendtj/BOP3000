import csv
import os


class HelmetValidator:
    """
    Loads a CSV of valid helmet numbers and validates OCR-read numbers against it.

    CSV format (valid_helmets.csv):
        helmet_number,skater_name
        1,Erik Haugen
        2,Jonas Berg
        ...
    """

    def __init__(self, csv_path: str = "valid_helmets.csv"):
        self.csv_path = csv_path
        self.valid_helmets: dict[str, str] = {}  # number string -> skater name
        self._load_csv()

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            print(f"[HelmetValidator] WARNING: CSV not found at '{self.csv_path}'. No validation will occur.")
            return

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                number = str(row["helmet_number"]).strip()
                name = str(row.get("skater_name", "")).strip()
                self.valid_helmets[number] = name

        print(f"[HelmetValidator] Loaded {len(self.valid_helmets)} valid helmet numbers from '{self.csv_path}'.")

    def is_valid(self, helmet_number: str) -> bool:
        """Return True if the helmet number exists in the CSV."""
        return str(helmet_number).strip() in self.valid_helmets

    def get_skater_name(self, helmet_number: str) -> str | None:
        """Return the skater name for a helmet number, or None if not found."""
        return self.valid_helmets.get(str(helmet_number).strip())

    def validate(self, helmet_number: str) -> dict:
        """
        Validate a helmet number and return a result dict with keys:
            - 'number'  : the input number string
            - 'valid'   : True / False
            - 'name'    : skater name if valid, else None
            - 'status'  : human-readable string e.g. '✅ Erik Haugen (#1)' or '❌ Invalid (#99)'
        """
        number = str(helmet_number).strip()
        if self.is_valid(number):
            name = self.get_skater_name(number)
            return {
                "number": number,
                "valid": True,
                "name": name,
                "status": f"✅ {name} (#{number})",
            }
        else:
            return {
                "number": number,
                "valid": False,
                "name": None,
                "status": f"❌ Invalid (#{number})",
            }
