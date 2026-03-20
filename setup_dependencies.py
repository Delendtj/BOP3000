0# Fil litt bygget på hardware.detector
# sjekker om nvidia gpu finnes
# om ja, installerer cuda pakker
# nei = installer openvino
# denne gjør alt automatisk uten brukerinput
# installerer heller ikke pakker som allerede finnes

# Import-setninger
import subprocess
import sys
import importlib.util


# definerer klassen + oppdager hvilken hardware maskinen har
# ikke nødvendig m/ brukerinput (installerer riktige pakker automatisk)
class DependencyManager:

    def __init__(self):    # konstruktør som kjører når jeg lager objekt
        # intern variable som senere blir "cuda" eller "openvino"
        # starter som None fordi krever ikke hardware ennå
        self.hardware_type = None

    # metode for å sjekke om maskinen har nvidia gpu
    def detect_hardware(self):

        # try-blokk for å håndtere feil om kommandoen ikke finnes
        try:
            # kjører en ekstern kommando
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],

                # fanger output istedet for å skrive i terminal
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # timeout på 3 sekunder
                timeout=3,
                # regner output som tekst i stedet for bytes
                text=True
            )
            # sjekker både om kommandoen ikke feilet
            # og det faktisk finnes GPU-navn i output
            if result.returncode == 0 and result.stdout.strip():
                print(f"Found NVIDIA GPU: {result.stdout.strip()}")
                self.hardware_type = 'cuda'
                return 'cuda'
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # om ikke gpu blir funnet, sett hardware openVINO
        self.hardware_type = 'openvino'
        return 'openvino'

    # priv metode for installasjon av pakker
    def _install(self, package):
        # sjekker om pakken allerede r installert
        if importlib.util.find_spec(package) is not None:
            print(f"{package} already installed")
            return True

        print(f"Installing {package}...")

        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",  # forhindre pip å spørre y/n
                "--quiet",
                package
            ])
            # om installasjon lykkes
            print(f"{package} installed")
            return True

        except subprocess.CalledProcessError:
            # om pip feiler
            print(f"Failed to install {package}")
            return False

    # hovedmeny som starter alt
    # kaller på hardware-detection
    def install_dependencies(self):
        hw = self.detect_hardware()
        # skriver hvilken hardware som ble valgt
        print(f"\nDetected hardware: {hw.upper()}\n")

        if hw == 'cuda':
            # hvis nvidia gpu finnes, installer tensor
            self._install('tensorrt')
            # installer pycuda
            self._install('pycuda')
        else:
            # hvis ingen nvidia gpu, installer openvino
            self._install('openvino')
