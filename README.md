# BOP3000
Repo til bachelor oppgave skøytekamera


Beskrivelse
I kortbaneløp er det en rekke oppgaver som må løses av sekretariatet manuelt. Siden et heat består av 2-6 løpere er det utfordrende å følge antall runder hver løper til enhver tid har igjen. Utøvere kan falle og bli tatt igjen med en eller flere runde osv.

Utøvere har hjelmtrekk med nummer som gjør dem identifiserbare. Til start stiller utøverne seg opp på rekke så systemet kan identifisere og spore dem under løpet.

Ved å bruke ett eller flere fastmonterte kameraer sammen med en modell for objekt detektering, skal det utvikles et program som kan følge utøverne rundt banen i sanntid. Programmet vil kunne holde følge med antall runder, splittider osv. Eventuelle fall vil også kunne identifiseres, og varsles.

Det er viktig at systemet analyserer data i nærmest sanntid. Helst innenfor 5-10 sekunder forsinkelse.

Dataen kan leveres videre til systemer som viser tider på storskjerm. Dette er noe som kan lages om det blir tid/anledning.

Oppgaven blir å finne kameraer som fungerer å bruke i ishaller, identifisere og trene modeller og sy dette sammen til et system som utfører oppgaven


## Keybinds

### Wide camera
| Key | Action |
|-----|--------|
| `r` | Set/replace YOLO ROI (detection region) |
| `o` | Set/replace OCR ROI (number recognition, must be inside YOLO ROI) |
| `f` | Set/replace finish line (for lap counting) |

### Close camera
| Key | Action |
|-----|--------|
| `c` | Set/replace close YOLO ROI (detection region) |
| `v` | Set/replace close OCR ROI (number recognition, must be inside close YOLO ROI) |

### Global
| Key | Action |
|-----|--------|
| `Space` | Pause/resume |
| `Esc` | Quit

## Setup

### Installer dependencies:

Systemet kan åpnes/kjøres på CPU. (Med veldig redusert ytelse)
Men systemet ble utviklet med tanke på at NVIDIA GPU skulle brukes.
Derfor er det anbefalt å kjøre dette på en NVIDIA GPU. Videre antas det at man har lastet ned riktig CUDA drivere.

```pip
pip install -r requirments.txt tensorrt
```

### Config

Programmet leser oppsett fra `data/config.ini`.
Hvis filen ikke finnes, blir den opprettet automatisk ved oppstart med standardverdier.

Viktigste felter som normalt må sjekkes/endres:
#### Input Stream
- `Path.WIDE_SOURCE` - videokilde for wide-kamera
- `Path.CLOSE_SOURCE` - videokilde for close-kamera
#### ROI
- `Path.YOLO_ROI_PATH` - lagret ROI for wide deteksjon
- `Path.OCR_ROI_PATH` - lagret ROI for wide OCR-område
- `Path.CLOSE_YOLO_ROI_PATH` - lagret ROI for close deteksjon
- `Path.CLOSE_OCR_ROI_PATH` - lagret ROI for close OCR-område
#### Homography
- `Path.RINK_WIDE_H_PATH` - homography for wide-kamera
- `Path.RINK_CLOSE_H_PATH` - homography for close-kamera
- `Path.FINISH_LINE_PATH` - lagret mållinje for rundetelling

Modell- og runtime-oppsett ligger også i samme fil, blant annet:

- `Model` for YOLO-modell og TensorRT-engine
- `Inference` for YOLO inference-parametere
- `Runtime` for terskler og frame skip
- `OCR` for OCR-modell, prompt og timeout

Applikasjonen startes med:

```bash
python main.py
```

Ved oppstart åpnes et GUI der du:

1. importerer hjelmnummer
2. setter antall runder
3. starter programmet

For format på hjelmnummerfilen, se `test_nums.csv` i prosjektroten for et eksempel.

### Homogrpahy Setup
```python	
python utilities/make_homography.py --wide WIDE_VIDEO --close CLOSE_VIDEO
```

Sett opp homography basert til banen som systemet skal brukes på.
(Foreløpig forventer skriptet at det sendes inn video og ike direkte kamera input)

#### NOTE: 
	Det er veldig viktig at punktenes rekke følge er samme når du lager H for begge vinklene
	Ellers blir beregningen feil! 

	Den virutelle banen per nå er predefinert med 11 punkter på banen.
	Disse vises top venstre når du har valgt bilde for homography.
	Derfor må du velge en frame der man kan se alle disse puntkene på banen.

(Homography per kamera lagres defualt til img/homography_close.json og img/homogrpahy_wide.json)
