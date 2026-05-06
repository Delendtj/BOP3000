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
