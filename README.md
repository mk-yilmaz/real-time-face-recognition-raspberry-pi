# v2_advanced_recognition.py (v2)

Dieses Projekt ist die fortgeschrittene Version meines Smart-Camera-Systems. Es nutzt künstliche Intelligenz (Deep Learning), um Personen nicht nur zu finden, sondern eindeutig zu identifizieren.

#Besonderheit: Der Hybrid-Ansatz
Um auf der Hardware des Raspberry Pi 5 Echtzeit-Performance zu erreichen, habe ich ein Hybrid-System implementiert:
1. Detection (Haar Cascades):Sucht blitzschnell nach Gesichtspositionen im Graustufenbild.
2. Recognition (Deep Learning):Berechnet nur für die gefundenen Ausschnitte ein 128-dimensionales Encoding zur Identifizierung.

#Technische Details
- Vektorisierung: Umwandlung biologischer Merkmale in mathematische Encodings mittels der `dlib`-basierten `face_recognition`-Library.
- Euklidische Distanz: Vergleich der Gesichter durch Berechnung des mathematischen Abstands zwischen Vektoren.
- In-Memory Processing: Nutzung von `capture_array()` zur direkten Verarbeitung im RAM (Zero-Disk-IO).

#Performance-Metriken
- Auflösung: 640x480 (VGA) optimiert für Edge-Computing.
- Genauigkeit: Das Gesicht muss zu mindestens 50 % mit dem Original übereinstimmen.

#Installation
```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install opencv-python face_recognition
python main.py
```
