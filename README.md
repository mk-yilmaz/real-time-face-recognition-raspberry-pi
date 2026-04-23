# v2_advanced_recognition.py (v2)

# Echtzeit-Gesichtserkennung mit Python (Advanced Version).

Dieses Projekt ist eine erweiterte Version einer einfachen Gesichtserkennung.  
Es verwendet eine Raspberry Pi Kamera sowie OpenCV und die face_recognition Library, um Gesichter in Echtzeit zu erkennen und mit einem Referenzbild zu vergleichen.

---

## Features

- Echtzeit-Kameraübertragung.
- Gesichtserkennung mit OpenCV (Haar Cascade).
- KI-basierte Gesichtserkennung mit face_recognition.
- Vergleich mit Referenzbild.
- Anzeige von Name und Erkennungswahrscheinlichkeit.
- Markierung erkannter Gesichter im Bild (Rahmen + Text).
- Live-Videoanzeige.

---

# Funktionsweise

1. Die Kamera nimmt kontinuierlich Bilder in Echtzeit auf.  
2. Gesichter werden zuerst mit OpenCV lokalisiert. 
3. Der erkannte Bereich wird ausgeschnitten.  
4. Mit face_recognition wird ein Encoding erstellt.  
5. Dieses Encoding wird mit einem Referenzbild verglichen.  
6. Ergebnis wird angezeigt:
   - 🟢 Bekannte Person
   - 🔴 Unbekannte Person

---

# Voraussetzungen

- Python 3.x  
- Raspberry Pi mit Kamera.  
- OpenCV cv2.  
- face_recognition.
- haarcascade_frontalface_default.xml  
  Wird für die klassische Gesichtserkennung (OpenCV Haar Cascade) benötigt.
  Die Datei kann über OpenCV mitgeliefert oder separat heruntergeladen werden.  
- numpy.  
- picamera2.  

Installation:
```bash
pip install opencv-python face-recognition numpy picamera2
