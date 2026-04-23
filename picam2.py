from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np

# --- EINSTELLUNGEN ---
PROFIL_NAME = "Erkannt"  # Hier kannst du deinen Namen eintragen
TOLERANZ = 0.5            # Mindest-Übereinstimmung in Prozent (50%), damit ein Gesicht erkannt wird 
# ---------------------

# 1. Schnelle Suche laden (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Kamera Setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# 3. Referenzbild laden
# Stelle sicher, dass 'ich.jpg' im selben Ordner liegt!
print("Lade Referenzbild...")
temp_filename = "/home/admin/Desktop/profilbild.jpg"
referenz_bild = face_recognition.load_image_file(temp_filename)
referenz_encoding = face_recognition.face_encodings(referenz_bild)[0]

print("Erkennung gestartet... Drücke 'q' zum Beenden.")

try:
    while True:
    #ein Bild von der Kamera nehmen
     frame = picam2.capture_array()
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     gesichter = face_cascade.detectMultiScale(gray, 1.1, 5)

     for (x, y, w, h) in gesichter:
            # SCHRITT B: Nur das gefundene Gesicht ausschneiden und für die KI vorbereiten
            gesicht_auschnitt = frame[y:y+h, x:x+w]
            rgb_auschnitt = cv2.cvtColor(gesicht_auschnitt, cv2.COLOR_BGR2RGB)
            
            # KI-Encoding für diesen Ausschnitt berechnen
            aktuelle_encodings = face_recognition.face_encodings(rgb_auschnitt)

            if aktuelle_encodings:
                # Mathematischen Abstand berechnen
                abstand = face_recognition.face_distance([referenz_encoding], aktuelle_encodings[0])[0]
                sicherheit = (1 - abstand) * 100
                
                # Prüfen, ob der Abstand innerhalb der Toleranz liegt
                if abstand < TOLERANZ:
                    anzeige_text = f"{PROFIL_NAME} ({sicherheit:.1f}%)"
                    farbe = (0, 255, 0) # Grün
                else:
                    anzeige_text = "Unbekannt"
                    farbe = (0, 0, 255) # Rot
                
                # Rahmen und Text im Originalbild zeichnen
                cv2.rectangle(frame, (x, y), (x+w, y+h), farbe, 2)
                cv2.putText(frame, anzeige_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)

                # Video-Fenster anzeigen
                cv2.imshow('Gesichtserkennung', frame)
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                 break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
