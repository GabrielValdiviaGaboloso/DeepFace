import cv2
import os
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace
import time
import threading
import shutil

DATASET_PATH = "dataset"
cap = None
current_name = ""
photo_count = 0
MAX_PHOTOS = 30  # 🔥 Aumentado de 10 a 30 fotos para mejor precisión
recognizing = False
auto_capture_active = False  # Para modo de captura automática


def start_camera():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None
        cv2.destroyAllWindows()


def reset_deepface_cache():
    if os.path.exists(".deepface"):
        shutil.rmtree(".deepface")
        print("Cache DeepFace borrado")


def create_person_folder():
    global current_name, photo_count

    print("BOTÓN CREAR PRESIONADO")
    
    # Solo resetear cache si es la primera persona o si ya existe
    name = name_entry.get().strip()
    print("Nombre ingresado:", name)

    if not name:
        messagebox.showerror("Error", "Ingresa un nombre primero")
        return

    current_name = name
    photo_count = 0

    person_path = os.path.join(DATASET_PATH, current_name)
    print("Creando carpeta:", person_path)
    
    # Si la carpeta ya existe, resetear cache para regenerar embeddings
    if os.path.exists(person_path):
        reset_deepface_cache()
        messagebox.showwarning("Aviso", f"La carpeta '{current_name}' ya existe. Se sobrescribirán las fotos.")

    os.makedirs(person_path, exist_ok=True)
    print("¿Existe?:", os.path.exists(person_path))

    messagebox.showinfo("Listo", f"Carpeta creada:\n{person_path}")
    # start_camera()
    # show_camera()


def show_camera():
    global cap

    if not cap:
        return

    ret, frame = cap.read()
    if not ret:
        return

    cv2.imshow("Captura de Fotos", frame)
    cv2.waitKey(1)

    root.after(30, show_camera)


def take_photo():
    global photo_count, cap

    if not current_name:
        messagebox.showerror("Error", "Primero crea la carpeta con un nombre")
        return

    if photo_count >= MAX_PHOTOS:
        messagebox.showinfo("Info", f"Ya se capturaron {MAX_PHOTOS} fotos")
        return

    ret, frame = cap.read()
    if not ret:
        return

    person_path = os.path.join(DATASET_PATH, current_name)
    img_path = os.path.join(person_path, f"{photo_count}.jpg")

    cv2.imwrite(img_path, frame)
    photo_count += 1

    # Dar sugerencias de distancia y variación
    if photo_count <= 10:
        sugerencia = "cerca, mirando al frente"
    elif photo_count <= 20:
        sugerencia = "a 1-2 metros, diferentes ángulos (izq/der)"
    else:
        sugerencia = "más lejos, con diferentes expresiones"

    messagebox.showinfo("Foto guardada", f"Foto {photo_count}/{MAX_PHOTOS} guardada\n\n💡 Sugerencia: {sugerencia}")

    if photo_count >= MAX_PHOTOS:
        messagebox.showinfo("✅ Listo", f"Captura completa ({MAX_PHOTOS} fotos). Ya puedes reconocer con mayor precisión.")


def auto_capture_photos():
    """Captura automática de 30 fotos con temporizador"""
    global auto_capture_active, photo_count
    
    if not current_name:
        messagebox.showerror("Error", "Primero crea la carpeta con un nombre")
        return
    
    if photo_count >= MAX_PHOTOS:
        messagebox.showinfo("Info", f"Ya se capturaron {MAX_PHOTOS} fotos")
        return
    
    auto_capture_active = True
    messagebox.showinfo(
        "Captura Automática", 
        f"Se tomarán {MAX_PHOTOS} fotos automáticamente cada 2 segundos.\n\n"
        "📸 Varía tu posición:\n"
        "• Acércate y aléjate\n"
        "• Gira la cabeza ligeramente\n"
        "• Cambia expresiones\n"
        "• Prueba con/sin lentes\n\n"
        "Presiona ESC para cancelar"
    )
    
    threading.Thread(target=auto_capture_loop, daemon=True).start()


def auto_capture_loop():
    """Loop de captura automática"""
    global auto_capture_active, photo_count, cap
    
    while auto_capture_active and photo_count < MAX_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar countdown en la imagen
        display_frame = frame.copy()
        remaining = MAX_PHOTOS - photo_count
        
        cv2.putText(
            display_frame,
            f"Foto {photo_count + 1}/{MAX_PHOTOS}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        
        cv2.putText(
            display_frame,
            "Muevete! Cambia de posicion",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )
        
        cv2.imshow("Captura de Fotos", display_frame)
        
        # Verificar si presionan ESC
        if cv2.waitKey(100) & 0xFF == 27:
            auto_capture_active = False
            break
        
        # Tomar foto
        person_path = os.path.join(DATASET_PATH, current_name)
        img_path = os.path.join(person_path, f"{photo_count}.jpg")
        cv2.imwrite(img_path, frame)
        photo_count += 1
        
        print(f"Foto {photo_count}/{MAX_PHOTOS} capturada automáticamente")
        
        # Esperar 2 segundos antes de la siguiente
        time.sleep(2)
    
    auto_capture_active = False
    
    if photo_count >= MAX_PHOTOS:
        messagebox.showinfo("✅ Completado", f"Se capturaron {MAX_PHOTOS} fotos exitosamente!")
    else:
        messagebox.showinfo("Cancelado", f"Se capturaron {photo_count} fotos antes de cancelar")


def start_recognition():
    stop_camera()

    if not os.path.exists(DATASET_PATH):
        messagebox.showerror("Error", "Dataset vacío")
        return
    
    # Filtrar solo directorios (ignorar archivos .pkl de DeepFace)
    personas = [
        item for item in os.listdir(DATASET_PATH) 
        if os.path.isdir(os.path.join(DATASET_PATH, item))
    ]
    
    if len(personas) == 0:
        messagebox.showerror("Error", "Dataset vacío")
        return

    # Mostrar personas registradas
    personas_str = ", ".join(personas)
    messagebox.showinfo("Personas registradas", f"Se reconocerán: {personas_str}")
    
    # Pregenerar embeddings para acelerar el reconocimiento
    messagebox.showinfo("Info", "Generando embeddings... Esto puede tomar unos segundos.")
    
    t = threading.Thread(target=recognize_loop, daemon=True)
    t.start()


def view_dataset():
    """Muestra las personas registradas en el dataset"""
    if not os.path.exists(DATASET_PATH):
        messagebox.showinfo("Dataset", "No hay personas registradas aún")
        return
    
    # Filtrar solo directorios (ignorar archivos .pkl de DeepFace)
    personas = [
        item for item in os.listdir(DATASET_PATH) 
        if os.path.isdir(os.path.join(DATASET_PATH, item))
    ]
    
    if not personas:
        messagebox.showinfo("Dataset", "No hay personas registradas aún")
        return
    
    info = "Personas registradas:\n\n"
    for persona in personas:
        path = os.path.join(DATASET_PATH, persona)
        num_fotos = len([f for f in os.listdir(path) if f.endswith('.jpg')])
        info += f"• {persona}: {num_fotos} fotos\n"
    
    messagebox.showinfo("Dataset", info)


def clear_dataset():
    """Borra todo el dataset"""
    if not os.path.exists(DATASET_PATH):
        messagebox.showinfo("Info", "No hay dataset para borrar")
        return
    
    respuesta = messagebox.askyesno(
        "Confirmar", 
        "¿Estás seguro de que quieres borrar TODAS las personas registradas?"
    )
    
    if respuesta:
        shutil.rmtree(DATASET_PATH)
        reset_deepface_cache()
        messagebox.showinfo("Listo", "Dataset borrado completamente")


def start_recognition():
    stop_camera()

    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        messagebox.showerror("Error", "Dataset vacío")
        return

    # Pregenerar embeddings para acelerar el reconocimiento
    messagebox.showinfo("Info", "Generando embeddings... Esto puede tomar unos segundos.")
    
    t = threading.Thread(target=recognize_loop, daemon=True)
    t.start()


def recognize_loop():
    global recognizing
    recognizing = True

    cap_rec = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_rec.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_rec.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_time = 0
    face_names = {}  # Diccionario para guardar nombre de cada rostro por posición
    
    # Intervalo entre reconocimientos (en segundos)
    RECOGNITION_INTERVAL = 2.0
    
    # 🔥 UMBRAL DE CONFIANZA - Ajusta este valor entre 0.3 y 0.6
    # Más bajo = más estricto (menos falsos positivos, pero puede no reconocer)
    # Más alto = más flexible (reconoce más fácil, pero más errores)
    CONFIDENCE_THRESHOLD = 0.35  # 35% = MÁS ESTRICTO para evitar confusiones
    
    # Detector de rostros Haar Cascade para pre-detección
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while recognizing:
        ret, frame = cap_rec.read()
        if not ret:
            break

        display_frame = frame.copy()
        
        # Detectar rostros con parámetros más estrictos para evitar falsos positivos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,  # Aumentado de 4 a 5 (más estricto)
            minSize=(80, 80),  # Tamaño mínimo más grande
            maxSize=(400, 400)  # Tamaño máximo para evitar detecciones raras
        )

        # Solo buscar reconocimiento cada X segundos
        current_time = time.time()
        if current_time - last_time > RECOGNITION_INTERVAL:
            last_time = current_time
            face_names = {}  # Limpiar nombres anteriores
            
            # Procesar cada rostro detectado individualmente
            for idx, (x, y, w, h) in enumerate(faces):
                try:
                    # Expandir un poco el área del rostro para mejor contexto
                    margin = 20
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # Verificar que el ROI sea válido
                    if face_roi.size == 0:
                        face_names[idx] = "Error: ROI vacío"
                        continue
                    
                    # Reconocer este rostro específico
                    results = DeepFace.find(
                        img_path=face_roi,
                        db_path=DATASET_PATH,
                        enforce_detection=False,
                        detector_backend="opencv",
                        model_name="Facenet512",
                        distance_metric="cosine",
                        silent=True,
                    )

                    if isinstance(results, list) and len(results) > 0:
                        df = results[0]
                        if len(df) > 0:
                            identity = df.iloc[0]["identity"]
                            distance = df.iloc[0]["distance"]
                            
                            # 🔥 FILTRO DE CONFIANZA - Solo aceptar si la distancia es baja
                            if distance < CONFIDENCE_THRESHOLD:
                                person_name = os.path.basename(os.path.dirname(identity))
                                confidence = f"{(1-distance)*100:.1f}%"
                                face_names[idx] = f"{person_name} ({confidence})"
                            else:
                                # Distancia muy alta = no es nadie conocido
                                face_names[idx] = "Desconocido"
                        else:
                            face_names[idx] = "Desconocido"
                    else:
                        face_names[idx] = "Desconocido"

                except Exception as e:
                    print(f"Error reconociendo rostro {idx}:", e)
                    face_names[idx] = "Error"

        # Dibujar rectángulos y nombres para cada rostro
        for idx, (x, y, w, h) in enumerate(faces):
            # Obtener el nombre para este rostro
            name = face_names.get(idx, "Procesando...")
            
            # Color del rectángulo según resultado
            if "Desconocido" in name or "Error" in name:
                rect_color = (0, 0, 255)  # Rojo
                text_color = (0, 0, 255)
            elif "Procesando" in name:
                rect_color = (0, 165, 255)  # Naranja
                text_color = (0, 165, 255)
            else:
                rect_color = (0, 255, 0)  # Verde
                text_color = (0, 255, 0)
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), rect_color, 2)
            
            # Dibujar el nombre arriba del rectángulo
            cv2.putText(
                display_frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )
        
        # Mostrar configuración actual
        cv2.putText(
            display_frame,
            f"Umbral: {CONFIDENCE_THRESHOLD:.2f} | Rostros: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        
        # Mostrar instrucciones
        cv2.putText(
            display_frame,
            "ESC: salir | +/-: ajustar umbral",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Reconocimiento Facial", display_frame)

        # Teclas de control
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('+') or key == ord('='):  # Aumentar umbral
            CONFIDENCE_THRESHOLD = min(0.60, CONFIDENCE_THRESHOLD + 0.05)
            print(f"Umbral aumentado a: {CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord('-') or key == ord('_'):  # Disminuir umbral
            CONFIDENCE_THRESHOLD = max(0.20, CONFIDENCE_THRESHOLD - 0.05)
            print(f"Umbral reducido a: {CONFIDENCE_THRESHOLD:.2f}")

    recognizing = False
    cap_rec.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Reconocimiento detenido")


# ================= UI =================

root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
root.geometry("400x450")

tk.Label(root, text="Sistema de Reconocimiento Facial", font=("Arial", 16, "bold")).pack(pady=10)

tk.Label(root, text="Nombre de la persona:").pack(pady=5)

name_entry = tk.Entry(root, font=("Arial", 14), width=25)
name_entry.pack(pady=5)

tk.Button(
    root, 
    text="1. Crear carpeta ", 
    command=create_person_folder,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 10, "bold"),
    width=30
).pack(pady=5)

tk.Button(
    root, 
    text="2. Tomar foto manual (30 en total)", 
    command=take_photo,
    bg="#2196F3",
    fg="white",
    font=("Arial", 9, "bold"),
    width=30
).pack(pady=3)

tk.Button(
    root, 
    text="2B. 📸 Captura automática (recomendado)", 
    command=auto_capture_photos,
    bg="#00BCD4",
    fg="white",
    font=("Arial", 9, "bold"),
    width=30
).pack(pady=3)

tk.Button(
    root, 
    text="3. Iniciar Reconocimiento", 
    command=start_recognition,
    bg="#FF9800",
    fg="white",
    font=("Arial", 10, "bold"),
    width=30
).pack(pady=10)

# Botones adicionales
frame_extra = tk.Frame(root)
frame_extra.pack(pady=10)

tk.Button(
    frame_extra, 
    text="Ver personas registradas", 
    command=view_dataset,
    bg="#9C27B0",
    fg="white",
    font=("Arial", 9),
    width=20
).pack(side=tk.LEFT, padx=5)

tk.Button(
    frame_extra, 
    text="Borrar todo el dataset", 
    command=clear_dataset,
    bg="#F44336",
    fg="white",
    font=("Arial", 9),
    width=20
).pack(side=tk.LEFT, padx=5)

tk.Label(root, text="Presiona ESC para cerrar ventanas de cámara", fg="gray").pack(pady=5)

root.mainloop()