import cv2
import os
import matplotlib.pyplot as plt

def ruota_frame(frame, direction):
    if direction == 'left':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'right':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'down':
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        return frame

def ruota_video(video_path, direction):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Non posso aprire il video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Modifica le dimensioni per la rotazione
    if direction in ['left', 'right']:
        output_size = (frame_height, frame_width)
    else:
        output_size = (frame_width, frame_height)

    temp_output = "temp_video.mp4"
    out = cv2.VideoWriter(temp_output, fourcc, fps, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated_frame = ruota_frame(frame, direction)
        out.write(rotated_frame)

    cap.release()
    out.release()

    os.replace(temp_output, video_path)
    print(f"Il video {video_path} è stato ruotato {direction}.")

def mostra_primo_frame_e_ruota(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Non posso aprire il video {video_path}")
        return
    
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Non posso leggere il primo frame del video {video_path}")
        return
    
    # Mostra il primo frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Chiedi all'utente da che parte ruotare
    direction = input("Inserisci la direzione di rotazione (left o right): ").strip().lower()
    if direction not in ['left', 'right', 'down']:
        print("Direzione non valida. Inserire 'left' o 'right' o 'down'.")
        return
    
    ruota_video(video_path, direction)

def elabora_videos_nella_cartella(cartella_path):
    for filename in os.listdir(cartella_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(cartella_path, filename)
            print(f"Elaborazione del video: {filename}")
            mostra_primo_frame_e_ruota(video_path)

# Usa la funzione
cartella_path = 'data/test_script'
elabora_videos_nella_cartella(cartella_path)
