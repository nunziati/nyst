import cv2
import numpy as np

class FirstFrameAnnotator:
    """
    Class responsible for annotating frames by drawing crosshairs on the detected pupil positions.
    """
    def __init__(self):
        pass

    def apply(self, frame, left_pupil_absolute_position, right_pupil_absolute_position, length:int=50):
        """
        Annotates the given frame with crosshairs at the positions of the left and right pupils.

        Arguments:
        - frame: The image frame  to be annotated.
        - left_pupil_absolute_position: A tuple (x, y) representing the absolute position of the left pupil in the frame.
        - right_pupil_absolute_position: A tuple (x, y) representing the absolute position of the right pupil in the frame.
        - length: The length of the crosshairs to be drawn. Default is 50 pixels.

        Returns:
        - The annotated frame with crosshairs drawn at the positions of the left and right pupils.
        """
        # Get the dimensions of the frame
        rows, cols, _ = frame.shape

        # Extract and convert the x and y coordinates of the left and right pupils to integers
        left_x, left_y = map(int, left_pupil_absolute_position)
        right_x, right_y = map(int, right_pupil_absolute_position)
        
        # Draw crosshairs on the left pupil if coordinates are available
        if left_x is not None and left_y is not None:
            cv2.line(frame, (max(left_x - length // 2, 0), left_y), (min(left_x + length // 2, cols), left_y), (0, 255, 0), 2)
            cv2.line(frame, (left_x, max(left_y - length // 2, 0)), (left_x, min(left_y + length // 2, rows)), (0, 255, 0), 2)

        # Draw crosshairs on the right pupil if coordinates are available
        if right_x is not None and right_y is not None:
            cv2.line(frame, (max(right_x - length // 2, 0), right_y), (min(right_x + length // 2, cols), right_y), (0, 255, 0), 2)
            cv2.line(frame, (right_x, max(right_y - length // 2, 0)), (right_x, min(right_y + length // 2, rows)), (0, 255, 0), 2)

        return frame # Return the annotated frame
    


    def apply_segmentation(self, frame, mask, pos, alpha_background=0.0, alpha_classes=0.1):
        """
        Applica la maschera con trasparenza al frame.
        
        Parameters:
        - frame: Il frame (immagine) su cui applicare la maschera.
        - mask: La maschera multi-classe (valori interi che rappresentano diverse classi).
        - alpha_background: Trasparenza per il background (0.0 = completamente trasparente).
        - alpha_classes: Trasparenza per le classi (1.0 = completamente opaco, valori più bassi = maggiore trasparenza).
        """
        # Assicurati che la maschera e il frame abbiano la stessa dimensione
        if frame.shape[:2] != mask.shape:
            raise ValueError("La dimensione della maschera non corrisponde a quella del frame")
        
        # Crea una copia del frame per la fusione con la maschera
        masked_frame = frame.copy()
        
        # Crea una nuova immagine per la maschera colorata
        mask_colored = np.zeros_like(frame)  # Inizializza una maschera colorata vuota
        
        # Applica i colori per ciascuna classe (1, 2, 3)
        # Classe 1: Rosso
        mask_colored[mask == 1] = [0, 0, 255]  # Red
        
        # Classe 2: Blu
        mask_colored[mask == 2] = [255, 0, 0]  # Blue
        
        # Classe 3: Verde
        mask_colored[mask == 3] = [0, 255, 0]  # Green
        
        # Il background (classe 0) rimarrà trasparente, quindi non modifichiamo il frame per quelli
        
        # Applicazione della trasparenza per il background
        frame_with_transparency = cv2.addWeighted(mask_colored, alpha_classes, frame, 1 - alpha_classes, 0)
        
        # Applicazione della trasparenza per il background (classe 0), che rimarrà completamente trasparente
        masked_frame[mask == 0] = frame[mask == 0]
        
        # Combina il frame colorato con la trasparenza per le altre classi
        masked_frame = cv2.addWeighted(frame_with_transparency, 1 - alpha_background, frame, alpha_background, 0)

        # Mostra il risultato
        cv2.imshow(f"Segmented Frame {pos} eye", masked_frame)
        cv2.waitKey(1)  # Attende 1000 ms (1 secondo)
        


