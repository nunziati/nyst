import numpy as np
import cv2
import dlib
from scipy.ndimage import gaussian_filter, convolve


# PRE-PROCESSING CLASS FOR SIGNALS FILTERING

class PreprocessingSignalsVideos:
    def __init__(self):
        pass
    
    # Convert None values in the array to np.nan
    def convert_none_to_nan(self, array):
        '''
        Converts None values in the array to np.nan.

        Args:
        - array (np.array): A 2D numpy array that may contain None values.

        Returns:
        - np.array: The input array with None values replaced by np.nan.
        '''
        array = np.array(array, dtype=object)  # Ensure array is of object type
        array = np.where(array == None, np.nan, array)
        return array.astype(float)

    # Interpolate NaN values in a 2D array of positions by replacing them with the average of the previous and next valid values
    def interpolate_nans(self, positions:np.array):
        '''
        Interpolates NaN values in a 2D array of positions by replacing them with the average of the previous and next valid values.

        Args:
        - positions (np.array): A 2D numpy array where each element is an array [x, y], with possible NaN values.

        Returns:
        - np.array: The input array with NaN values replaced by interpolated values.
        '''
        
        # Convert None to np.nan
        positions = self.convert_none_to_nan(positions)
        # Elements count
        n = len(positions)
       
        # Replace initial NaN positions with the first valid value
        if np.isnan(positions[0]).any():
            first_valid_index = np.where(~np.isnan(positions).any(axis=1))[0][0]
            positions[0] = positions[first_valid_index]

        # Replace final NaN positions with the last valid value
        if np.isnan(positions[-1]).any():
            last_valid_index = np.where(~np.isnan(positions).any(axis=1))[0][-1]
            positions[-1] = positions[last_valid_index]
        
        # Flag initialization and counter 
        flag = True
        i = 0

        # Loop through the array to identify and interpolate NaNs
        while flag:
            if np.isnan(positions[i]).any():
                start = i-1 # Start index of interpolation
                while i < n and np.isnan(positions[i]).any():
                    i += 1
                end = i # End index of interpolation
                
                # Interpolate between the previous valid value and the next valid value of the sequence NaN
                prev_val = positions[start]
                next_val = positions[end]
                interpolated_val = (prev_val + next_val) / 2
                
                # Sostituzione dei NaN con il valore interpolato
                positions[(start+1):end] = np.rint(interpolated_val).astype(int)
            else:
                # Flag check
                if i == n-1:
                    flag = False
                
                i += 1
        # Ensure the entire array is of integer type
        positions = positions.astype(int)
                
        return positions


# PRE-PROCESSING CLASS FOR FRAME QUALITY IMPROVEMENT

class PreprocessingFramesVideos:
    def __init__(self):
        pass

    # Apply Gaussian filter to an image to reduce noise
    def apply_gaussian_filter(self, frame, sigma=1):
        '''
        Applies a Gaussian filter to an image to reduce noise.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.
        - sigma (float): The standard deviation for Gaussian kernel.

        Returns:
        - np.array: The filtered image.
        '''
        return gaussian_filter(frame, sigma=sigma)

    # Apply binomial filter (approximation to Gaussian filter) to smooth the image
    def apply_binomial_filter(self, frame):
        '''
        Applies a binomial filter to smooth the image.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.

        Returns:
        - np.array: The filtered image.
        '''
        # Binomial kernel
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]]) / 16.0
        return convolve(frame, kernel)

    # Enhance edges to make the contours more distinguishable
    def apply_edge_enhancement(self, frame):
        '''
        Enhances edges in the image to make the contours more distinguishable.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.

        Returns:
        - np.array: The image with enhanced edges.
        '''
        # Edge enhancement kernel (Laplacian filter)
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(frame, -1, kernel)

      
    # Apply super resolution to improve image quality
    def apply_super_resolution(self, frame):
        '''
        Applies super resolution to improve image quality.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.

        Returns:
        - np.array: The image with enhanced resolution.
        '''
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('EDSR_x3.pb')
        sr.setModel('edsr', 3)
        return sr.upsample(frame)
    

    # Remove reflections from the image
    def remove_reflections(self, frame):
        '''
        Removes reflections from an image.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.

        Returns:
        - np.array: The image with reduced reflections.
        '''

        # Apply a median filter to reduce noise
        median = cv2.medianBlur(frame, 5)

        # Use adaptive thresholding to create a mask of the reflections
        thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Use morphology to improve the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find the contours of the reflections
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for the reflections
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Use the mask for inpainting (repairing) the image
        result = cv2.inpaint(self.frame, mask, 3, cv2.INPAINT_TELEA)

        return result

    # Apply all preprocessing steps to a frame
    def preprocess_frame(self, frame):
        '''
        Applies a series of preprocessing steps to improve frame quality before analysis.

        Args:
        - frame (np.array): A 2D or 3D numpy array representing the image.

        Returns:
        - np.array: The preprocessed image.
        '''
        # Normal frame
        cv2.imshow('Image', frame)
        cv2.waitKey(1)  # Wait for a key press to proceed

        # Convert the image from RGB to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale Image', frame_gray)
        cv2.waitKey(1)  # Wait for a key press to proceed
        
        # Apply Gaussian filter
        frame_gaussian = self.apply_gaussian_filter(frame_gray, sigma=1)
        cv2.imshow('After Gaussian Filter', frame_gaussian)
        cv2.waitKey(1)

        # Apply binomial filter
        frame_binomial = self.apply_binomial_filter(frame_gaussian)
        cv2.imshow('After Binomial Filter', frame_binomial)
        cv2.waitKey(1)

        # Apply edge enhancement
        frame_edge_enhanced = self.apply_edge_enhancement(frame_binomial)
        cv2.imshow('After Edge Enhancement', frame_edge_enhanced)
        cv2.waitKey(1)

        # Apply super resolution (commented out)
        # frame = self.apply_super_resolution(frame)
        
        # Remove reflections
        frame_no_reflections = self.remove_reflections(frame_edge_enhanced)
        cv2.imshow('After Removing Reflections', frame_no_reflections)
        cv2.waitKey(1)

        # Convert the image back to BGR
        frame_bgr = cv2.cvtColor(frame_no_reflections, cv2.COLOR_GRAY2BGR)

        cv2.destroyAllWindows()  # Close all windows after the process

        return frame_bgr

    


if __name__ == '__main__':
    
    # Crea un'istanza della classe
    preprocessor = PreprocessingFramesVideos()

    # Apri il video
    cap = cv2.VideoCapture('D:/nyst_flatten_video/001_001.mp4')

    # Ottieni le proprietà del video originale
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Definisci il codec e crea l'oggetto VideoWriter
    out = cv2.VideoWriter('D:/preprocessed_video.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, 
                          (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Applica il preprocessing al frame
        preprocessed_frame = preprocessor.preprocess_frame(frame)

        # Scrivi il frame pre-elaborato nel nuovo video
        out.write(preprocessed_frame)

        # Mostra il frame pre-elaborato
        cv2.imshow('Preprocessed Frame', preprocessed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia il video e chiudi tutte le finestre
    cap.release()
    out.release()
    cv2.destroyAllWindows()


