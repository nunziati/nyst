import cv2

class VideoProcessor:
    def __init__(self):
        self.invert_resolution = False
        self.rotation_type = self.select_rotation_type()

    def select_rotation_type(self):
        while True:
            print("\nSelect the video rotation type:\n")
            print("1. 90 degrees clockwise\n")
            print("2. 90 degrees counter clockwise\n")
            print("3. 180 degrees\n")
            print("4. No rotation\n\n")
            choice = input("Enter your choice (1/2/3/4):\n ")
            
            if choice == '1':
                self.invert_resolution = True
                return cv2.ROTATE_90_CLOCKWISE
            elif choice == '2':
                self.invert_resolution = True
                return cv2.ROTATE_90_COUNTERCLOCKWISE
            elif choice == '3':
                return cv2.ROTATE_180
            elif choice == '4':
                return None
            else:
                print("\t-> Invalid choice. Please select a valid option.")

    