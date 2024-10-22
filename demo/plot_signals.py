import pandas as pd 
import matplotlib.pyplot as plt 
import ast  

# Function to visualize the signals
def plot_signals(data, index):
    """
    Plots the eye positions and speeds (X and Y) from the given data for a specific index, including the video name, 
    resolution, and label in the plot title.

    Args:
        data (pandas.DataFrame): The input data containing columns with eye positions, eye speeds, video path, resolution, and label.
        index (int): The index of the row in the dataframe to visualize.

    Returns:
        None: The function displays the plots but does not return any value.
    """
    # Extracts data for the current index
    left_position_x = ast.literal_eval(data['left_position X'].iloc[index])  
    left_position_y = ast.literal_eval(data['left_position Y'].iloc[index])  
    right_position_x = ast.literal_eval(data['right_position X'].iloc[index]) 
    right_position_y = ast.literal_eval(data['right_position Y'].iloc[index]) 
    
    left_speed_x = ast.literal_eval(data['left_speed X'].iloc[index]) 
    left_speed_y = ast.literal_eval(data['left_speed Y'].iloc[index]) 
    right_speed_x = ast.literal_eval(data['right_speed X'].iloc[index]) 
    right_speed_y = ast.literal_eval(data['right_speed Y'].iloc[index])

    # Extract additional information
    video_path = data['video'].iloc[index]
    video_name = video_path.split('/')[-1].split('.')[0]  # Extracts the video name without its extension
    resolution = data['resolution'].iloc[index]  # Retrieves the resolution
    label = data['label'].iloc[index]  # Retrieves the label

    # Create subplots
    plt.figure(figsize=(12, 10))

    # Add a title with video information
    plt.suptitle(f'Video: {video_name}, Resolution: {resolution}, Label: {label}', fontsize=14)

    # Plots for eye position X
    plt.subplot(2, 2, 1)
    plt.plot(left_position_x, label='Left Position X', color='blue')  # Plots left eye position X
    plt.plot(right_position_x, label='Right Position X', color='green')  # Plots right eye position X
    plt.title('Eye Position X')
    plt.xlabel('Frame')
    plt.ylabel('Position X')
    plt.legend()

    # Plots for eye position Y
    plt.subplot(2, 2, 2)  # Focuses on the second subplot.
    plt.plot(left_position_y, label='Left Position Y', color='orange')  # Plots left eye position Y
    plt.plot(right_position_y, label='Right Position Y', color='red')  # Plots right eye position Y
    plt.title('Eye Position Y') 
    plt.xlabel('Frame')
    plt.ylabel('Position Y') 
    plt.legend() 

    # Plots for eye speed X
    plt.subplot(2, 2, 3) 
    plt.plot(left_speed_x, label='Left Speed X', color='blue')  # Plots left eye speed X
    plt.plot(right_speed_x, label='Right Speed X', color='green')  # Plots right eye speed X
    plt.title('Eye Speed X') 
    plt.xlabel('Frame') 
    plt.ylabel('Speed X') 
    plt.legend()

    # Plots for eye speed Y
    plt.subplot(2, 2, 4)  # Focuses on the fourth subplot.
    plt.plot(left_speed_y, label='Left Speed Y', color='orange')  # Plots left eye speed Y
    plt.plot(right_speed_y, label='Right Speed Y', color='red')  # Plots right eye speed Y
    plt.title('Eye Speed Y')
    plt.xlabel('Frame') 
    plt.ylabel('Speed Y') 
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusts layout to accommodate the main title.
    plt.show(block=True)  # Displays the plots and waits for the window to be closed before continuing.

# Main function
def main():
    # Read the CSV file
    file_path = "D:/nyst_labelled_videos/merged_data.csv" 
    data = pd.read_csv(file_path)

    index = 0  # Initializes an index variable to iterate through the DataFrame

    # Loop through the DataFrame until all rows are processed
    while index < len(data):  
        plot_signals(data, index)  # Calls the function to plot signals

        # At this point, the plot window is closed, increment the index
        index += 1
        
        # Check if end of data is reached
        if index >= len(data):
            print("End of data.") 
            break

# Start the program
if __name__ == "__main__":
    main()
