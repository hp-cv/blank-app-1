import os
import zipfile
import streamlit as st
from PIL import Image
import gdown
import subprocess
import sys


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def uninstall_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package])

# Install OpenCV
uninstall_package('opencv-python')
uninstall_package('opencv-contrib-python')
install_package('opencv-python-headless')

from ultralytics import YOLO

def download_and_unzip(url: str, output_dir: str) -> None:
    """
    Downloads a file from Google Drive and extracts its contents.
    
    Args:
        url: The Google Drive URL to download from
        output_dir: Directory to extract contents to
    """
    gdown.download(url, "temp.zip", quiet=False)

    # Save the zip file to the output directory
    zip_path = os.path.join(output_dir, "temp.zip")

    try:
        # Download the zip file
        gdown.download(url, str(zip_path), quiet=False)
        
        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
        print(f"Data downloaded and extracted to {output_dir}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

    # Clean up by removing the zip file
    os.remove(zip_path)
    print(f"Data downloaded and extracted to {output_dir}")



def main():
    st.title("Blueprint Symbol Detection Demo--by HP S")
    
    # Create columns for upload and images
    col1, col2, col3 = st.columns(3)
    
    # Upload image button in first column
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

        
        # Display input image in second column
        with col2:
            st.header("Input Image")
            st.image(input_image, caption='Input Image', use_column_width=True)
            
        # Display output image in third column
        with col3:
            st.header("Processed Image")
            # Add flip button
            if st.button('Detect Symbols'):

                model_fine_tuned = YOLO("yolo11x-seg-best.pt")
                results = model_fine_tuned.predict(source=input_image, conf=0.4, show=True,show_labels=True)
                pred_image = results[0].plot()
                print("finished")

                st.image(pred_image, caption='detected Image', use_column_width=True)
            else:
                st.image(input_image, caption='Original Image', use_column_width=True)

if __name__ == '__main__':
    if not os.path.exists("yolo11x-seg-best.pt"): 
        print("downloading model")
        model_url = f"https://drive.google.com/uc?id=1ht61aEJf65DudtQbT6q8B8IsEh_4Xzqx"
        model_output_dir = "."
        download_and_unzip(model_url, model_output_dir)
        print("downloading model -- Done")

    main()
