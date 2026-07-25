import nibabel as nib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class fMRILoader:
    """
    A robust data loader for 3D and 4D NIfTI neuroimaging files.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_nifti(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
        
        try:
            # Load the NIfTI image using nibabel
            img = nib.load(filepath)
            logging.info(f"Successfully loaded {filename}")
            logging.info(f"Image shape (X, Y, Z, Time): {img.shape}")
            return img
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            raise
