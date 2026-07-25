from nilearn.maskers import NiftiMasker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_voxel_features(img, mask_img=None):
    """
    Transforms 4D fMRI brain data into a 2D feature matrix for machine learning.
    This step flattens the spatial dimensions (voxels) over time.
    """
    logging.info("Initializing NiftiMasker...")
    
    # Initialize the masker
    # If no mask is provided, it automatically computes a background mask
    masker = NiftiMasker(mask_img=mask_img, standardize=True, memory="nilearn_cache", memory_level=1)
    
    # Fit and transform the data to 2D matrix (time_points x voxels)
    # This is where the magic happens: converting 4D brain data into a ML-ready format
    voxel_time_series = masker.fit_transform(img)
    
    logging.info(f"Extracted features shape: {voxel_time_series.shape} (Time points x Voxels)")
    logging.info("Data is now ready to be fed into scikit-learn models (SVM, Ridge, etc.).")
    
    return voxel_time_series
