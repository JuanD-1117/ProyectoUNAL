import os
import pandas as pd
import kagglehub

# Descarga / localiza el dataset de Kaggle
path = kagglehub.dataset_download("jaimetrickz/galaxy-zoo-2-images")
print("Ruta base del dataset:", path)

# Cargar el mapping de imágenes
mapping_path = os.path.join(path, "gz2_filename_mapping.csv")
mapping = pd.read_csv(mapping_path)

# Cargar el CSV de Hart 2016 (desde URL o local si ya lo tienes)
# Corrected URL
hart_url = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
hart_local_path = "gz2_hart16.csv.gz" # Local path to save the downloaded file

# Download the file if it doesn't exist locally
if not os.path.exists(hart_local_path):
    !wget -O "{hart_local_path}" "{hart_url}"

# Load the dataset from the downloaded location
df_labels = pd.read_csv(hart_local_path, compression="gzip")


# Merge por objid / dr7objid
merged = pd.merge(
    mapping,
    df_labels,
    left_on="objid",
    right_on="dr7objid",
    how="inner"
)