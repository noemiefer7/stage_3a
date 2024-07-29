import zipfile
import os
import shutil


folder = r'D:\donnees_S1_SAR\sar_zip'
filenames = os.listdir(folder)

for filename in filenames:

    zip_file = folder + '/'+filename
    extract_to = r'D:\donnees_S1_SAR\sar'    
    
    # Ouvrir le fichier zip en mode lecture
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extraire tous les fichiers et répertoires dans le répertoire de destination
        zip_ref.extractall(extract_to)
        print("Le fichier a été décompressé avec succès dans :", extract_to)


shutil.rmtree(folder)
    