import joblib
import os

class FileMaker:

    def __init__(self):

        self.base_path = "/configs/"
        os.makedirs(self.base_path, exist_ok=True)

    def create(self, files: dict):
        # .items() permet de récupérer la clé ET l'objet en même temps
        for file_name, obj in files.items():
            # Construction du chemin complet
            full_path = os.path.join(self.base_path, f"ressources.{file_name}.pkl")

            # Sauvegarde de l'objet (obj) et non de la clé
            joblib.dump(obj, full_path)
            print(f"Fichier créé : {full_path}")