import os
import subprocess

import pkg_resources
from setuptools import find_packages, setup

# Chemin du fichier requirements.txt
requirements_file = "requirements.txt"


def get_installed_packages():
    """Retourne une liste des packages installés dans l'environnement."""
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    return installed_packages


def get_required_packages():
    """Retourne la liste des packages requis à partir du fichier requirements.txt."""
    with open(requirements_file) as f:
        required_packages = {
            line.strip() for line in f if line.strip() and not line.startswith("#")
        }
    return required_packages


def install_missing_packages(missing_packages):
    """Installe les packages manquants."""
    if missing_packages:
        print(f"Installation des packages manquants : {', '.join(missing_packages)}")
        missing_packages = [pkg for pkg in missing_packages if pkg != "itertools"]
        # Remplacer 'sklearn' par 'scikit-learn' lors de l'installation
        missing_packages = [
            pkg.replace("sklearn", "scikit-learn") for pkg in missing_packages
        ]
        subprocess.check_call(
            [os.sys.executable, "-m", "pip", "install", *missing_packages]
        )
    else:
        print("Tous les packages requis sont déjà installés.")


def update_requirements_file(missing_packages):
    """Complète le fichier requirements.txt avec les packages manquants."""
    if missing_packages:
        print(f"Ajout des packages manquants dans {requirements_file}")
        with open(requirements_file, "a") as f:
            for pkg in missing_packages:
                f.write(f"{pkg}\n")


def format_code():
    """Utilise black, pylint et isort pour améliorer l'écriture du code."""
    print("Formatage du code avec black, pylint et isort...")

    # Black
    subprocess.run(["black", "."])

    # Isort
    subprocess.run(["isort", "."])

    # Pylint
    pylint_output = subprocess.run(
        ["pylint", "src", "scripts"], capture_output=True, text=True
    )
    with open(os.path.join("tests", "pylint_report.txt"), "w") as f:
        f.write(pylint_output.stdout)

    print("Formatage terminé. Rapport pylint généré dans tests/pylint_report.txt.")


def main():
    """Point d'entrée du script."""
    # Étape 1 : Comparer les packages installés avec ceux du requirements.txt
    installed_packages = get_installed_packages()
    required_packages = get_required_packages()

    # Identifier les packages manquants
    missing_packages = required_packages - installed_packages

    # Installer les packages manquants
    install_missing_packages(missing_packages)

    # Mettre à jour le fichier requirements.txt si nécessaire
    update_requirements_file(missing_packages)

    # Étape 2 : Formatage du code avec black, isort, et génération de rapport pylint
    format_code()


if __name__ == "__main__":
    main()

# # Configuration de setuptools
# setup(
#     name="fissure_analysis",
#     version="1.0",
#     packages=find_packages(),
#     install_requires=list(get_required_packages()),  # Packages à installer
# )
