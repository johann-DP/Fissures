import os


def file_exists(filename):
    """Retourne True si le fichier existe."""
    return os.path.exists(filename)


def save_if_not_exists(filename, save_func, *args, **kwargs):
    """
    Si le fichier filename n'existe pas, appelle save_func(*args, **kwargs)
    pour le générer, sinon ne fait rien.
    """
    if not file_exists(filename):
        save_func(*args, **kwargs)
        return True
    return False
