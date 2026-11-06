from pathlib import Path


def load_supernova_data(filename):
    
    # Pfad zur Datendatei-
    data_file = Path(filename)

    # Datei einlesen
    with data_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Kommentare entfernen
    data_lines = [
        line for line in lines 
        if line.strip() and not line.startswith("#")
    ]

    # Daten in einzelne Spalten zerlegen
    data = []

    for line in data_lines:
        columns = line.split()
        
        #name = columns[0]
        z = float(columns[1])
        mu = float(columns[2])
        sigma_stat = float(columns[3])
        sigma_syst = float(columns[4])
        
        data.append([z, mu, sigma_stat, sigma_syst])
    
    return data




# Erste fünf Supernovae ausgeben
if __name__ == "__main__":
    data = load_supernova_data("data/Supernovadaten.txt")
    
    print(f"Anzahl der Supernovae: {len(data)}")
    print()

    print("Erste fünf Datensätze:")
    print()

    for row in data[:5]:
        print(row)