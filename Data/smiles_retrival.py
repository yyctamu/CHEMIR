import requests
import pandas as pd

def fetch_smiles_from_pubchem(chebi_id):
    # Request IsomericSMILES from PubChem
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/CHEBI:{chebi_id}/property/IsomericSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract IsomericSMILES from the response
        smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
        return smiles if smiles else None
    except Exception as e:
        print(f"Failed to fetch from PubChem: {e}")
        return None

def update_smiles(data):
    for index, row in data.iterrows():
        if pd.isna(row['SMILES']) or row['SMILES'] == '':
            smiles = fetch_smiles_from_pubchem(row['chebi_id'])
            data.at[index, 'SMILES'] = smiles
    return data

# Load dataset
data = pd.read_csv('GNN_SMILES_dataset.csv')

# Update dataset
data = update_smiles(data)

# Save the updated dataset
data.to_csv('GNN_SMILES_dataset_Updated.csv', index=False)
