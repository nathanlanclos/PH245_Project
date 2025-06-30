import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import json
from tqdm import tqdm

class StructureFeatureExtractor:
    def __init__(self, pdb_file):
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", pdb_file)

    def num_atoms(self):
        return len([atom for atom in self.structure.get_atoms()])

    def atom_type_distribution(self):
        atom_types = {}
        for atom in self.structure.get_atoms():
            atom_types[atom.element] = atom_types.get(atom.element, 0) + 1
        return atom_types

    def atom_type_proportions(self):
        atom_distribution = self.atom_type_distribution()
        total_atoms = sum(atom_distribution.values())
        return {atom: count / total_atoms for atom, count in atom_distribution.items()}

    def total_mass(self):
        atomic_weights = {
            'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'P': 30.97, 'S': 32.06
        }
        mass = 0
        for atom in self.structure.get_atoms():
            mass += atomic_weights.get(atom.element, 0)
        return mass

    def center_of_mass(self):
        atoms = [atom for atom in self.structure.get_atoms()]
        coords = np.array([atom.coord for atom in atoms])
        return np.mean(coords, axis=0)

    def num_residues(self):
        return len([residue for residue in self.structure.get_residues()])

    def residue_composition(self):
        residues = [residue for residue in self.structure.get_residues()]
        residue_counts = {}
        for residue in residues:
            res_name = residue.get_resname()
            residue_counts[res_name] = residue_counts.get(res_name, 0) + 1
        return residue_counts

    def num_chains(self):
        return len([chain for chain in self.structure.get_chains()])

    def chain_lengths(self):
        chain_lengths = {}
        for chain in self.structure.get_chains():
            chain_lengths[chain.id] = len([residue for residue in chain.get_residues()])
        return chain_lengths

    def radius_of_gyration(self):
        atoms = [atom for atom in self.structure.get_atoms()]
        coords = np.array([atom.coord for atom in atoms])
        center_of_mass = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center_of_mass)**2, axis=1)))
        return rg

    def bounding_box_volume(self):
        atoms = [atom for atom in self.structure.get_atoms()]
        coords = np.array([atom.coord for atom in atoms])
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        dimensions = max_coords - min_coords
        volume = np.prod(dimensions)
        return volume

    def principal_axes_sum(self):
        atoms = [atom for atom in self.structure.get_atoms()]
        coords = np.array([atom.coord for atom in atoms])
        center_of_mass = np.mean(coords, axis=0)
        coords_centered = coords - center_of_mass
        covariance_matrix = np.cov(coords_centered.T)
        eigenvalues, _ = eigh(covariance_matrix)
        return np.sum(eigenvalues[::-1])

    def aspect_ratio(self):
        atoms = [atom for atom in self.structure.get_atoms()]
        coords = np.array([atom.coord for atom in atoms])
        center_of_mass = np.mean(coords, axis=0)
        coords_centered = coords - center_of_mass
        covariance_matrix = np.cov(coords_centered.T)
        eigenvalues, _ = eigh(covariance_matrix)
        return max(eigenvalues[::-1]) / min(eigenvalues[::-1])

    def b_factors(self):
        b_factors = [atom.bfactor for atom in self.structure.get_atoms()]
        return {
            'mean': np.mean(b_factors),
            'variance': np.var(b_factors),
            'min': np.min(b_factors),
            'max': np.max(b_factors)
        }

    def contact_map(self, threshold=5.0):
        residues = [residue for residue in self.structure.get_residues()]
        residue_coords = [residue['CA'].coord for residue in residues if 'CA' in residue]
        distances = squareform(pdist(residue_coords))
        return (distances < threshold).astype(int)

    def num_contacts_per_residue(self, threshold=5.0):
        contact_map = self.contact_map(threshold)
        contacts = np.sum(contact_map, axis=1)
        mean_contacts = np.mean(contacts) if len(contacts) > 0 else 0
        return contacts, mean_contacts

    def solvent_exposed_fraction(self, hydrophobic_residues):
        residues = [residue for residue in self.structure.get_residues()]
        exposed_hydrophobic = 0
        total_hydrophobic = 0
        for residue in residues:
            if residue.get_resname() in hydrophobic_residues:
                total_hydrophobic += 1
                if self.is_exposed(residue):
                    exposed_hydrophobic += 1
        return exposed_hydrophobic / total_hydrophobic if total_hydrophobic > 0 else 0

    def exposed_residue_proportions(self):
        residues = [residue for residue in self.structure.get_residues()]
        exposed_counts = {'polar': 0, 'nonpolar': 0, 'positive': 0, 'negative': 0}
        total_exposed = 0

        # Define residue types
        polar = {'SER', 'THR', 'ASN', 'GLN'}
        nonpolar = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'GLY', 'CYS'}
        positive = {'LYS', 'ARG', 'HIS'}
        negative = {'ASP', 'GLU'}

        for residue in residues:
            if self.is_exposed(residue):
                total_exposed += 1
                res_name = residue.get_resname()
                if res_name in polar:
                    exposed_counts['polar'] += 1
                elif res_name in nonpolar:
                    exposed_counts['nonpolar'] += 1
                elif res_name in positive:
                    exposed_counts['positive'] += 1
                elif res_name in negative:
                    exposed_counts['negative'] += 1

        # Calculate proportions
        if total_exposed > 0:
            return {key: count / total_exposed for key, count in exposed_counts.items()}
        else:
            return {key: 0.0 for key in exposed_counts}

    def is_exposed(self, residue, threshold=8.0):
        center_of_mass = self.center_of_mass()
        for atom in residue:
            if np.linalg.norm(atom.coord - center_of_mass) > threshold:
                return True
        return False

    @staticmethod
    def generate_structure_features(input_data, pdb_dir, output_path=None, hydrophobic_residues=['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']):
        """
        Generate pdb structures and structure-based features for a set of protein sequences.

        Args:
            csv_path (str): Full path to input csv with columns: 'gene' and 'sequence'.
            output_path (str, optional): Optional path to save the output CSV/PKL file.
            model_name (str): Pretrained model to use for folding (default = ESMFold).
            

        Returns:
            DataFrame: The input DataFrame with appended feature columns.
        """
        # Load the input file
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        elif input_data.endswith('.pkl'):
            df = pd.read_pickle(input_data)
        else:
            raise ValueError("Unsupported file format. Only .csv, .pkl, or pandas dataframe are supported.")

        #Create a dataframe with all features above generated from PDB
        # Expand the `~` to the full home directory path if present
        pdb_dir = os.path.expanduser(pdb_dir)

        results = []
        for pdb_file in tqdm(os.listdir(pdb_dir), desc="Extracting features from PDBs"):
            if pdb_file.endswith(".pdb") and pdb_file.split("_")[0][:-4] in df["gene"].values:
                file_path = os.path.join(pdb_dir, pdb_file)
                features = StructureFeatureExtractor(file_path).extract_features(hydrophobic_residues)
                features["gene"] = pdb_file.split("_")[0][:-4]   # Add file name to features
                results.append(features)

        # Convert results to a DataFrame
        structure_df = pd.DataFrame(results)

        #Merges the original dataframe and the dataframe based on the pdb structures
        final_df = pd.merge(
            df,
            structure_df,
            left_on="gene",
            right_on="gene",
            how="inner"
        )

        # Save if requested
        if output_path:
            if output_path.endswith('.csv'):
                final_df['center_of_mass'] = final_df['center_of_mass'].apply(lambda x: json.dumps(x.tolist()))
                final_df.to_csv(output_path, index=False)
            elif output_path.endswith('.pkl'):
                final_df.to_pickle(output_path)
            else:
                raise ValueError("Output must be a .csv or .pkl file.")

        return final_df

    def extract_features(self, hydrophobic_residues):
        atom_proportions = self.atom_type_proportions()
        exposed_proportions = self.exposed_residue_proportions()

        features = {
            "num_atoms": self.num_atoms(),
            "total_mass": self.total_mass(),
            "center_of_mass": self.center_of_mass(),
            "num_residues": self.num_residues(),
            "num_chains": self.num_chains(),
            "radius_of_gyration": self.radius_of_gyration(),
            "bounding_box_volume": self.bounding_box_volume(),
            "principal_axes_sum": self.principal_axes_sum(),
            "aspect_ratio": self.aspect_ratio(),
            "b_factors_mean": self.b_factors()["mean"],
            "b_factors_variance": self.b_factors()["variance"],
            "b_factors_min": self.b_factors()["min"],
            "b_factors_max": self.b_factors()["max"],
            "mean_contacts_per_residue": self.num_contacts_per_residue()[1],
            "solvent_exposed_fraction": self.solvent_exposed_fraction(hydrophobic_residues)
        }

        # Add decomposed atom type proportions
        for atom, proportion in atom_proportions.items():
            features[f"{atom}_atom_type_proportion"] = proportion

        # Add decomposed exposed residue proportions
        for key, proportion in exposed_proportions.items():
            features[f"{key}_exposed_residue_proportion"] = proportion

        return features
    
