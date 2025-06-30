import os
import pandas as pd
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch
from tqdm import tqdm

class StructureGenerator:
    def __init__(self, model_name = "facebook/esmfold_v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True).eval().to(self.device)

        if self.device.type == "cuda":
            self.model.esm = self.model.esm.half()
            torch.backends.cuda.matmul.allow_tf32 = True

        self.model.trunk.set_chunk_size(128)

    def convert_outputs_to_pdb(self, outputs):
        """
        Converts model outputs into PDB-format strings.

        Args:
            outputs (dict): Output dictionary from ESMFold containing:
                - positions
                - atom37_atom_exists
                - aatype
                - residue_index
                - plddt
                - (optional) chain_index

        Returns:
            list of str: PDB-format strings, one per structure in batch.
        """
        final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
        final_atom_positions_np = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs["atom37_atom_exists"].cpu().numpy()

        pdbs = []
        for i in range(outputs["aatype"].shape[0]):
            aa = outputs["aatype"][i].cpu().numpy()
            pred_pos = final_atom_positions_np[i]
            mask = final_atom_mask[i]
            resid = outputs["residue_index"][i].cpu().numpy() + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs["plddt"][i].cpu().numpy(),
                chain_index=outputs["chain_index"][i].cpu().numpy() if "chain_index" in outputs else None,
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def generate_structures_from_csv(self, csv_path, output_dir = None, batch_size = 4):
        """
        Generates 3D structures for all sequences in CSVs in a directory using ESMFold.
        Now processes sequences in batches to optimize performance.

        Args:
            csv_path (str): Path to CSV file with 'sequence' and 'gene' columns.
            model_name (str): Pretrained model to use for folding (default = ESMFold).
            batch_size (int): Number of sequences to fold per batch (adjust based on memory).
        """

        try:
            directory_path = os.path.dirname(csv_path)
            csv_file = os.path.basename(csv_path)
            df = pd.read_csv(csv_path)

            #If no output directory specified, create one in the same directory as csv_path
            if output_dir is None:
                output_dir = os.path.join(directory_path, "structures", os.path.splitext(csv_file)[0])
            os.makedirs(output_dir, exist_ok=True)

            # Filter out sequences that already have PDBs
            df_to_generate = df[~df['gene'].apply(lambda g: os.path.exists(os.path.join(output_dir, f"{g}.pdb")))]
            if df_to_generate.empty:
                print("All PDBs already exist. Skipping generation.")
                return output_dir
            if self.device.type == "cpu":
                batch_size = 1

            batch_indices = range(0, len(df_to_generate), batch_size)
            
            for i in tqdm(batch_indices, desc="Generating 3D structures"):
                batch = df_to_generate.iloc[i:i+batch_size]
                sequences = batch["sequence"].tolist()
                names = batch["gene"].tolist()

                tokenized = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**tokenized)

                pdbs = self.convert_outputs_to_pdb(outputs)

                for name, pdb in zip(names, pdbs):
                    pdb_path = os.path.join(output_dir, f"{name}.pdb")
                    with open(pdb_path, "w") as f:
                        f.write(pdb)

            print(f"Structures saved for file: {csv_file}")
            return output_dir

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None
