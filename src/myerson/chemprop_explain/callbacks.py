import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from myerson.chemprop_explain.utils import unbatch
from chemprop.data.collate import BatchMolGraph
import pandas as pd
import myerson

class MyersonExplanations(Callback):
    def __init__(self, save_dir: str = "explanations"):
        # TODO: add parameters for explainer/sampler
        # TODO: allow multiclass, chose automatically
        """
        Args:
            save_dir: Directory where the isolated predictions will be saved.
        """
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: any,
        batch: any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if isinstance(batch.bmg, BatchMolGraph):
            molgraphs = unbatch(batch.bmg)
        else:
            raise TypeError(f"Unexpected batch: {batch}")

        batch_idx_column = []
        data_idx_column = []
        explanations_column = []
        explanations_type_column = []

        with torch.no_grad():
            for i, mg in enumerate(molgraphs):
                num_nodes = mg.V.shape[0]
                if num_nodes > 20:
                    sampler = myerson.chemprop_explain.MyersonSamplingExplainer(mg, pl_module)
                    my_values = sampler.sample_all_myerson_values()
                    xai_type = "sampled" # TODO: exchange for __name__ of explainer class
                else: 
                    explainer = myerson.chemprop_explain.MyersonExplainer(mg, pl_module)
                    my_values = explainer.calculate_all_myerson_values()
                    xai_type = "exact"
                my_values = list(my_values.values())

                batch_idx_column.append(batch_idx)
                data_idx_column.append(i)
                explanations_column.append(my_values)
                explanations_type_column.append(xai_type)
        df = pd.DataFrame({
            "batch_idx": batch_idx_column,
            "data_idx": data_idx_column,
            "explanations": explanations_column,
            "explanations_type": explanations_type_column
        })
        save_path = os.path.join(self.save_dir, f"explanations.csv")
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

            