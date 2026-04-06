import unittest
from myerson.pyg_explain import ShapleyExplainer, ShapleySamplingExplainer
from myerson.pyg_explain import ShapleyClassExplainer, ShapleySamplingClassExplainer
from myerson.pyg_explain import MyersonExplainer, MyersonClassExplainer
import torch
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys, create_complete_graph_edges

class TestShapleyExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.model = GATConvModel(dim_in=5, dim_out=1)
        with open("tests/testmodelparams.ckpt", "rb") as f:
            params = torch.load(f, map_location=torch.device(device), weights_only=True)
        state_dict = rename_state_dict_keys(params['state_dict'], device)
        cls.model.load_state_dict(state_dict)
        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device), weights_only=False)
        cls.graph.edge_index = create_complete_graph_edges(cls.graph.x.shape[0])
        myerson_explainer = MyersonExplainer(cls.graph, cls.model)
        cls.solution = myerson_explainer.calculate_all_myerson_values()

    def test_complete_explanations(self):
        explainer = ShapleyExplainer(self.graph, self.model, disable_tqdm=False)
        sh_values = explainer.calculate_all_shapley_values()
        for sh, sol in zip(sh_values.values(), self.solution.values()):
            self.assertAlmostEqual(sh, sol, msg=f"{sh_values=}, {self.solution=}",
                                   places=5)

    def test_sampled_explanations(self):
        sampler = ShapleySamplingExplainer(self.graph, self.model, seed=42, number_of_samples=1000, disable_tqdm=False)
        sh_values = sampler.sample_all_shapley_values()
        for sh, sol in zip(sh_values.values(), self.solution.values()):
            print(f"{sh:.4f} {sol:.4f}")
        for sh, sol in zip(sh_values.values(), self.solution.values()):
            self.assertAlmostEqual(sh, sol, msg=f"{sh_values=}, {self.solution=}",
                                   places=1)

class TestShapleyClassExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # setting multiclass output with random but reproducible weights
        torch.manual_seed(0)
        cls.model = GATConvModel(dim_in=5, dim_out=3)
        last_params = torch.tensor([0.0366, -0.0206, -0.0398])
        assert(torch.allclose(last_params, list(cls.model.parameters())[-1], atol=0.0001)), f"{last_params=}, {list(cls.model.parameters())[-1].detach()=}"

        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device), weights_only=False)
        cls.graph.edge_index = create_complete_graph_edges(cls.graph.x.shape[0])
        myerson_explainer = MyersonClassExplainer(cls.graph, cls.model)
        cls.solution = myerson_explainer.calculate_all_myerson_values()
                       
    def test_complete_class_explanations(self):
        explainer = ShapleyClassExplainer(self.graph, self.model, disable_tqdm=False)
        sh_values = explainer.calculate_all_shapley_values()
        for sh, sol in zip(sh_values.values(), self.solution.values()):
            self.assertTrue(torch.allclose(sh, sol, atol=0.0001), msg=f"{sh_values=}, {self.solution=}")

    def test_sampled_class_explanations(self):
        explainer = ShapleySamplingClassExplainer(self.graph, self.model, disable_tqdm=False)
        sh_values = explainer.sample_all_shapley_values()
        for sh, sol in zip(sh_values.values(), self.solution.values()):
            sh = torch.tensor(sh, dtype=torch.float32)
            self.assertTrue(torch.allclose(sh, sol, atol=0.01), msg=f"{sh_values=}, {self.solution=}")
