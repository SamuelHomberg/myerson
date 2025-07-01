import unittest
from myerson import PerturbationExplainer, PerturbationClassExplainer
import torch
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys, create_complete_graph_edges

class TestPerturbationExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.model = GATConvModel(dim_in=5, dim_out=1)
        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device), weights_only=False)
        cls.graph.edge_index = create_complete_graph_edges(cls.graph.x.shape[0])
        cls.solution = {
                        0: -0.11366,
                        1: -0.11366,
                        2: -0.11123,
                        3: -0.14156,
                        4: -0.11366,
                        5: -0.11366,
                        6: -0.11123,
                        7: -0.07453,
                        8: -0.12062,
                        9: -0.07787,
                        }

    def test_complete_explanations(self):
        explainer = PerturbationExplainer(self.graph, self.model, disable_tqdm=False)
        perturbation_values = explainer.calculate_all_perturbation_values()
        for xp, sol in zip(perturbation_values.values(), self.solution.values()):
            self.assertAlmostEqual(xp, sol, msg=f"{perturbation_values=}, {self.solution=}",
                                   places=5)

class TestPerturbationClassExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.model = GATConvModel(dim_in=5, dim_out=3)
        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device), weights_only=False)
        cls.graph.edge_index = create_complete_graph_edges(cls.graph.x.shape[0])
        cls.solution = {
                        0: torch.tensor([[-0.1137,  0.0520, -0.0139]]),
                        1: torch.tensor([[-0.1137,  0.0520, -0.0139]]),
                        2: torch.tensor([[-0.1112,  0.0462, -0.0268]]),
                        3: torch.tensor([[-0.1416,  0.0737, -0.0028]]),
                        4: torch.tensor([[-0.1137,  0.0520, -0.0139]]),
                        5: torch.tensor([[-0.1137,  0.0520, -0.0139]]),
                        6: torch.tensor([[-0.1112,  0.0462, -0.0268]]),
                        7: torch.tensor([[-0.0745,  0.0644, -0.0214]]),
                        8: torch.tensor([[-0.1206,  0.0754, -0.0202]]),
                        9: torch.tensor([[-0.0779,  0.0699, -0.0080]]),
                        }

    def test_complete_explanations(self):
        import logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', force=True)
        explainer = PerturbationClassExplainer(self.graph, self.model, disable_tqdm=False)
        perturbation_values = explainer.calculate_all_perturbation_values()
        for xp, sol in zip(perturbation_values.values(), self.solution.values()):
            self.assertTrue(torch.allclose(xp, sol, atol=0.01), msg=f"{perturbation_values=}, {self.solution=}")