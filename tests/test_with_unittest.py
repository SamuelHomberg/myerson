import unittest
from myerson import MyersonCalculator, MyersonExplainer, MyersonSampler, MyersonSamplingExplainer
import networkx as nx
import torch
from .testmodels import GATConvModel
import platform

def rename_state_dict_keys(state_dict, device='cuda:0'):
    """Change name of model parameter  tensors, e.g., `model.fc1.bias` --> 
    `fc1.bias` to use with pure pytorch instead of lightning."""
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k in state_dict:
        new_key = '.'.join(k.split('.')[1:])
        new_dict.update({new_key: state_dict[k]})
    if device == 'cpu':
        newer_dict = OrderedDict()
        for k in new_dict:
            if 'lin_dst' in k:
                pass
            elif 'lin_src' in k:
                newer_dict.update({k.replace('lin_src', 'lin'): new_dict[k]})
            else:
                newer_dict.update({k: new_dict[k]})
        new_dict = newer_dict

    return new_dict

class TestMyersonCalculator(unittest.TestCase):

    def gloves_game_coalition_function(self, coalition: tuple,
                                       nx_graph: nx.classes.graph.Graph) -> float:
        """Coalition function for the gloves game.

        Args:
            coalition (tuple): The coalition for which to calculate the payoff
                of the game.
            nx_graph (nx.classes.graph.Graph): For this implementation of the
                gloves game, we expect a networkX graph which has nodes with a
                `glove` attribute that can be either `right` or `left`.

        Returns:
            float: Worth of the coalition.
        """
        if len(coalition) <= 1:
            return 0.

        gloves = nx.get_node_attributes(nx_graph, 'glove')
        r = sum([1 for k, v in gloves.items() if (v=="right" and k in coalition)])
        l = sum([1 for k, v in gloves.items() if (v=="left" and k in coalition)])
        return float(min(r, l))

    # test calculate_calculate_all_myerson_values
    def test_gloves_game_case0(self):
        # testing the L--R--R graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")

        myerson_calculator = MyersonCalculator(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = {1: 0.5, 2: 0.5, 3: 0.}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}")

    def test_gloves_game_case1(self):
        # testing the L--R--R  L graph (disconnected graph)
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")
        self.graph.add_node(4, glove="left")

        myerson_calculator = MyersonCalculator(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = {1: 0.5, 2: 0.5, 3: 0., 4: 0.}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}")

    def test_gloves_game_case2(self):
        # testing the complete LRR graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")
        myerson_calculator = MyersonCalculator(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = {1: 2/3, 2: 1/6, 3: 1/6}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}")
    
class TestMyersonSampler(unittest.TestCase):

    def gloves_game_coalition_function(self, coalition: tuple,
                                       nx_graph: nx.classes.graph.Graph) -> float:
        """Coalition function for the gloves game.

        Args:
            coalition (tuple): The coalition for which to calculate the payoff
                of the game.
            nx_graph (nx.classes.graph.Graph): For this implementation of the
                gloves game, we expect a networkX graph which has nodes with a
                `glove` attribute that can be either `right` or `left`.

        Returns:
            float: Worth of the coalition.
        """
        if len(coalition) <= 1:
            return 0.

        gloves = nx.get_node_attributes(nx_graph, 'glove')
        r = sum([1 for k, v in gloves.items() if (v=="right" and k in coalition)])
        l = sum([1 for k, v in gloves.items() if (v=="left" and k in coalition)])
        return float(min(r, l))

    # test calculate_calculate_all_myerson_values
    def test_gloves_game_case0(self):
        # testing the L--R--R graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")

        sampler = MyersonSampler(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = {1: 0.5, 2: 0.5, 3: 0.}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}", places=1)

    def test_gloves_game_case1(self):
        # testing the L--R--R  L graph (disconnected graph)
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")
        self.graph.add_node(4, glove="left")

        sampler = MyersonSampler(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = {1: 0.5, 2: 0.5, 3: 0., 4: 0.}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}", places=1)

    def test_gloves_game_case2(self):
        # testing the complete LRR graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
        self.graph.add_node(1, glove="left")
        self.graph.add_node(2, glove="right")
        self.graph.add_node(3, glove="right")

        sampler = MyersonSampler(graph=self.graph,
            coalition_function=self.gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = {1: 2/3, 2: 1/6, 3: 1/6}
        for my, sol in zip(my_values.values(), solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {solution=}", places=1)

class TestMyersonExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.model = GATConvModel(dim_in=5, dim_out=1)
        with open("tests/testmodelparams.ckpt", "rb") as f:
            params = torch.load(f, map_location=torch.device(device))
        state_dict = rename_state_dict_keys(params['state_dict'], device)
        cls.model.load_state_dict(state_dict)
        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device))

        cls.solution = {0: 0.3334993031053314,
                        1: 0.32414250585531335,
                        2: 0.09926704892090386,
                        3: -0.235700370016553,
                        4: 0.32414250134948625,
                        5: 0.3334992930174817,
                        6: 0.2064565007175723,
                        7: 0.14213201623587385,
                        8: -0.12044816162614626,
                        9: 0.11972642920556467}

    # def test_with_fast_restrict(self):
    #     explainer = MyersonExplainer(self.graph, self.model)
    #     if platform.system == 'Linux':
    #         self.assertEqual(explainer.fast_restrict_available, True)
    #     my_values = explainer.calculate_all_myerson_values()
    #     for my, sol in zip(my_values.values(), self.solution.values()):
    #         self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {self.solution=}", places=5)

    def test_with_restrict(self):
        explainer = MyersonExplainer(self.graph, self.model)
        # explainer.set_restrict(use_fast_restrict=False)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values.values(), self.solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {self.solution=}", places=5)

class TestMyersonSamplingExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.model = GATConvModel(dim_in=5, dim_out=1)
        with open("tests/testmodelparams.ckpt", "rb") as f:
            params = torch.load(f, map_location=torch.device(device))
        state_dict = rename_state_dict_keys(params['state_dict'], device)
        cls.model.load_state_dict(state_dict)
        with open("tests/testgraph.pt", "rb") as f:
            cls.graph = torch.load(f, map_location=torch.device(device))

        cls.solution = {0: 0.3334993031053314,
                        1: 0.32414250585531335,
                        2: 0.09926704892090386,
                        3: -0.235700370016553,
                        4: 0.32414250134948625,
                        5: 0.3334992930174817,
                        6: 0.2064565007175723,
                        7: 0.14213201623587385,
                        8: -0.12044816162614626,
                        9: 0.11972642920556467}

    # def test_with_fast_restrict(self):
    #     sampler = MyersonSamplingExplainer(self.graph, self.model, seed=42)
    #     if platform.system == 'Linux':
    #         self.assertEqual(sampler.fast_restrict_available, True)
    #     my_values = sampler.sample_all_myerson_values()
    #     for my, sol in zip(my_values.values(), self.solution.values()):
    #         self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {self.solution=}", places=1)

    def test_with_restrict(self):
        sampler = MyersonSamplingExplainer(self.graph, self.model, seed=42)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values.values(), self.solution.values()):
            self.assertAlmostEqual(my, sol, msg=f"{my_values=}, {self.solution=}", places=1)

if __name__ == '__main__':
    unittest.main()