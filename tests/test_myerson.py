import unittest
from myerson import MyersonCalculator, MyersonSampler
import networkx as nx

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