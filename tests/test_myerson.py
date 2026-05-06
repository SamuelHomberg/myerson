import pytest
import numpy as np
import networkx as nx
from myerson import MyersonCalculator, MyersonSampler


def gloves_game_coalition_function(coalition: tuple,
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


class TestMyersonCalculator:

    # test calculate_calculate_all_myerson_values
    def test_gloves_game_case0(self):
        # testing the L--R--R graph
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")

        myerson_calculator = MyersonCalculator(graph=graph,
            coalition_function=gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = np.array([0.5, 0.5, 0.])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"

    def test_gloves_game_case1(self):
        # testing the L--R--R  L graph (disconnected graph)
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")
        graph.add_node(4, glove="left")

        myerson_calculator = MyersonCalculator(graph=graph,
            coalition_function=gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = np.array([0.5, 0.5, 0., 0.])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"

    def test_gloves_game_case2(self):
        # testing the complete LRR graph
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")
        myerson_calculator = MyersonCalculator(graph=graph,
            coalition_function=gloves_game_coalition_function)
        my_values = myerson_calculator.calculate_all_myerson_values()
        solution = np.array([2/3, 1/6, 1/6])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"


class TestMyersonSampler:

    # test calculate_calculate_all_myerson_values
    def test_gloves_game_case0(self):
        # testing the L--R--R graph
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")

        sampler = MyersonSampler(graph=graph,
            coalition_function=gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = np.array([0.5, 0.5, 0.])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"

    def test_gloves_game_case1(self):
        # testing the L--R--R  L graph (disconnected graph)
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")
        graph.add_node(4, glove="left")

        sampler = MyersonSampler(graph=graph,
            coalition_function=gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = np.array([0.5, 0.5, 0., 0.])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"

    def test_gloves_game_case2(self):
        # testing the complete LRR graph
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
        graph.add_node(1, glove="left")
        graph.add_node(2, glove="right")
        graph.add_node(3, glove="right")

        sampler = MyersonSampler(graph=graph,
            coalition_function=gloves_game_coalition_function,
            number_of_samples=1000,
            seed=42,
            disable_tqdm=True)
        my_values = sampler.sample_all_myerson_values()
        solution = np.array([2/3, 1/6, 1/6])
        for my, sol in zip(my_values, solution):
            assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"