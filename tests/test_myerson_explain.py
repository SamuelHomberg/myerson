import pytest
import torch
from myerson.pyg_explain import MyersonExplainer, MyersonClassExplainer
from myerson.pyg_explain import MyersonSamplingExplainer, MyersonSamplingClassExplainer
from .testmodels import GATConvModel
from .utils import rename_state_dict_keys


@pytest.fixture(scope="module")
def device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="module")
def regression_setup(device):
    model = GATConvModel(dim_in=5, dim_out=1)
    with open("tests/testmodelparams.ckpt", "rb") as f:
        params = torch.load(f, map_location=torch.device(device), weights_only=True)
    state_dict = rename_state_dict_keys(params['state_dict'], device)
    model.load_state_dict(state_dict)
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)

    solution = {0: 0.3334993031053314,
                1: 0.32414250585531335,
                2: 0.09926704892090386,
                3: -0.235700370016553,
                4: 0.32414250134948625,
                5: 0.3334992930174817,
                6: 0.2064565007175723,
                7: 0.14213201623587385,
                8: -0.12044816162614626,
                9: 0.11972642920556467}
    
    return model, graph, solution


@pytest.fixture(scope="module")
def classification_setup(device):
    torch.manual_seed(0)
    model = GATConvModel(dim_in=5, dim_out=3)
    last_params = torch.tensor([0.0366, -0.0206, -0.0398])
    assert torch.allclose(last_params, list(model.parameters())[-1], atol=0.0001), f"{last_params=}, {list(model.parameters())[-1].detach()=}"
    with open("tests/testgraph.pt", "rb") as f:
        graph = torch.load(f, map_location=torch.device(device), weights_only=False)

    solution = {0: torch.tensor([[-0.1079,  0.0471, -0.0231]]),
                1: torch.tensor([[-0.1079,  0.0464, -0.0229]]),
                2: torch.tensor([[-0.1230,  0.0496, -0.0087]]),
                3: torch.tensor([[-0.1097,  0.0562, -0.0421]]),
                4: torch.tensor([[-0.1079,  0.0464, -0.0229]]),
                5: torch.tensor([[-0.1079,  0.0471, -0.0231]]),
                6: torch.tensor([[-0.1090,  0.0511, -0.0247]]),
                7: torch.tensor([[-0.0823,  0.0611, -0.0204]]),
                8: torch.tensor([[-0.1285,  0.0739, -0.0161]]),
                9: torch.tensor([[-0.0874,  0.0686, -0.0289]])}

    return model, graph, solution


class TestMyersonExplainer:

    # def test_with_fast_restrict(self):
    #     model, graph, solution = regression_setup
    #     explainer = MyersonExplainer(graph, model)
    #     if platform.system == 'Linux':
    #         assert explainer.fast_restrict_available == True
    #     my_values = explainer.calculate_all_myerson_values()
    #     for my, sol in zip(my_values.values(), solution.values()):
    #         assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        explainer = MyersonExplainer(graph, model)
        # explainer.set_restrict(use_fast_restrict=False)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values.values(), solution.values()):
            assert my == pytest.approx(sol, abs=1e-5), f"{my_values=}, {solution=}"


class TestMyersonSamplingExplainer:

    # def test_with_fast_restrict(self):
    #     model, graph, solution = regression_setup
    #     sampler = MyersonSamplingExplainer(graph, model, seed=42)
    #     if platform.system == 'Linux':
    #         assert sampler.fast_restrict_available == True
    #     my_values = sampler.sample_all_myerson_values()
    #     for my, sol in zip(my_values.values(), solution.values()):
    #         assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"

    def test_with_restrict(self, regression_setup):
        model, graph, solution = regression_setup
        sampler = MyersonSamplingExplainer(graph, model, seed=42)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values.values(), solution.values()):
            assert my == pytest.approx(sol, abs=1e-1), f"{my_values=}, {solution=}"


class TestMyersonClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        explainer = MyersonClassExplainer(graph, model)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = explainer.calculate_all_myerson_values()
        for my, sol in zip(my_values.values(), solution.values()):
            assert torch.allclose(my, sol, atol=0.0001), f"{my_values=}, {solution=}"


class TestMyersonSamplingClassExplainer:

    def test_with_restrict(self, classification_setup):
        model, graph, solution = classification_setup
        sampler = MyersonSamplingClassExplainer(graph, model, seed=42)
        # sampler.set_restrict(use_fast_restrict=False)
        my_values = sampler.sample_all_myerson_values()
        for my, sol in zip(my_values.values(), solution.values()):
            my = torch.tensor(my, dtype=torch.float32)
            assert torch.allclose(my, sol, atol=0.01), f"{my_values=}, {solution=}"