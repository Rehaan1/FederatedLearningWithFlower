from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl

from collections import OrderedDict
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainLoader, valLoader, num_classes) -> None:
        
        super().__init__()
    
        self.trainLoader = trainLoader
        self.valLoader = valLoader

        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)


    def get_parameters(self, config: Dict[str, Scalar]):
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # The Config here is set by the on_fit_config_fn in the strategy
    def fit(self, parameters, config):

        # copy parameters sent by server to client's local model
        self.set_parameters(parameters)

        # Extracting from the config file. The config file is sent by the strategies
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.trainLoader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainLoader), {}
    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valLoader, self.device)

        return float(loss), len(self.valLoader), {"accuracy": float(accuracy)}


def generate_client_fn(trainLoaders, valLoaders, num_classes):

    def client_fn(cid: str):
        return FlowerClient(trainLoader=trainLoaders[int(cid)],
                            valLoader=valLoaders[int(cid)],
                            num_classes=num_classes,
                            )

    return client_fn


