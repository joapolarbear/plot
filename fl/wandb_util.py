import wandb
import os

METHOD2WANDB_RUN = {
    "Eff-FL": "sylviana-hkucs/AFL-FederatedEMNIST-10-None-3400/c6d5ba1h",
    "FedCor": "sylviana-hkucs/AFL-FederatedEMNIST-10-None-3400/f5atwme0",
    "Random": "sylviana-hkucs/AFL-FederatedEMNIST-10-None-3400/5yo3imec",
    "Pow-d": "sylviana-hkucs/AFL-FederatedEMNIST-10-None-3400/nrus6jk3"
}

### Input `wandb login` to login first

def wandb_to_csv(method_name = "Eff-FL"):
    api = wandb.Api()
    run = api.run(METHOD2WANDB_RUN[method_name])
    df = run.history()
    df.to_csv(os.path.join(os.path.dirname(__file__), "data", f"{method_name}.csv"))


for method_name in METHOD2WANDB_RUN.keys():
    print(method_name)
    wandb_to_csv(method_name)

