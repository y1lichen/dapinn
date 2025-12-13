import os

import ml_collections
import warnings
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
warnings.filterwarnings("ignore", message="juliacall module already imported")

# Import juliacall before torch
from pysr import PySRRegressor

import torch
from examples.qdho.models import Corrector
import multiprocessing

def run_pysr(X, s, corrector_sr_dir):

    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs"],
        population_size=100,
        maxsize=10,
        progress=False,  # Disable progress bars
        output_directory=corrector_sr_dir,
    )

    model.fit(X, s)

    print(model)

def execute_sr(config, workdir):
    # Load corrector and inputs as before
    corrector = Corrector(config).to(config.device)
    corrector_dir = os.path.join(workdir, config.saving.save_dir, config.saving.corrector_path)
    corrector_sr_dir = os.path.join(corrector_dir, "sr")
    
    os.makedirs(corrector_sr_dir, exist_ok=True)

    corrector.load_corrector_model(corrector_dir)

    corrector_inputs_dir = os.path.join(corrector_dir, config.corrector_model_name)
    corrector_checkpoint = torch.load(corrector_inputs_dir, map_location=config.device, weights_only=False)
    if 'corrector_inputs' in corrector_checkpoint:
        corrector_inputs = corrector_checkpoint['corrector_inputs']
        print("Corrector inputs loaded successfully.")
    else:
        corrector_inputs = None
        print("No corrector inputs found in the checkpoint.")

    corrector.eval()
    with torch.no_grad():
        s = corrector(corrector_inputs)

    X = corrector_inputs.detach().cpu().numpy()
    s = s.cpu().numpy().flatten()

    # Run PySR in a separate process
    process = multiprocessing.Process(target=run_pysr, args=(X, s, corrector_sr_dir))
    process.start()
    process.join()