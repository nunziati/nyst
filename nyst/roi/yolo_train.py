from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback as add_wandb_callbacks
from roboflow import Roboflow
import wandb
import torch
import yaml

# Funzione di addestramento che verrà chiamata per ogni combinazione di parametri nel grid search
def train():
    with wandb.init() as run:
        config = wandb.config  # Accede ai parametri di configurazione gestiti da W&B

        # Stampa per verificare i valori
        print("W&B Config:", config)

        # Imposta il dispositivo di training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 2. Inizializza il modello YOLO
        model = YOLO("yolov8m.pt")  # Assicurati che il peso corretto sia presente

        # 5. Avvia l'addestramento del modello YOLO con i parametri dal config
        results = model.train(
            data=data_yaml,
            epochs=config.epochs,  # Corretto accesso a 'epochs'
            device=device,
            project='Nystagmus detection',
            batch=config.batch_size,  # Corretto accesso al batch_size
            lr0=config.lr,  # Corretto accesso al learning rate
            optimizer=config.optimizer,  # Corretto accesso all'optimizer
        )

        # 6. Segnala la fine del run
        wandb.finish()

if __name__ == '__main__':

    # 1. Inizializza il client Roboflow
    rf = Roboflow(api_key="2PgwSNm3Z6uQgPaynNMR")  # Sostituisci con la tua chiave API

    # Usa l'ID del progetto per accedere direttamente al progetto
    try:
        project = rf.project("andreap/eyes3.0-ascvn")
        dataset = project.version(2).download("yolov8")
        data_yaml = dataset.location + "/data.yaml"
    except RuntimeError as e:
        print("Error downloading dataset:", e)
        exit(1)

    # Logs into the Weights and Biases (W&B) platform, ensuring the user is authenticated
    wandb.login()

    # Carica la configurazione dello sweep da un file YAML
    with open('config_wb.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Crea uno sweep in W&B utilizzando la configurazione caricata
    sweep_id = wandb.sweep(sweep_config, project="yolo-eyes")

    # Avvia lo sweep
    wandb.agent(sweep_id, function=train)  # count specifica il numero massimo di esecuzioni dello sweep
