from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback  # Import corretto
from roboflow import Roboflow
import wandb
import torch
import yaml
import shutil

# Funzione di addestramento che verrà chiamata per ogni combinazione di parametri nel grid search
def train():
    with wandb.init() as run:
        config = wandb.config  # Accede ai parametri di configurazione gestiti da W&B

        # Stampa per verificare i valori
        print("W&B Config:", config)

        # Imposta il dispositivo di training
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        # 2. Inizializza il modello YOLO
        model = YOLO("yolov8s.pt")  # Assicurati che il peso corretto sia presente

        # 3. Aggiungi il callback di W&B per loggare i risultati durante il training
        add_wandb_callback(model)  # Passa il modello direttamente a add_wandb_callback()

    
        # 5. Avvia l'addestramento del modello YOLO con i parametri dal config
        results = model.train(
            data=data_yaml,
            epochs=config.epochs,  # Corretto accesso a 'epochs'
            device=device,
            project='Nystagmus_detection',
            batch=config.batch_size,  # Corretto accesso al batch_size
            lr0=config.lr,  # Corretto accesso al learning rate
            optimizer=config.optimizer,  # Corretto accesso all'optimizer
            save=True  # Salva il modello
        )


        # Ottieni il percorso del miglior modello salvato
        best_model_path = f'{results.save_dir}/weights/best.pt'  # Percorso del miglior modello YOLOv8

        # Usa l'ID del run per generare un nome univoco per il modello
        unique_model_name = f"/repo/porri/yolo_models/best_model_{run.id}.pt"

        # Copia il miglior modello salvato nella nuova destinazione
        shutil.copy(best_model_path, unique_model_name)

        print(f"Best YOLO model saved at: {unique_model_name}")


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

    # Recupera gli sweep esistenti dal progetto
    sweep_id = None
    project_name = "yolo-eyes"  # Assicurati che il nome del progetto sia corretto

    # Ottieni gli sweep esistenti
    sweeps = wandb.sweeps(project=project_name)

    if sweeps:
        # Se ci sono sweep esistenti, prendi il primo
        sweep_id = sweeps[0]['id']
        print(f"Using existing sweep ID: {sweep_id}")
    else:
        # Se non ci sono sweep esistenti, creane uno nuovo
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        print(f"Created new sweep ID: {sweep_id}")

    # Avvia lo sweep
    wandb.agent(sweep_id, function=train)  # count specifica il numero massimo di esecuzioni dello sweep
