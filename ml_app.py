from ml.train_model import train
from ml.predict import predict

if __name__ == "__main__":
    print("Training the model...")
    train()
    print("Training complete.\n")

    print("Making a prediction...")
    predict()
