import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SNNEncoder
import matplotlib.pyplot as plt
import numpy as np
from spikingjelly.clock_driven import functional
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
from visualisation import visualize_embeddings, create_tsne_animation


# Parameters for experimentations
n_bins = 1

class TripletNMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # Pre-compute indices for each label
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Convert lists to numpy arrays for faster random sampling
        self.label_to_indices = {
            label: np.array(indices) 
            for label, indices in self.label_to_indices.items()
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        anchor, target = self.dataset[index]

        # Get positive sample (same label, different sequence)
        positive_idx = np.random.choice(self.label_to_indices[target])
        while positive_idx == index:  # Ensure we don't get the same sequence
            positive_idx = np.random.choice(self.label_to_indices[target])
        positive, _ = self.dataset[positive_idx]

        # Get negative sample (different label)
        negative_label = np.random.choice(list(self.label_to_indices.keys()))
        while negative_label == target:
            negative_label = np.random.choice(list(self.label_to_indices.keys()))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative, _ = self.dataset[negative_idx]

        anchor = torch.from_numpy(reshape_events(anchor))
        positive = torch.from_numpy(reshape_events(positive))
        negative = torch.from_numpy(reshape_events(negative))

        return anchor.float(), positive.float(), negative.float()

def reshape_events(events):
    return events.reshape(-1, events.shape[2], events.shape[3])  # Shape: (40, 34, 34)

def reshape_data(events):
    return events.reshape(-1, events.shape[1], events.shape[2])

def get_dataset(train=False):
    sensor_size = tonic.datasets.NMNIST.sensor_size
    transform_seq = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_bins)
    dataset = tonic.datasets.NMNIST(save_to="./data", train=True, transform=transform_seq)
    return dataset

def to_dataloader(dataset, batch_size=8, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def training_step(model, criterion, optimizer, anchor, positive, negative):
    model.train()
    optimizer.zero_grad()

    model(anchor)
    anchor_output = model.sn_final.v
    functional.reset_net(model)
    model(positive)
    positive_output = model.sn_final.v
    functional.reset_net(model)
    model(negative)
    negative_output = model.sn_final.v
    functional.reset_net(model)

    training_loss = criterion(anchor_output, positive_output, negative_output)
    training_loss.backward()
    optimizer.step()

    # Calculate accuracy
    pos_dist = F.pairwise_distance(anchor_output, positive_output)
    neg_dist = F.pairwise_distance(anchor_output, negative_output)
    accuracy = (pos_dist < neg_dist).float().mean().item()
    
    return training_loss.item(), accuracy

def validation_step(model, criterion, anchor, positive, negative):
    model.eval()
    with torch.no_grad():
        model(anchor)
        anchor_output = model.sn_final.v
        functional.reset_net(model)
        model(positive)
        positive_output = model.sn_final.v
        functional.reset_net(model)
        model(negative)
        negative_output = model.sn_final.v
        functional.reset_net(model)
        val_loss = criterion(anchor_output, positive_output, negative_output)

        # Calculate accuracy
        pos_dist = F.pairwise_distance(anchor_output, positive_output)
        neg_dist = F.pairwise_distance(anchor_output, negative_output)
        accuracy = (pos_dist < neg_dist).float().mean().item()

    return val_loss.item(), accuracy

def train_network(model, train_dataloader, test_dataloader, viz_dataloader, optimizer, criterion, device, epochs=10):
    model.to(device)
    embedding_frames = []  # Store embeddings for animation
    labels = []

    print(f"Using device: {device}")
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_val_loss = 0
        epoch_val_acc = 0
        train_steps = 0
        val_steps = 0

        # Training loop
        for anchor, positive, negative in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            loss, acc = training_step(model, criterion, optimizer, anchor, positive, negative)
            epoch_train_loss += loss
            epoch_train_acc += acc
            train_steps += 1

            # Vizualisation during training
            if train_steps%1000 == 0:
                embeddings, embedded_labels = visualize_embeddings(model, viz_dataloader, device, n_samples=len(viz_dataloader))
                embedding_frames.append(embeddings)
                labels.append(embedded_labels)  

        # Validation loop
        for anchor, positive, negative in tqdm(test_dataloader, desc=f'Validation Epoch {epoch+1}'):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            loss, acc = validation_step(model, criterion, anchor, positive, negative)
            epoch_val_loss += loss
            epoch_val_acc += acc
            val_steps += 1

        # Calculate epoch metrics
        avg_train_loss = epoch_train_loss / train_steps 
        avg_train_acc = epoch_train_acc / train_steps
        avg_val_loss = epoch_val_loss / val_steps
        avg_val_acc = epoch_val_acc / val_steps

        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

    return model, embedding_frames, labels

def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis, frame in zip(axes, frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = get_dataset(train=True)
    test_dataset = get_dataset(train=False)

    # Dataloader for embedding visualization
    viz_dataloader = to_dataloader(test_dataset, batch_size=4, shuffle=True)

    # Create a subset of the visualization dataset
    subset_size = 8000
    subset_indices = torch.randperm(len(test_dataset))[:subset_size]
    viz_subset = torch.utils.data.Subset(test_dataset, subset_indices)
    viz_dataloader = to_dataloader(viz_subset, batch_size=4, shuffle=True)

    # Convert to triplet dataset
    train_dataset = TripletNMNIST(train_dataset)
    test_dataset = TripletNMNIST(test_dataset)

    # Get dataloaders
    train_dataloader = to_dataloader(train_dataset, batch_size=8)
    test_dataloader = to_dataloader(test_dataset, batch_size=8)

    # Create model
    model = SNNEncoder(input_dim=2 * n_bins, hidden_dim=128, output_dim=10) 

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    model, embedding_frames, labels = train_network(model, train_dataloader, test_dataloader, viz_dataloader, optimizer, criterion, device, epochs=1)

    # Save embeddings and labels
    if not os.path.exists('saved_embeddings'):
        os.makedirs('saved_embeddings')    
    np.save('saved_embeddings/embedding_frames.npy', embedding_frames)
    np.save('saved_embeddings/labels.npy', labels)
    print("Embeddings and labels saved to saved_embeddings/")

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    # Save the model
    torch.save(model.state_dict(), 'saved_model/snn_encoder.pth')
    print("Model saved to saved_model/snn_encoder.pth")

    print("Visualizing embeddings...")
    visualize_embeddings(model, viz_dataloader, device, n_samples=len(viz_dataloader))

    # Create t-SNE animation
    #create_tsne_animation(embedding_frames, labels, n_frames=30, fps=10)


    # model = SNNEncoder(input_dim=2 * 20, hidden_dim=128, output_dim=10) 
    # model.load_state_dict(torch.load('saved_model/snn_encoder.pth'))
    # model.to(device)
    # model.eval()

    # # Add visualization after training
    # print("Creating t-SNE visualization...")
    # visualize_embeddings(model, viz_dataloader, device, n_samples=len(viz_dataloader))
