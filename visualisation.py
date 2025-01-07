import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import TSNE
from spikingjelly.clock_driven import functional

def visualize_embeddings(model, viz_dataloader, device, n_samples=1000):
    """
    Create t-SNE visualization of the network embeddings using the validation dataset
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for anchor, label in viz_dataloader:
            b, f, c, h, w = anchor.shape
            anchor = anchor.reshape(b, f*c, h, w)
            anchor = anchor.float().to(device)
            
            # Forward pass
            model(anchor)
            batch_embedding = model.sn_final.v.cpu().numpy()
            embeddings.append(batch_embedding)
            functional.reset_net(model)
            
            labels.extend(label.numpy())

            if len(labels) >= n_samples:
                break
    
    # Concatenate all embeddings and convert labels to numpy array
    embeddings = np.concatenate(embeddings, axis=0)[:n_samples]
    labels = np.array(labels[:n_samples])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Digit Class')
    plt.title("t-SNE Visualization of NMNIST Embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig("tsne_embedding.png", dpi=300, bbox_inches='tight')
    plt.close()
    return embeddings_2d, labels


def create_tsne_animation(embeddings_list, labels, n_frames=30, fps=10):
    """
    Create a smooth animation of t-SNE embeddings transitioning between epochs.
    
    Args:
        embeddings_list: List of t-SNE embeddings for each epoch [(n_samples, 2), ...]
        labels: Array of labels for coloring the points (n_samples,)
        n_frames: Number of interpolation frames between each epoch
        fps: Frames per second in the output GIF
    """ 
    def interpolate_points(start, end, n):
        """Linear interpolation between two sets of points"""
        steps = np.linspace(0, 1, n)
        return np.array([(1-step)*start + step*end for step in steps])
        
    # Create figure and initialize scatter plot with first frame
    fig, ax = plt.subplots(figsize=(10, 10))
    first_frame = embeddings_list[-1]  # Start with the last frame
    first_label = labels[-1]
    scatter = ax.scatter(first_frame[:, 0], first_frame[:, 1], 
                        c=first_label, cmap='hsv', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Digit Class')
    title = plt.title("t-SNE Visualization - Frame 0")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    # Set consistent axis limits
    all_embeddings = np.concatenate(embeddings_list)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    margin = 0.1  # Add 10% margin
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Prepare interpolated frames in reverse order
    all_frames = []
    for i in range(len(embeddings_list)-1, 0, -1):  # Reverse the iteration
        interpolated = interpolate_points(
            embeddings_list[i], 
            embeddings_list[i-1], 
            n_frames
        )
        all_frames.extend(interpolated)
    
    # Animation update function
    def update(frame_num):
        scatter.set_offsets(all_frames[frame_num])
        title.set_text(f"t-SNE Visualization - Frame {frame_num}")
        return scatter, title
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(all_frames),
        interval=1000/fps,  # interval in milliseconds
        blit=True
    )
    
    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save('tsne_animation.gif', writer=writer)
    plt.close()