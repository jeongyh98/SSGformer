import torch
import torch.nn.functional as F

import numpy as np

# Helper function to convert an image to grayscale
def to_grayscale(image):
    return torch.matmul(image[..., :3],
                        torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).to(image.device))


# Gaussian kernel for blurring
def gaussian_kernel(size=5, sigma=1.0, device='cpu'):
    x = torch.arange(-size//2 + 1, size//2 + 1).to(device)
    y = torch.arange(-size//2 + 1, size//2 + 1).to(device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


# Function to apply convolution
def convolve(image, kernel, device='cpu'):
    B = image.shape[0]
    kernel = kernel.expand(B,-1,-1).unsqueeze(1)
    image = image.unsqueeze(1)
    output = F.conv2d(image, kernel, padding=kernel.size(-1) // 2)
    return output[:,0]


# Sobel kernels
def sobel_kernels(device='cpu'):
    Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(device)
    Ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(device)
    return Kx, Ky


# Non-maximum suppression
def non_maximum_suppression(G, theta, device='cpu'):
    B, M, N = G.shape
    Z = torch.zeros((B, M, N), dtype=torch.float32).to(device)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    # Get angles and gradients for the neighborhoods
    angle_center = angle[:, 1:-1, 1:-1]
    
    angle_0 = ((angle_center >= 0) & (angle_center < 22.5)) | ((angle_center >= 157.5) & (angle_center <= 180))
    angle_45 = (angle_center >= 22.5) & (angle_center < 67.5)
    angle_90 = (angle_center >= 67.5) & (angle_center < 112.5)
    angle_135 = (angle_center >= 112.5) & (angle_center < 157.5)
    
    q_0 = G[:,1:-1, 2:]   # G[i, j+1]
    r_0 = G[:,1:-1, :-2]  # G[i, j-1]
    
    q_45 = G[:,2:, :-2]   # G[i+1, j-1]
    r_45 = G[:,:-2, 2:]   # G[i-1, j+1]
    
    q_90 = G[:,2:, 1:-1]  # G[i+1, j]
    r_90 = G[:,:-2, 1:-1] # G[i-1, j]
    
    q_135 = G[:,:-2, :-2] # G[i-1, j-1]
    r_135 = G[:,2:, 2:]   # G[i+1, j+1]
    
    # Apply non-maximum suppression
    Z[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_0) & (G[:,1:-1, 1:-1] >= r_0) & angle_0] = G[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_0) & (G[:,1:-1, 1:-1] >= r_0) & angle_0]
    Z[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_45) & (G[:,1:-1, 1:-1] >= r_45) & angle_45] = G[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_45) & (G[:,1:-1, 1:-1] >= r_45) & angle_45]
    Z[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_90) & (G[:,1:-1, 1:-1] >= r_90) & angle_90] = G[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_90) & (G[:,1:-1, 1:-1] >= r_90) & angle_90]
    Z[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_135) & (G[:,1:-1, 1:-1] >= r_135) & angle_135] = G[:,1:-1, 1:-1][(G[:,1:-1, 1:-1] >= q_135) & (G[:,1:-1, 1:-1] >= r_135) & angle_135]

    return Z


# Double threshold
def double_threshold(image, lowThreshold, highThreshold):
    B, M, N = image.shape
    
    highThreshold = image.view(B,-1).max(1)[0] * highThreshold
    lowThreshold = highThreshold * lowThreshold

    res = torch.zeros((B, M, N), dtype=torch.float32, device=image.device)

    strong_b, strong_i, strong_j = torch.where(image >= highThreshold.view(-1,1,1))
    weak_b, weak_i, weak_j = torch.where((image <= highThreshold.view(-1,1,1)) & (image >= lowThreshold.view(-1,1,1)))

    res[strong_b, strong_i, strong_j] = 1.0
    res[weak_b, weak_i, weak_j] = 0.5

    return res


# Edge tracking by hysteresis
def edge_tracking_by_hysteresis(image, weak=0.5, strong=1.0):
    B, M, N = image.shape
    weak_mask = image[:, 1:M-1, 1:N-1] == weak

    strong_neighbors = (
        (image[:,2:M, 0:N-2] == strong) |  # bottom-left
        (image[:,2:M, 1:N-1] == strong) |  # bottom-center
        (image[:,2:M, 2:N] == strong) |    # bottom-right
        (image[:,1:M-1, 0:N-2] == strong) | # center-left
        (image[:,1:M-1, 2:N] == strong) |  # center-right
        (image[:,0:M-2, 0:N-2] == strong) | # top-left
        (image[:,0:M-2, 1:N-1] == strong) | # top-center
        (image[:,0:M-2, 2:N] == strong)    # top-right
    )

    # Update the weak pixels that have strong neighbors
    image[:, 1:M-1, 1:N-1][weak_mask & strong_neighbors] = strong

    # Set the rest of the weak pixels to 0
    image[:, 1:M-1, 1:N-1][weak_mask & ~strong_neighbors] = 0.0
    return image


# Canny edge detection function
def canny_edge_detection(image, sigma=1.0, lowThreshold=0.05, highThreshold=0.1):
    device = image.device
    image = image.permute(0,2,3,1)
    
    # Step 1: Grayscale conversion
    if image.ndim == 3:
        gray = to_grayscale(image)
    elif image.ndim == 4:
        gray = to_grayscale(image)
    else:
        gray = image

    # Step 2: Gaussian blur
    kernel = gaussian_kernel(size=5, sigma=sigma, device=device)
    blurred = convolve(gray, kernel, device=device)

    # Step 3: Gradient calculation
    Kx, Ky = sobel_kernels(device=device)
    Gx = convolve(blurred, Kx)
    Gy = convolve(blurred, Ky)
    G = torch.hypot(Gx, Gy)
    theta = torch.atan2(Gy, Gx)

    # Step 4: Non-maximum suppression
    suppressed = non_maximum_suppression(G, theta, device=device)

    # Step 5: Double threshold
    thresholded = double_threshold(suppressed, lowThreshold, highThreshold)

    # Step 6: Edge tracking by hysteresis
    edges = edge_tracking_by_hysteresis(thresholded)

    return edges.unsqueeze(1)

if __name__ == '__main__':
    
    import cv2
    from PIL import Image
    from torchvision.transforms import ToTensor
    
    toTensor = ToTensor()
    degraded_path = '/mnt/ssd4/allweather_dataset/test/rain_drop_test/input/0_rain.png'
    degraded = np.array(Image.open(degraded_path).convert('RGB'))
    degraded = toTensor(degraded)
    degraded = degraded.cuda().unsqueeze(0)
    canny = canny_edge_detection(degraded)
    npcanny = canny.detach().cpu().numpy()
    
    cv2.imwrite('/mnt/ssd4/AWIR/test.png', npcanny)
    import pdb; pdb.set_trace()
    