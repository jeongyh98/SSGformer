import cv2
import torch

def get_blur_map_torch_unfold(image_file, win_size=10, sv_num=3, device='cuda'):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    patches = torch.nn.functional.pad(img, (win_size, win_size, win_size, win_size), mode='reflect')
    patches = patches.unfold(2, win_size * 2, 1).unfold(3, win_size * 2, 1)
    
    patches = patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3], win_size * 2, win_size * 2)
    
    s = torch.linalg.svdvals(patches)
    
    top_sv = torch.sum(s[:, :, :, :, :sv_num], dim=-1)
    total_sv = torch.sum(s, dim=-1)
    sv_degree = top_sv / (total_sv + 1e-8)
    
    sv_degree = sv_degree.squeeze()
    
    max_sv = torch.max(sv_degree)
    min_sv = torch.min(sv_degree)

    blur_map = (sv_degree - min_sv) / (max_sv - min_sv+ 1e-8)
    blur_map = blur_map[:img.shape[2], :img.shape[3]]
    
    if torch.any(torch.isnan(blur_map)):
        import pdb; pdb.set_trace()
    if torch.any(torch.isinf(blur_map)):
        import pdb; pdb.set_trace()
    
    blur_map = blur_map.cpu().numpy()
    
    return blur_map