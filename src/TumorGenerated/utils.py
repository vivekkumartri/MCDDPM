### Tumor Generateion
import random
import elasticdeform
import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

def apply_brainmask(brainmask, erode=True , iterations=2):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return brainmask
 
 
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=1, kernel_size=3, sigma=1., dim=2):
        super(GaussianSmoothing, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = [sigma] * dim
        self.kernel_size = kernel_size
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
    
        kernel = 1
        meshgrids = torch.meshgrid( [torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')
        #print(kernel_size, sigma, meshgrids)
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input=input.double()
        return self.conv(input, weight=self.weight.double(), groups=self.groups, padding = [i//2 for i in self.kernel_size])
        
def generate_prob_function(mask_shape, device='cuda'):
    sigma = torch.tensor([np.random.uniform(3, 15)], device=device)
    
    # Uniform noise generation
    a = torch.rand(mask_shape, device=device)
    gaussian_filter=GaussianSmoothing(channels=mask_shape[0],sigma=sigma).to(device)
    # Gaussian filter
    a_2 = gaussian_filter(a)
    scale = torch.tensor([np.random.uniform(0.19, 0.21)], device=device)
    base = torch.tensor([np.random.uniform(0.04, 0.06)], device=device)
    a = scale * (a_2 - torch.min(a_2)) / (torch.max(a_2) - torch.min(a_2)) + base
    return a

# first generate 5*200*200*200

def get_texture(mask_shape, device='cuda'):
    # Get the probability function
    a = generate_prob_function(mask_shape, device=device)
    random_sample = torch.rand(mask_shape, device=device)
    
    # If a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).float()

    # Gaussian filter
    if torch.rand(1).item() < 0.7:
        sigma_b = torch.tensor([np.random.uniform(3, 5)], device=device)
    else:
        sigma_b = torch.tensor([np.random.uniform(5, 8)], device=device)
    gaussian_filter=GaussianSmoothing(channels=mask_shape[0],sigma=sigma_b).to(device)
    # Apply Gaussian filter
    b2 = gaussian_filter(b)
    
    # Scaling and clipping
    u_0 = torch.tensor([np.random.uniform(0.5, 0.55)], device=device)
    threshold_mask = b2 > 0.12
    
    mean_b2 = torch.sum(b2 * threshold_mask) / threshold_mask.sum()
    beta = u_0 / mean_b2
    
    Bj = torch.clamp(beta * b2, 0, 1)
    
    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b, device='cuda'):
    # Uniform noise generation
    print(mask_shape)
    a = torch.tensor(np.random.uniform(0, 1, size=mask_shape), device=device)
    gaussian_filter_a=GaussianSmoothing(channels=mask_shape[0],sigma=sigma_a).to(device)
    a_2 = gaussian_filter_a(a)
    scale = torch.tensor([np.random.uniform(0.19, 0.21)], device=device)
    base = torch.tensor([np.random.uniform(0.04, 0.06)], device=device)
    a = scale * (a_2 - torch.min(a_2)) / (torch.max(a_2) - torch.min(a_2)) + base

    # Sample once
    random_sample = torch.tensor(np.random.uniform(0, 1, size=mask_shape), device=device)
    b = (a > random_sample).float()
    gaussian_filter_b=GaussianSmoothing(channels=mask_shape[0],sigma=sigma_b).to(device)
    # Gaussian filter
    b = gaussian_filter_b(b)

    # Scaling and clipping
    u_0 = torch.tensor([np.random.uniform(0.5, 0.55)], device=device)
    threshold_mask = b > 0.12
    mean_b = torch.sum(b * threshold_mask) / threshold_mask.sum()
    beta = u_0 / mean_b
    Bj = torch.clamp(beta * b, 0, 1)

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # Find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan.cpu().numpy(), axis=(0, 1)))[0][[0, -1]]
    
    # Ensure z's position is between 0.3 - 0.7 in the middle of the liver
    z = round(random.uniform(0., 1.) * (z_end - z_start)) + z_start
    liver_mask = mask_scan[..., int(z)]
    # Erode the mask using cv2.erode
    #erode=Erosion2d(mask_scan.shape[0],mask_scan.shape[0])
    #liver_mask = apply_brainmask(liver_mask.cpu(), erode=True , iterations=2)

    liver_mask = torch.tensor(liver_mask, dtype=torch.bool, device=mask_scan.device)

    coordinates = torch.nonzero(liver_mask)
    random_index = torch.randint(0, len(coordinates), (1,)).item()
    xyz = coordinates[random_index].tolist()  # Get x, y
    xyz.append(int(z.item()))
    potential_points = xyz
    return potential_points
# Step 2 : generate the ellipsoid
# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4*x, 4*y, 4*z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = np.floor(com-radii).clip(0,None).astype(int)
    bboxh = (np.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(np.square,np.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    np.copyto(roiaux,dst,where=mask)

    return out
def get_fixed_geo(mask_scan, tumor_type):
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = torch.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=torch.float32).to("cuda")
    tiny_radius, small_radius, medium_radius, large_radius = 8,16,24,32

    if tumor_type == 'tiny':
        num_tumor = torch.randint(3, 11, (1,)).item()
        for _ in range(num_tumor):
            # Tiny tumor
            x = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            y = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            z = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste tiny tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

    elif tumor_type == 'small':
        num_tumor = torch.randint(3, 11, (1,)).item()
        for _ in range(num_tumor):
            # Small tumor
            x = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            y = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            z = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

    elif tumor_type == 'medium':
        num_tumor = torch.randint(2, 6, (1,)).item()
        for _ in range(num_tumor):
            # Medium tumor
            x = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            y = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            z = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            sigma = random.randint(3,6)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

    elif tumor_type == 'large':
        num_tumor = torch.randint(1, 4, (1,)).item()
        for _ in range(num_tumor):
            # Large tumor
            x = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            y = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            z = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            sigma = random.randint(5,10)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste large tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

    elif tumor_type == "mix":
        # Tiny tumors
        num_tiny_tumors = torch.randint(3, 11, (1,)).item()
        for _ in range(num_tiny_tumors):
            x = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            y = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            z = torch.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius), (1,)).item()
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste tiny tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

        # Small tumors
        num_small_tumors = torch.randint(5, 11, (1,)).item()
        for _ in range(num_small_tumors):
            x = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            y = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            z = torch.randint(int(0.75 * small_radius), int(1.25 * small_radius), (1,)).item()
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

        # Medium tumors
        num_medium_tumors = torch.randint(2, 6, (1,)).item()
        for _ in range(num_medium_tumors):
            x = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            y = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            z = torch.randint(int(0.75 * medium_radius), int(1.25 * medium_radius), (1,)).item()
            sigma =random.randint(3,6)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into the test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += torch.tensor(geo).to("cuda")

        # Large tumors
        num_large_tumors = torch.randint(1, 4, (1,)).item()
        for _ in range(num_large_tumors):
            x = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            y = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            z = torch.randint(int(0.75 * large_radius), int(1.25 * large_radius), (1,)).item()
            sigma = random.randint(5,10)

            geo = get_ellipsoid(x, y, z)  # Assuming get_ellipsoid is a PyTorch function
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste large tumor geo into the test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] +=torch.tensor(geo).to("cuda")

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]
    geo_mask = (geo_mask * mask_scan) >= 1

    return geo_mask




def get_tumor(volume_scan, mask_scan, tumor_type, texture):
    geo_mask = get_fixed_geo(mask_scan, tumor_type)
    geo_mask = torch.tensor(geo_mask, dtype=torch.float32)

    return geo_mask*mask_scan

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan.cpu().numpy(), axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan.cpu().numpy(), axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan.cpu().numpy(), axis=(0, 1)))[0][[0, -1]]
    # shrink the boundary
    x_start, x_end = max(0, x_start+1), min(mask_scan.shape[0], x_end-1)
    y_start, y_end = max(0, y_start+1), min(mask_scan.shape[1], y_end-1)
    z_start, z_end = max(0, z_start+1), min(mask_scan.shape[2], z_end-1)
    
    liver_volume = volume_scan[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]
    liver_mask = mask_scan[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]

    x_length, y_length, z_length = x_end - x_start + 1, y_end - y_start + 1, z_end - z_start + 1
    start_x = torch.randint(0, texture.shape[0] - x_length, (1,))
    start_y = torch.randint(0, texture.shape[1] - y_length, (1,))
    start_z = torch.randint(0, mask_scan.shape[2] - z_length, (1,))
    #print(x_start, x_end ,y_start, y_end,z_start, z_end)
    cut_texture = texture[start_x:start_x + x_length, start_y:start_y + y_length, start_z:start_z + z_length]
     
    liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture)
    mask_scan=torch.zeros_like(mask_scan)
    mask_scan[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1] = liver_mask

    return mask_scan
