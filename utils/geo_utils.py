import numpy as numpy
import torch

def GridGenerator(width, height, homography):
    # num_batch = tf.shape(homography)[0]
    num_batch = homography.shape[0]
    x_t, y_t = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))

    # flatten
    x_t_flat = x_t.view(-1)
    y_t_flat = y_t.view(-1)

 
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat)
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = sampling_grid.unsqueeze(0).repeat(num_batch, 1, 1)

    # cast to float32 (required for matmul)
    homography = homography.to(torch.float32)
    sampling_grid = sampling_grid.to(torch.float32)

    # # transform the sampling grid - batch multiply
    batch_grids = torch.bmm(homography, sampling_grid)
    # ones = torch.zeros([num_batch, 2, width*height])
    out1 = torch.div(batch_grids[:, 0, :], batch_grids[:, 2, :]).view(num_batch, 1, width * height)
    out2 = torch.div(batch_grids[:, 1, :], batch_grids[:, 2, :]).view(num_batch, 1, width * height)
    out = torch.cat([(out1/height-0.5)*2., (out2/height-0.5)*2.], 1).view(num_batch, 2, height, width)

    # # reshape to (num_batch, H, W, 2)
    # out = tf.reshape(out, [num_batch, 2, height, width])

    return out

def Linear_sampler(array, x):# [b, c, h], [b, h]
    # B = tf.shape(array)[0]
    # L = tf.shape(array)[1]
    B, C, L = array.shape

    # grab 4 nearest corner points for each (x_i, y_i)

    x0 = torch.floor(x).to(torch.int32)# [b, H]
    x1 = x0 + 1# [b, H]

    # clip to range [0, H/W] to not violate img boundaries


    x0 = torch.clamp(x0, min=0, max=L - 1)
    x1 = torch.clamp(x1, min=0, max=L - 1)

    # get pixel value at corner coords
    Ia = Get_pixel_value_1D(array, x0)
    Ib = Get_pixel_value_1D(array, x1)



    # calculate deltas
    wa = 1 - (x - x0)
    wb = 1 - (x1 - x)

    # add dimension for addition


    wa = wa.view(B, C, L)
    wb = wb.view(B, C, L)



    # compute output


    out = wa * Ia + wb * Ib

    return out

def Get_pixel_value_1D(array, x0):
    B, C, L = array.shape

    I = torch.zeros_like(array, dtype=torch.float32)

    B_grid = torch.arange(B).view(B, 1).repeat(1, L).view(-1)
  
    I = array[B_grid, :, x0.view(-1)].view(B, C, L)

    # array = 

    return I

def EpipolarTransferHeatmap_siamese_src(prob, H1, a, b, ext_size):
    B, C, H, W = prob.shape
    H += 2*ext_size
    W += 2*ext_size
 
    grid = GridGenerator(H, W, H1) #ranging in [-1,1]
    warped = torch.grid_sample(prob, grid, mode='bilinear')
    # rowmax = tf.reduce_max(warped, axis=2)
    rowmax = torch.max(warped, dim=-1) # [b, c, h]

 

    u = torch.arange(H, dtype=torch.float32)
    u = u.unsqueeze(0)
    u = u.repeat(B, 1)

    a = a[:,0].unsqueeze(1).repeat(1, H)
    b = b[:,0].unsqueeze(1).repeat(1, H)

    u1 = a * u + b # [b, h]
    rowmax = Linear_sampler(rowmax, u1)





    return rowmax

def EpipolarTransferHeatmap_siamese_ref(prob, H1, ext_size):
    B, C, H, W = prob.shape
    H += 2*ext_size
    W += 2*ext_size


    grid = GridGenerator(H, W, H1) #ranging in [-1,1]
    warped = torch.grid_sample(prob, grid, mode='bilinear')

    rowmax = torch.max(warped, dim=-1) # [b, c, h]


    u = torch.arange(H, dtype=torch.float32)
    u = u.unsqueeze(0)
    u = u.repeat(B, 1)

    rowmax = Linear_sampler(rowmax, u)

    return rowmax

