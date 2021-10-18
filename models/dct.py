from scipy.fftpack import dct, idct
import numpy as np
import copy
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt


class ProcessorDCT(object):
    def __init__(self, n_keep, gt_mask_len):
        self.n_keep = n_keep
        self.gt_mask_len = gt_mask_len

        inputs = np.zeros((self.gt_mask_len, self.gt_mask_len))
        _, zigzag_table = self.zigzag(inputs)
        self.zigzag_table = zigzag_table[:self.n_keep]

    def sigmoid(self, x):
        """Apply the sigmoid operation.
        """
        y = 1./(1.+1./np.exp(x))
        dy = y*(1-y)
        return y

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        y = -1 * np.log((1-x)/x)
        return y

    def zigzag(self, input, gt=None):
        """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.
        
        Args:
            input (np.array): shape [h,w], value belong to [-127, 128], transformed from gt.
            gt (np.array): shape [h,w], value belong to {0,1}, original instance segmentation gt mask.
        Returns:
            output (np.array): zig-zag encoded values, shape [h*w].
            indicator (np.array): positive sample indicator, shape [h,w].
        """
        # initializing the variables
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = input.shape[0]
        hmax = input.shape[1]
        assert vmax == hmax

        i = 0
        output = np.zeros(( vmax * hmax))
        indicator = []

        while ((v < vmax) and (h < hmax)):
            if ((h + v) % 2) == 0:                 # going up
                if (v == vmin):
                    output[i] = input[v, h]        # if we got to the first line
                    indicator.append(v*vmax+h)
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        
                    i = i + 1
                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    output[i] = input[v, h]
                    indicator.append(v*vmax+h)
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    output[i] = input[v, h]
                    indicator.append(v*vmax+h)
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:                                    # going down
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    output[i] = input[v, h]
                    indicator.append(v*vmax+h)
                    h = h + 1
                    i = i + 1
                elif (h == hmin):                  # if we got to the first column
                    output[i] = input[v, h]
                    indicator.append(v*vmax+h)
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif ((v < vmax -1) and (h > hmin)):     # all other cases
                    output[i] = input[v, h]
                    indicator.append(v*vmax+h)
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                output[i] = input[v, h] 
                indicator.append(v*vmax+h)
                break
        return output, indicator

    def inverse_zigzag(self, input, vmax, hmax):
        """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.
        """
        # initializing the variables
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        output = np.zeros((vmax, hmax))
        i = 0
        while ((v < vmax) and (h < hmax)): 
            if ((h + v) % 2) == 0:                 # going up
                if (v == vmin):
                    output[v, h] = input[i]        # if we got to the first line
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        
                    i = i + 1
                elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                    output[v, h] = input[i] 
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                    output[v, h] = input[i] 
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:                                    # going down
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                    output[v, h] = input[i] 
                    h = h + 1
                    i = i + 1
                elif (h == hmin):                  # if we got to the first column
                    output[v, h] = input[i] 
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif((v < vmax -1) and (h > hmin)):     # all other cases
                    output[v, h] = input[i] 
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
                output[v, h] = input[i] 
                break
        return output
    
if __name__ == "__main__":
    img_path = '/data/dongbin/projects/SparseR-CNN/projects/SparseRCNN/debug/tgt_1.jpg'

    # complete API (wavelet first and then DCT)
    dct_helper = DCTHelper()
    n_keep = 96
    target_size=(128,128)
    mask = cv2.imread(img_path, 0)
    mask = cv2.resize(mask, target_size)
    mask = np.where(mask>=127,255,0)

    # step 1, perform wavelet compression
    # mask = mask.astype(np.float32)
    # coeffs = dct_helper.encode_wavelet(mask, target_size)
    # cA, (cH, cV, cD) = coeffs
    # step 2, perform dct compression on cA
    # cA_dct = dct_helper.encode_dct(cA, gt=None, target_size=cA.shape, n_keep=96)
    # step 3, reconstruct cA from cA_dct
    # cA_idct = dct_helper.decode_idct(cA_dct, cA.shape)
    # step 4, perform wavelet decompression
    # re_mask = dct_helper.decode_iwavelet(cA_idct, target_size)

    # plt.imshow(mask,'gray')
    # plt.show()

    # plt.imshow(re_mask,'gray')
    # plt.show()