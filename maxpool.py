import numpy as np

class MaxPool2:

    def iterate_regions(self,image):
        h,w, = image.shape
        new_h = h//2
        new_w = w//2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2),(j*2):(j * 2 + 2)]
                yield im_region,i,j
    
    def forward(self,input):
        #input is a 3d numpy array with dimensions (h, w, num_filters)

        h,w,num_filters = input.shape
        self.last_input = input
        #output is 13x13x8
        output = np.zeros((h//2,w//2,num_filters))

        for k in range(num_filters):
            for im_region,i,j in self.iterate_regions(input[:,:,k]):
                output[i,j,k] = np.amax(im_region)
        
        return output
    
    def backprop(self,d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        _,_,num_filters = self.last_input.shape

        for k in range(num_filters):
            for im_region,i,j in self.iterate_regions(self.last_input[:,:,k]):
                h,w = im_region.shape
                amax = np.amax(im_region)
                for i2 in range(h):
                    for j2 in range(w):
                        if im_region[i2,j2] == amax:
                            d_L_d_input[i*2+i2,j*2+j2,k] = d_L_d_out[i,j,k]


        return d_L_d_input

