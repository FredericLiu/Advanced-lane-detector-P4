import cv2
import numpy as np
import math
from collections import deque

class Line():
    def __init__(self, mtx, dist):
        '''
        The inintialization of  some internal variables.
        '''        
        self.mtx = mtx
        self.dist = dist
        # if the last 3 frames failed to detect, the reset would be set to True
        self.reset = True  
        # store the last 5 frames with good detection in the queue.
        self.recent_leftfit = deque(maxlen = 5)
        self.recent_rightfit = deque(maxlen = 5)
        #polynomial coefficients averaged over the last 5 iterations
        self.best_leftfit = None  
        self.best_rightfit = None 
        #radius of curvature of the line and car center positon
        self.left_curverad = 0 
        self.right_curverad = 0
        self.center_offset_m = 0
        
    def img_process(self, image):   
        '''
        This function perform the process of each frame, and return the final image, as well as draw the detected region on video
        '''
        #undistort the input frame
        image_undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        # Choose a Sobel kernel size
        ksize = 3 
        
        #caculate the variation of threhold of the undistorted frame
        gradx = self.abs_sobel_thresh(image_undist, orient='x', sobel_kernel=ksize, thresh=(20, 120))
        grady = self.abs_sobel_thresh(image_undist, orient='y', sobel_kernel=ksize, thresh=(20, 120))
        mag_binary = self.mag_thresh(image_undist, sobel_kernel=ksize, mag_thresh=(80, 200))
        dir_binary = self.dir_threshold(image_undist, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))
        color_binary = self.color_threshold(image_undist)
        
        #combine the above thresholded image, and transform it to a top down perspective
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) | ((grady == 1) & (mag_binary == 1) & (dir_binary == 1)))|(color_binary == 1)] = 1
        binary_warped,undist, M, Minv = self.corners_unwarp(combined, self.mtx, self.dist)
               
        #detect the two lanes by sliding window and fitint the polynomials,
        #using two poynomials as the detected lanes.
        #Also perform sanity check and smoothing. for 3 consecutive frame with bad detection, the algrithom will reset and caculate histogram again
        #This algorithm always trust the first detection after reset.
        if(self.reset == True):
            leftx_base, rightx_base = self.FindingBase(binary_warped)
            leftx, lefty, rightx, righty,left_fit, right_fit, out_img = self.firstSliding(binary_warped, leftx_base, rightx_base, margin = 100, minpix = 50, nwindows = 9, window_height= 80)
            self.recent_leftfit.append(left_fit)
            self.recent_rightfit.append(right_fit)            
            self.smoothing()            
        elif(self.reset == False):
            leftx, lefty, rightx, righty,left_fit, right_fit, out_img= self.skipSliding(binary_warped, margin=100, left_fit=self.best_leftfit, right_fit=self.best_rightfit)           
            sanityCheck = self.sanityCheck(left_fit,right_fit)
            if (sanityCheck == True):
                self.recent_leftfit.append(left_fit)
                self.recent_rightfit.append(right_fit)            
                self.smoothing()
            else:
                reset_counter += 1 
                if (reset_counter == 3):
                    self.reset = True 
                    reset_counter = 0
         
        #draw the lanes region and two small window to display some info on the original frame
        final_image = self.drawLane(image_undist, binary_warped, self.best_leftfit, self.best_rightfit, Minv, out_img)
        
        #caculate the curve radius
        self.measureCurv(leftx, lefty, rightx, righty,left_fit, right_fit)
        print(self.left_curverad, 'm', self.right_curverad, 'm')          
        #self.plotTestImage(color_binary, gradx, mag_binary, combined, out_img, image)
        return final_image
    
    def sanityCheck(self, left_fit,right_fit):
        '''
        check whether two detected lanes are roughly parallel
        check whether the horizontal distance between two detected lanes make sense
        '''    
        #check whether two detected lanes are roughly parallel
        leftx_top = left_fit[2]
        leftx_bottom = left_fit[0]*719**2 + left_fit[1]*719 + left_fit[2]
        rightx_top = right_fit[2]
        rightx_bottom = right_fit[0]*719**2 + right_fit[1]*719 + right_fit[2] 
        para_value = np.absolute((leftx_bottom - leftx_top)/(rightx_bottom-rightx_top))
        if((para_value<1.3) & (para_value>0.7)):
            para_check = True
        else:
            para_check = False
        
        #check the horizontal distance between two lanes
        print(rightx_bottom - leftx_bottom)
        if (((rightx_bottom - leftx_bottom)<900) & ((rightx_bottom - leftx_bottom)>700)):
            dist_check = True
        else:
            dist_check = False
        return para_check&dist_check
    
    def smoothing(self):
        '''
        calculate the average of the recent 5 good detection fit coeffects
        '''
        self.best_leftfit = sum(self.recent_leftfit)/len(self.recent_leftfit)
        self.best_rightfit = sum(self.recent_rightfit)/len(self.recent_rightfit)
        
    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        '''
        Calculate directional gradient and apply threhold
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
        if orient == 'y':
            sobel_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))        
        sobel_scaled = np.uint8(255*sobel_abs/np.max(sobel_abs))
        grad_binary = np.zeros_like(sobel_scaled)
        grad_binary[(sobel_scaled>=thresh[0]) & (sobel_scaled<=thresh[1])] = 1 
        return grad_binary
    
    def mag_thresh(self,image, sobel_kernel=3, mag_thresh=(0, 255)):       
        '''
        Calculate gradient magnitude and apply threhold
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize = sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize = sobel_kernel)
        sobelxy = np.sqrt(sobelx**2 + sobely**2)
        sobelxy_scaled = np.uint8(255*sobelxy/np.max(sobelxy))
        mag_binary = np.zeros_like(sobelxy_scaled)
        mag_binary[(sobelxy_scaled >= mag_thresh[0]) & (sobelxy_scaled <= mag_thresh[1])] = 1
        return mag_binary
    
    def dir_threshold(self,image, sobel_kernel=3, thresh=(0, np.pi/2)):
        '''
        Calculate gradient direction and apply threshold
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F,1,0, ksize = sobel_kernel))
        sobely_abs = np.absolute(cv2.Sobel(gray, cv2.CV_64F,0,1, ksize = sobel_kernel))
        direction = np.arctan2(sobely_abs,sobelx_abs)
        dir_binary = np.zeros_like(direction)
        dir_binary[(direction >= thresh[0]) & (direction<=thresh[1])] =1
        return dir_binary  
    
    def color_threshold(self,img):      
        '''
        apply threshold on color
        '''
        img = np.copy(img)
        # Convert to HLS color space, seperate the three channels 
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        #compute the threshold which could give the yellow lane out
        yellow_bin = np.zeros_like(h_channel)
        yellow_bin[((h_channel >= 15) & (h_channel <= 35))
                     & ((l_channel >= 30) & (l_channel <= 204))
                     & ((s_channel >= 115) & (s_channel <= 255))                
                    ] = 1
            
        # compute the threshold which could give the white lane out
        white_bin = np.zeros_like(h_channel)
        white_bin[(l_channel>= 200) & (l_channel <= 255)] = 1            
        # combine the white and yellow thresholded binary image
        s_binary = np.zeros_like(s_channel)
        s_binary[(white_bin==1)|(yellow_bin ==1)] = 1    
        return s_binary
    
    def FindingBase(self,binary_warped):
        '''
        use histogram to find the two starting points of the two lanes
        '''        
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base
    
    def firstSliding(self,binary_warped, leftx_base, rightx_base, margin = 100, minpix = 50, nwindows = 9, window_height = 80):
        '''
        find two lanes using slidding window and polynomials fitting
        '''
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])    
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 10) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 10) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]  
        
        return leftx, lefty, rightx, righty,left_fit, right_fit, out_img
    
    def skipSliding(self,binary_warped, margin, left_fit, right_fit):
        '''
        if the in the last frame the detected lanes passed the sanity check, then don't need to caculate hitogram and sliding window again,
        just use this function to caculate the lanes based on the last frame
        '''
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]   
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)    
        
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        return  leftx, lefty, rightx, righty,left_fit, right_fit, out_img
    
    def measureCurv(self,leftx, lefty, rightx, righty, left_fit, right_fit):
        '''
        Calcualte the radius ofcurvature 
        '''
        # choose the maximum y-value, corresponding to the bottom of the image
        ploty = np.linspace(0, 719, num=720)
        y_eval = np.max(ploty)
        y_eval_left = np.max(lefty)
        y_eval_right = np.max(righty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        
        #compute the position of the car center
        
        img_center  = 1280/2
        center_offset_img_space = (((left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]) + 
                                    (right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2])) / 2) - img_center
        self.center_offset_m = center_offset_img_space * xm_per_pix   
        
    
    def corners_unwarp(self,img, mtx, dist):
        '''
        undistort the original frame, and return the top down perspective of the frame
        '''
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, mtx, dist, None, mtx)
    
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])
    
        # For source points (260,680), (1040,680), (580,460), (700,460)
        src = np.float32([[210,720],[1100,720],[595,450],[690,450]])
        # For destination points, I'm arbitrarily choosing some points to be
        dst = np.float32([[200,720], [1000,720], [200,0], [1000,0]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
        return warped,undist, M, Minv    

    def drawLane(self, image, binary_warped, left_fit, right_fit, Minv, lane_topdown_img):
        '''
        draw detected region on original frame
        Also draw two small window with top-down perspective on original frame, to indicate the internal info of the algrithom
        Also draw the text of curavture and car positon on the fram
        '''
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0) 
        
        #add two small-size video on the result video
        color_warp_small = cv2.resize(color_warp, (256, 144))
        cv2.polylines(lane_topdown_img,np.int_([pts]),False,(227,238,214),8)  
        lane_topdown_img_small = cv2.resize(lane_topdown_img, (256, 144))
        result[20: 164, 10: 266] = color_warp_small
        result[20: 164, 276: 532] = lane_topdown_img_small
        
        #add text of curvature and car center position on frames
        template = "{0:17}{1:17}{2:17}"
        txt_header = template.format("Left Curvature", "Right Curvature", "Center Alignment") 
        txt_values = template.format("{:.4f}m".format(self.left_curverad), 
                                             "{:.4f}m".format(self.right_curverad),
                                             "{:.4f}m Right".format(self.center_offset_m))
        if self.center_offset_m < 0.0:
            txt_values = template.format("{:.4f}m".format(self.left_curverad), 
                                                 "{:.4f}m".format(self.right_curverad),
                                             "{:.4f}m Left".format(math.fabs(self.center_offset_m)))      

        cv2.putText(result, txt_header, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(result, txt_values, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)        
        return result
    
    def plotTestImage(self, color_binary,gradx,mag_binary,combined,color_transformed,orginal_img):
        '''
        Plot the image processing on test images, for tuning the parameters of the algrithom
        '''
        import matplotlib.pyplot as plt
        f, axes= plt.subplots(2, 3, figsize=(24, 9))
        ax1,ax2,ax3,ax4,ax5,ax6 = axes.ravel()
        f.tight_layout()
        ax1.imshow(color_binary, cmap='gray')
        ax1.set_title('threshold color', fontsize=10)
        ax2.imshow(gradx, cmap='gray')
        ax2.set_title('Thresholded gradx', fontsize=10)
        ax3.imshow(mag_binary, cmap='gray')
        ax3.set_title('Thresholded mag_binary', fontsize=10)
        ax4.imshow(combined, cmap='gray')
        ax4.set_title('Thresholded combined', fontsize=10)
        ax5.imshow(color_transformed/255)
        ax5.set_title('color_transformed', fontsize=10)
        ax6.imshow(orginal_img)
        ax6.set_title('original images', fontsize=10)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()          