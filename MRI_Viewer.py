# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:55:48 2025

@author: Isaac Ronning
"""

import os
import cv2
import numpy as np
import math
import tkinter as tk
import pydicom
import time
from PIL import Image, ImageTk
from tkinter import filedialog, PhotoImage
from pydicom import dcmread
from pydicom.dataset import FileDataset, Dataset
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, flood
from scipy.ndimage import gaussian_filter, uniform_filter, label
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import morphology


"Part 2: Creating an MRI_mask of the selected scan"
#store a single dicom slice with all of the info to access the voxel dimensions later
global dcm

"Make Image Files a list containing all the DCM scans in the desired plane"
image_files = dicom_collection[last_button_pressed]
dcm = dcmread(image_files[3])

global MRI_Pixels
global MRI_mask
global labeled_mask
global footprint_size
footprint_size = 5
# contain the greyscale values from 0-255 for the MRI
MRI_arrays = [(dcmread(image_files[i5]).pixel_array) for i5 in range(len(image_files))]

# uncomment to use without a gausssian filter
MRI_Stack = np.stack(MRI_arrays, axis = -1).astype(np.uint8)
MRI_Pixels_no_filt = np.round(255 * ((MRI_Stack - MRI_Stack.min())/(MRI_Stack.max() - MRI_Stack.min()))).astype(np.uint8)


# comment to use without a gaussian fillter
sigma = 1 # Blur amount
MRI_Pixels = gaussian_filter(MRI_Pixels_no_filt,sigma =sigma)


class MRIViewer(tk.Tk):
    def __init__(self, image_files, MRI_Pixels):
        
        #Initialize the Viewer
        super().__init__()
        self.title("MRI Slices")
        self.MRI_Pixels = MRI_Pixels       
        self.image_files = image_files
        self.num_images = len(self.image_files)
        if self.num_images == 0:
            raise ValueError("No image files found in the specified folder.")
        
        # Start Building the Widgets
        
        
        # Create a label widget to display the image.
        self.image_label = tk.Label(self)
        self.image_label.grid(row=0, column=0, rowspan = 5)
        
        # Create Text input boxes to modify scan thickness
        # Labels and Entries for Min and Max Values
        tk.Label(self, text="Min Value:").grid(row=0, column=1)
        self.min_entry = tk.Entry(self)
        self.min_entry.grid(row=1, column=1)
        self.min_entry.insert(0, "20")  # Default value
        
        tk.Label(self, text="Max Value:").grid(row=0, column=2)
        self.max_entry = tk.Entry(self)
        self.max_entry.grid(row=1, column=2)
        self.max_entry.insert(0, "130")  # Default value
        
        # create a status label for modified endpoints
        self.status_label = tk.Label(self, text="", fg="black")
        self.status_label.grid(row=2, column=1, columnspan = 2)

        # Update button for modifying scan thickness
        self.update_button = tk.Button(self, text="Update Slider", command=self.update_slider)
        self.update_button.grid(row=3, column=1, columnspan = 2)


        # Create a slider to navigate through slices.
        self.slider = tk.Scale(self, from_=0, to=self.num_images - 1,
                               orient=tk.HORIZONTAL, length=400,label = "Slice Number",
                               command= self.callback_functions)
        self.slider.grid(row=4, column=1, columnspan = 2)
        self.slider.set = 0
        
        
        
        # On top of the image label create a canvas to draw a ROI
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.grid(row=0, column=0, rowspan = 5)
        
        # Variables to hold ROI points and the freehand line
        self.roi_points = []  # Will store the list of x,y coordinates
        self.line = None      # Canvas ID for the free-drawn line
        
        # Bind mouse events to the canvas
        self.canvas.bind("<ButtonPress-1>", self.start_roi)
        self.canvas.bind("<B1-Motion>", self.draw_roi)
        self.canvas.bind("<ButtonRelease-1>", self.end_roi)
        
        self.update_slicenum

        # Create label widget for the seed point image
        self.seed_image = tk.Label(self)
        self.seed_image.grid(row=0, column=3, rowspan = 5)
        
        # create a slider for the seed intensity of the mask
        min_intensity = int(MRI_Pixels.min())
        max_intensity = int(MRI_Pixels.max())
        self.Seed = tk.Scale(self, from_=min_intensity, to=max_intensity, resolution=1,
                                    orient=tk.VERTICAL, label="Seed Value", command=self.callback_functions)
        self.Seed.grid(row=0, column=4)
        self.Seed.set = 50
        
        # Create a slider for the connectivity tolerance of the seed points
        self.Tolerance = tk.Scale(self, from_=0, to=200, resolution= 1,
                                   orient=tk.VERTICAL, label="Tolerance", command=self.callback_functions)
        self.Tolerance.grid(row=0, column=5)

        # create a slider for the additional edge tolerance of the segmented mask
        self.Edge_Tolerance = tk.Scale(self, from_=0, to=50, resolution= 1,
                                   orient=tk.VERTICAL, label="Mask EdgeTolerance", command=self.callback_functions)
        self.Edge_Tolerance.grid(row=3, column=4)
       
        # Create the button to perform the segmentation algorithm from seed points
        self.update_mask = tk.Button(self, text="Update Mask", command=self.create_connected_mask)
        self.update_mask.grid(row=6, column=1)
        
    def callback_functions(self,_):
        self.update_slicenum()
        self.update_image()
        
    def update_slicenum(self):
        """
        update the original image label according to slice number
        """
        index = int(self.slider.get())
        pil_image = Image.fromarray(self.MRI_Pixels[...,index], mode ='L')
        photo = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image= photo)
        self.canvas.create_image(0, 0, image=photo, anchor = tk.NW)
        self.image_label.image = photo
        self.image_label.pixels = self.MRI_Pixels[...,index]
        
    def update_image(self):
        """
        update the seed points image with new seed and tolerance levels
        Please note that if the ROI is not drawn there will be no seed points displayed
        """
        global mask
        Seed_val = int(self.Seed.get())
        Tol_val = int(self.Tolerance.get())
        i4 = int(self.slider.get())
        image_path = self.image_files[i4]
              
        # Get pixels in the seed and tolerance range
        Image_pixels = np.array(self.MRI_Pixels[...,i4])
        low_val = int(Seed_val - Tol_val)
        Image_pixels[ low_val > Image_pixels] = 0
        high_val = int(Seed_val + Tol_val)
        Image_pixels[Image_pixels > high_val] = 0
        # get the pixels in the mask
        Image_pixels[mask < 100] = 0
        
        Segment = ImageTk.PhotoImage(Image.fromarray(Image_pixels))
        # Prevent garbage collection
        self.seed_image.config(image = Segment)
        self.seed_image.label = Segment
        self.seed_image.pixels = Image_pixels

        
    def update_slider(self):
        """
        The purpose of this function is to update the thickness of the scan
        """
        try:
            min_val = float(self.min_entry.get())
            max_val = float(self.max_entry.get())
            if min_val < max_val:
                self.slider.config(from_=min_val, to=max_val)
                #self.slider.set(min_val)
            else:
                self.status_label.config(text="Min should be less than Max", fg="red")
        except ValueError:
            self.status_label.config(text="Please enter valid numbers", fg="red")

        
    def start_roi(self, event):
        """
        Start drawing with the mouse
        """
        # Start the ROI with the initial click coordinates
        self.roi_points = [event.x, event.y]
        # Create a line which will be updated as the mouse moves
        self.line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="red", width=2)
       
    def draw_roi(self, event):
        """
        keep drawing with the mouse
        """
        # Append new points as the mouse is dragged
        self.roi_points.extend([event.x, event.y])
        # Update the freehand line with the new list of coordinates
        self.canvas.coords(self.line, *self.roi_points)
        
           
    def end_roi(self, event):
        """
        upon finishing drawing close the ROI and create a mask for it
        """
        # Remove the freehand line
        self.canvas.delete(self.line)
        # Draw a polygon from the freehand points, closing the ROI shape
        roi_polygon = self.canvas.create_polygon(self.roi_points, outline="red", fill="", width=2)
        #print("ROI points (x, y pairs):", self.roi_points)
        
        poly = np.array((np.array(self.roi_points)), dtype=np.int32)
        poly_pts = np.reshape(poly, (-1, 2))
        global mask
        # Mask is the 2d projection of the ROI
        global image_size
        mask = np.zeros(image_size, dtype=np.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)
        
        
        
    def create_connected_mask(self):
        """
    From the parameterss specified by the sliders in the main window create a 
    mask and assign it and the original MRI to different color channels for display
    """
        global MRI_mask # 3d array of values in the shape of the segmented labels
        global labeled_mask # 3d mask of labels from watershed segmentation
        
        global mask # this is the previously defined 2d ROI
        mri_image = np.array(self.MRI_Pixels)
        Seed_val = int(self.Seed.get())
        Seed_tolerance = int(self.Tolerance.get())
        Connection_tolerance = int(self.Edge_Tolerance.get())

        # Propogate the ROI into 3D 
        ROI = np.zeros(self.MRI_Pixels.shape, dtype=bool)
        for t in range(len(self.image_files)):
          ROI[:,:,t] = mask > 100
          
        
        Binary_Seed_3D_Mask = np.where(ROI, mri_image, 0) == Seed_val
     
        
        
        mri_ROI = np.where(ROI, mri_image, 0)
        mri_ROI_Thresh = np.array(mri_ROI)
        mri_ROI_Thresh[mri_ROI_Thresh > Seed_val+Seed_tolerance] = 0
        mri_ROI_Thresh[mri_ROI_Thresh < Seed_val-Seed_tolerance] = 0

        labeled_mask_all = self.intensity_based_connected_components_3d(self,Binary_Seed_3D_Mask,mri_ROI_Thresh,mri_ROI);
        
        # assign true pixel values to a boolean mask of desired  intensity labels
        
        background_val = labeled_mask_all[1,1,1]
        tf_mask = labeled_mask_all != background_val
        MRI_mask = np.where(tf_mask, mri_image, 0)
        labeled_mask = labeled_mask_all # Uncomment to use with more regions
        #no_low_labels = np.array(labeled_mask_all)
        ## iterate through all the labels
        # for l in range(np.max(labeled_mask_all)):
        #    label_val_sum = sum(MRI_mask_all[labeled_mask_all == l])
         #   label_val_num = len(np.array(np.where(labeled_mask_all == l)).T)
          #  avg_intensity = label_val_sum/label_val_num
            
          #  # if the average of MRI_Pixels of that label <(Seed_val - 0.25*Seed_tolerance) remove it because it is not bright enoough to be the synovitis selected
         #   if avg_intensity < (Seed_val - 0.5* Seed_tolerance) or avg_intensity > (Seed_val + 0.5*Seed_tolerance):
         #       no_low_labels[labeled_mask_all == l] = 0
         #   else:
         #       continue        
            
        # set that label = 0
        #labeled_mask = no_low_labels
        #back_val_2 = no_low_labels[1,1,1]
        #MRI_mask = np.where(no_low_labels!=back_val_2,mri_image,0)
        
        
        
        
        global Final_Mask # Final Mask 
        global composite_volume # Overlayed image
        Final_Mask = np.array(MRI_mask)
        
        
        # create first rendition of the composite volume so it can be called by updateslicenum2
        red_mask = Final_Mask > 1
        blue_knee = (MRI_Pixels - MRI_Pixels.min()) / MRI_Pixels.ptp()
            
        blue_volume = np.zeros((*blue_knee.shape, 3))
        blue_volume[..., 2] = blue_knee

    # Create a red overlay volume: red channel at full intensity
        red_overlay = np.zeros_like(blue_volume)
        red_overlay[..., 0] = 1.0  

        # Set the transparency factor for blending (alpha value)
        alpha = 0.4

        # Create the composite volume:
                # It starts as the blue volume, and for voxels where mask==True,
                #blend the blue value with the red overlay using the transparency factor.
        composite_volume = blue_volume.copy()
        composite_volume[red_mask] = (1 - alpha) * blue_volume[red_mask] + alpha * red_overlay[red_mask]
        
        
        # Once the mask has been created boot up the final window
        self.finalwindow()
        
        
        
    def intensity_based_connected_components_3d(self, ____, Binary_Seed_3D_Mask, mri_ROI_Thresh, mri_ROI):
        """
    Function for enacting flood fill for univisited pixels from seed points
    Creates the mask
    """
        global footprint_size
        # Create a new seed mask where basins have to be at least a distance of x away from each other
        # Get an array of all seeds points
        bool_mask = mri_ROI_Thresh > 1
        
        distance = ndi.distance_transform_edt(bool_mask)
        min_distance = self.Edge_Tolerance.get()
        # Find local peaks in the distance map to use as markers
        #peak_local_max returns coordinates of peaks
        coords = peak_local_max(distance, footprint=np.ones((footprint_size,)*3), labels=bool_mask, min_distance=min_distance)
        # Build an empty marker volume, then label each peak with a unique integer
        markers = np.zeros_like(distance, dtype=int)
        for idx, (z, y, x) in enumerate(coords, start=1):
            markers[z, y, x] = idx

    #Optionally,remove small markers:
        #markers = morphology.remove_small_objects(markers, min_size=)

        #Run watershed, confined to the bool_mask of seed values +- tolerance
        labeled_image = watershed( -distance,markers,mask=bool_mask)
           
        return labeled_image
    

        
    def finalwindow(self):
        """
        Creates a toop level tk window that enables the user to selectively edit the mask

        """
        TopL = tk.Toplevel(self)
        global MRI_mask
        
        # Label to display the masked image
        self.mask_label = tk.Label(TopL)
        self.mask_label.grid(row = 0, column = 0)
        
        # Slider to scroll through the mask
        min_val = float(self.min_entry.get())
        max_val = float(self.max_entry.get())
        self.sliderF = tk.Scale(TopL, from_=min_val, to=max_val,
                               orient=tk.HORIZONTAL, length=400,label = "Slice Number",
                               command= self.update_slicenum2)
        self.sliderF.grid(row = 1, column = 0)
        
        # Canvas for Eraser/adding tool
        self.canvasF = tk.Canvas(TopL, width=256, height=256)
        self.canvasF.grid(row=0, column=0)
        self.selectpoints = []
        self.canvasF.bind("<Button-1>", self.start_select)
        self.canvasF.bind("<B1-Motion>", self.track_move)
        self.canvasF.bind("<ButtonRelease-1>", self.end_select)
        
        # Label to display the volume in mm^3
        self.voxel_num = tk.Label(TopL,text = "")
        self.voxel_num.grid(row = 1, column = 1)
        
        # Swtich for adding/deleting select labels
        global switch_var
        switch_var = tk.BooleanVar(value=False)
        self.switch = tk.Checkbutton(TopL, text="Switch", variable=switch_var, command=self.toggle)
        self.switch.grid(row=2, column=1)

        # Label to display the current state
        self.labelswitch = tk.Label(TopL, text="Delete regions")
        self.labelswitch.grid(row=2, column=2)
        
        # Color overlayed images
        self.overlayimage =  tk.Label(TopL)
        self.overlayimage.grid(row = 0, column = 2)
        
        # Export the Mask
        self.write_mask = tk.Button(TopL, text="Write Mask as Dicom", command=self.Write_Dicom)
        self.write_mask.grid(row=2, column=3, columnspan = 2)
      
    def toggle(self):
        """
        Switch for changing the variable determining if the cursor movement adds a region or deletes one
        """
        global switch_var
        if switch_var.get():
            self.labelswitch.config(text="Add Regions")
        else:
            self.labelswitch.config(text="Delete Regions")
            
            
    def start_select(self, event2):
        """
        Handler for the initial click event. Records the starting position.
        """
        if labeled_mask[event2.y,event2.x,self.sliderF.get()] != 0:
            self.selectpoints.extend([event2.y, event2.x])
            #self.erase_at()

    def track_move(self, event2):
        """
        Handler for the mouse movement with button pressed. Records each movement
        and simulates an erase action by drawing a white shape over the existing content.
        """
        if labeled_mask[event2.y,event2.x,self.sliderF.get()] != 0:
            self.selectpoints.extend([event2.y, event2.x])
            
    def end_select(self,event2):   
        global Final_Mask
        global labeled_mask
        global MRI_mask
        global MRI_Pixels
        global composite_volume
        global switch_var
        
        #global label_pts
        index = self.sliderF.get()
        p_pts = np.array((np.array(self.selectpoints)), dtype=np.int32)
        pts_select = np.reshape(p_pts, (-1, 2))
        
        # Find the labels that were selected by the user drawing on the canvas
        selectedlabels = []
        for i6 in range(len(pts_select)):
            label_pts = (np.c_[pts_select[i6,0],pts_select[i6,1],index])
            selectedlabels.extend([labeled_mask[(label_pts[0,0]), (label_pts[0,1]),(label_pts[0,2]) ]])

        selectedlabels = np.unique(selectedlabels)
        
        # Ensure that the background cannot be selected
        if np.any(selectedlabels==0):

            selectedlabels = np.delete(np.array(selectedlabels), np.where(selectedlabels == 0))
            selectedlabels.tolist()
        
        if switch_var.get():   # if we are adding regions set the pixel value with the associated label = 1  
            Final_Mask[np.isin(labeled_mask,np.array(selectedlabels))] = 1
             
        else: # if we are deleting regions set the pixel value with the associated label = 0
            Final_Mask[np.isin(labeled_mask,np.array(selectedlabels))] = 0
        
        
        red_mask = Final_Mask>1
        green_mask = Final_Mask == 1
        blue_knee = (MRI_Pixels - MRI_Pixels.min()) / MRI_Pixels.ptp()
        
        blue_volume = np.zeros((*blue_knee.shape, 3))
        blue_volume[..., 2] = blue_knee  # Only the blue channel is filled
        
        # Create a Green Overlay volume
        green_overlay = np.zeros_like(blue_volume)
        green_overlay[..., 1] = 1.0
        # Create a red overlay volume: red channel at full intensity
        red_overlay = np.zeros_like(blue_volume)
        red_overlay[..., 0] = 1.0

        # Set the transparency factor for blending
        alpha_red = 0.4
        alpha_green = 0.5

        # Create the composite volume:
        # It starts as the blue volume, and for voxels where mask==True,
        # blend the blue value with the red overlay using the transparency factor.
        composite_volume = blue_volume.copy()
        
        if switch_var.get():
            # if added volume is important then display red and green
            composite_volume[red_mask] = (1 - alpha_red) * blue_volume[red_mask] + alpha_red * red_overlay[red_mask]
            composite_volume[green_mask] = (1 - alpha_green) * composite_volume[green_mask] + alpha_green * green_overlay[green_mask]

        else:
            composite_volume[red_mask] = (1 - alpha_red) * blue_volume[red_mask] + alpha_red * red_overlay[red_mask]
        
                
        return Final_Mask
        
    
    def update_slicenum2(self,___):
        
        global MRI_mask
        global Final_Mask
        global dcm # contains the voxel dimensions
        global composite_volume
        global switchvar
        global synoscore
        # Get the contraints of the knee set by the user on the slider
        min_val = int(self.min_entry.get())
        max_val = int(self.max_entry.get())
        
        # display the synovitis score
        if switch_var.get(): # If we only want the regions we added
            num_vox = sum(sum(sum(Final_Mask[...,min_val:max_val] ==1)))
            volume_vox = num_vox*dcm.SliceThickness * np.prod(dcm.PixelSpacing)
        else: # If we want all regions that were not deleted
            num_vox = sum(sum(sum(Final_Mask[...,min_val:max_val] >=1)))
            volume_vox = num_vox * dcm.SliceThickness * np.prod(dcm.PixelSpacing)
            
        synoscore = str(volume_vox)
        # Update the displayed image
        index = int(self.sliderF.get())
        pil_image = Image.fromarray(MRI_mask[...,index], mode ='L')
        photo = ImageTk.PhotoImage(pil_image)
        self.voxel_num.config(text = str(volume_vox))
        self.mask_label.config(image= photo)
        self.mask_label.image = photo
        self.canvasF.create_image(0, 0, image=photo, anchor = tk.NW)
        self.mask_label.pixels = MRI_mask[...,index]
        
        
        # Display the overlay image
        overlayimageraw = (composite_volume[:, :, index, :] * 255).astype(np.uint8)
        O_image = ImageTk.PhotoImage(Image.fromarray(overlayimageraw))
        self.overlayimage.config(image = O_image )
        self.overlayimage.image = O_image
        
        
    def Write_Dicom(self):
        global Path
        global Final_Mask
        global MRI_Pixels
        global synoscore
        mask_folder = os.path.join(Path,"mask_folder")
        os.makedirs(mask_folder, exist_ok=True)
        # Determine which mask to write to the file according to switchvar
        if switch_var.get():
            array_3d = np.array(np.where(Final_Mask == 1,MRI_Pixels,0)).astype(np.int16)
        else:
            array_3d = np.array(np.where(Final_Mask >= 1, MRI_Pixels,0)).astype(np.int16)
        # Write the mask
        for i in range(array_3d.shape[2]):
            file_path = os.path.join(mask_folder, f"mask_slice_{i+1}.dcm")
            self.create_dicom(self,array_3d[:, :, i], file_path, i+1)
            
        text_path = os.path.join(mask_folder, "Volumetric_Score.txt")
        text = f"The volume of the highlighted area is {synoscore}"
        with open(text_path, "w") as file:
            file.write(text)
            
            
    def create_dicom(self,___,image_slice, file_path, instance_number):
        "function to write the mask to a dicom file"
        global dcm
        height, width = image_slice.shape
        n_bits = image_slice.dtype.itemsize * 8
        ds = pydicom.Dataset(dcm)
        
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_name = "mask"
        ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)


        ds.Rows = height
        ds.Columns = width
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = n_bits
        ds.BitsStored    = n_bits
        ds.HighBit       = n_bits - 1
        ds.PixelRepresentation = 0  # unsigned
        ds.PixelSpacing = [1.0, 1.0]
        ds.ImageOrientationPatient = [1,0,0, 0,1,0]
        ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        ds.InstanceNumber = 1
        
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        
        ds.PixelData = image_slice.tobytes()

        pydicom.dcmwrite(file_path, ds)
            
if __name__ == "__main__":
    # Replace 'mri_images' with the path to your folder containing MRI slice images.
    
    
    # Launch the MRI Segmenting viewer
    app = MRIViewer(image_files, MRI_Pixels)
    app.minsize(1000, 400)
    app.mainloop()
    
    
    
file_path = "my_text_file.txt"
text_content = "This is some text to write to the file.\nThis is another line of text."

with open(file_path, "w") as file:
    file.write(text_content)