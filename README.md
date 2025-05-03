# Semi-automated_MRI_Segmenter
Semi-automated Selective Segmentation Algorithm for any Greyscale Level in an MRI Image

This algorithm was developed for the semi-automated quantification of the synovitis-effusion complex to correlate the severity of inflammation to the progression of Knee Osteoarthritis.

This algorithm was then adapted to be more general for all levels of greyscale.

The primary mechanism of segmentation is a watershed filter on a binary mask of a windowed 3D array. This is particularly effective because it mimics how we as humans utilize the geography of similarly intense regions without a clear border to distinguish the regions from each other.

In order to use the segmenter, 2 variables must be filled in the MRI viewer code. 
image_files should be a list containing the path of each scan
Path should be the desired folder for which the end mask is written, should that be desired by the user

This Viewer is comprised of 2 UIs:
  UI #1: Specify the parameters for the mask
    Use the text and update the slider buttons to adjust the depth of the MRI
    Draw a general/larger ROI on the MRI stack that would propagate through the stack
    Seed & Tolerance specify the window level
    Mask Edge Tolerance specifies the minimum distance between distinctive geographic regions and aids in the size of selective segmentation in the next UI
    
  UI #2: Provide further specificity to the mask
    Utilizing the Image on the left, draw with the mouse to select regions of the mask to either add or delete from the final mask, depending on the switch
    The number displayed is the voxel volume of the mask
    The regions in green are the added regions
    The regions in purple are the original windowed levels that haven't been deleted

    Write Mask --> if the switch is on "add", the region in green is written to Path
    --> if the switch is on "delete", the regions in green and purple are written to Path


    
