# TSP-SAM

**Implementation for "*Task-Specified SAM Multi-task Gastric Cancer Diagnosis in Endoscopic Images*"**



### 1. Data Prepare

#### Dataset Annotation Format

The dataset is organized via a text file in which each line corresponds to one image sample. Each line includes the relative image path and three associated clinical labels, separated by spaces:

 [image_path] [pathological_type_label] [differentiation_label] [invasion_depth_label]

**Images and annotations examples**

| Image_Path                                                   | pathological | differentiation | infiltration | segmentation |
| ------------------------------------------------------------ | ------------ | --------------- | ------------ | ------------ |
| <img src="C:\Users\dianzishi03\AppData\Roaming\Typora\typora-user-images\image-20250419162118485.png" alt="image-20250419162118485" style="zoom:33%;" /> |              |                 |              |              |
|                                                              |              |                 |              |              |



 

For example:

images/001.jpg 1 0 1

images/002.jpg 0 1 0

This annotation file is used for both training and validation splits and should be formatted consistently.

 

#### Label Definitions

The multi-task labels are assigned based on pathological reports and expert annotation. The detailed category definitions are as follows:

**Task I. Pathological Type (binary classification):**

- **Label 0**: Benign lesions, including inflammation, intestinal metaplasia, and low-grade intraepithelial neoplasia.
- **Label 1**: Malignant or pre-malignant lesions, including high-grade intraepithelial neoplasia and adenocarcinoma.

 

**Task II. Differentiation Degree (three-class classification)**

-  Label 0: Poorly differentiated types, including poorly and moderately-poorly differentiated adenocarcinoma.
-  Label 1: Well to moderately differentiated types, including moderately, moderately-well, and well-differentiated adenocarcinoma.

-   Label 2: High-grade intraepithelial neoplasia (HGIN), which does not yet show invasive behavior but is clinically treated as an early malignant lesion. It is thus separated into its own category.

 

**Task III. Invasion Depth (three-class classification)**

- Label 0: Intra-mucosal invasion, including the mucosal layer and muscularis mucosa.
-  Label 1: Submucosal invasion.
- Label 2: HGIN, which lacks stromal invasion but is treated with similar caution. It is assigned a separate category.



**Segmentation**

For lesion segmentation tasks, each input image is paired with a corresponding binary segmentation mask. The naming convention and format are standardized as follows:

For an image file named image1.jpg, its corresponding segmentation label should be named image1_mask.jpg.

The segmentation mask is a single-channel binary image, where:

- Pixel value 1 indicates the lesion region.

- Pixel value 0 indicates the background (non-lesion areas).



#### Notes on Data Preparation

- All clinical labels should be verified by experienced endoscopists and pathologists to ensure consistency with diagnosis and clinical treatment strategies.
- The image path should be relative to the dataset root directory and should be consistent with the data loading pipeline used in training.
- l To prevent information leakage during model training and evaluation, care should be taken to split images from the same patient into the same subset (either training or validation), if applicable.****
