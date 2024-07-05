import dicom2nifti
# import SimpleITK as sitk
# import pydicom
# import os
# import numpy as np
# def safe_sitk_dcm_read(dcm_dir, *args, **kwargs):
#     dcm_list = [os.path.join(dcm_dir, i) for i in os.listdir(dcm_dir)]
#     indices = np.array([pydicom.dcmread(i).InstanceNumber for i in dcm_list])
#     dcm_list = np.array(dcm_list)[indices.argsort()[::-1]]
#     return sitk.ReadImage(dcm_list, *args, **kwargs)

# def dcm2nii(dcms_path, nii_path, replace=False):
#     # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
#     reader.SetFileNames(dicom_names)
#     image2 = reader.Execute()
#     # 2.将整合后的数据转为array，并获取dicom文件基本信息
#     image_array = sitk.GetArrayFromImage(image2)  # z, y, x
#     if replace:
#         if list(np.unique(image_array)) != [0, 1]:
#             # logger.warning('{} label array value as unique: {} max: {}, min:{}, mean: {}'.format(nii_path,np.unique(image_array),image_array.max(),image_array.min(),image_array.mean()))
#             image_array = image_array+1024
#         assert list(np.unique(image_array)) == [0, 1]
#     image_array = image_array.astype(np.uint8)
#     size = image2.GetSize()
#     origin = image2.GetOrigin()  # x, y, z
#     spacing = image2.GetSpacing()  # x, y, z
#     direction = image2.GetDirection()  # x, y, z
#     # 3.将array转为img，并保存为.nii.gz
#     image3 = sitk.GetImageFromArray(image_array)
#     image3.SetSpacing(spacing)
#     image3.SetDirection(direction)
#     image3.SetOrigin(origin)
#     sitk.WriteImage(image3, nii_path)
#     return size, origin, spacing, direction,image_array

# import os
# import pydicom
# import numpy as np
# import nibabel as nib

# def dicom2nifti(input_dicom_dir, output_nifti_path):
#     """
#     Convert DICOM files in a directory to NIfTI (.nii.gz) format.

#     Parameters:
#         input_dicom_dir (str): Path to the directory containing DICOM files.
#         output_nifti_path (str): Path to save the output NIfTI file.
#     """
#     # Get list of DICOM files in the input directory
#     dicom_files = [os.path.join(input_dicom_dir, f) for f in os.listdir(input_dicom_dir) if f.endswith(".dcm")]
#     dicom_files.sort()  # Ensure files are in the correct order
    
#     # Read the first file to get the metadata and size
#     ref_dicom = pydicom.dcmread(dicom_files[0])
    
#     # Get image dimensions from the DICOM metadata
#     pixel_dims = (int(ref_dicom.Rows), int(ref_dicom.Columns), len(dicom_files))
#     pixel_spacing = (float(ref_dicom.PixelSpacing[0]), float(ref_dicom.PixelSpacing[1]), float(ref_dicom.SliceThickness))
    
#     # Create an empty NumPy array and populate it with the DICOM pixel data
#     image_data = np.zeros(pixel_dims, dtype=np.int16)
#     for i, file in enumerate(dicom_files):
#         dicom_file = pydicom.dcmread(file)
#         image_data[:, :, i] = dicom_file.pixel_array
        
#     # Generate affine matrix using pixel spacing and orientation info
#     affine = np.eye(4)
#     affine[:3, :3] = np.diag(pixel_spacing)
    
#     # Create NIfTI image
#     nifti_image = nib.Nifti1Image(image_data, affine)
    
#     # Save NIfTI image
#     nib.save(nifti_image, output_nifti_path)
dicom_path = "/ai/data/data/kejibu/tiantan3/天坛医院ANGEL-act影像资料/001-0003-SYXI/04 2017111924小时CTA/20171119000377/6_1E8C61F6C49A40D7A2FE5BCE35A01C11"
nii_path = "/ai/data/data/CTA_celebral_vessel/test_data_results/1_0000.nii.gz"
# size, origin, spacing, direction,image_array = dcm2nii(dicom_path,nii_path)
# Usage example
dicom2nifti.convert_directory(dicom_path, nii_path)
