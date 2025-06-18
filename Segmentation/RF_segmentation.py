import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import matplotlib
from PIL import Image
import rasterio

matplotlib.use('QT5Agg')
#
# imo = rasterio.open(r"E:\Doctoraat\Drone\2021\Ruffus\14062021\20210614_Ruffus_MS_NaDir_Clipped.tif")
# im = imo.read(3)
# imo.close()
# msk = Image.open((r"E:\Doctoraat\Drone\2021\Ruffus\14062021\ruffus_annotation.tiff"))
# mask = np.array(msk)
# msk.close()
# img = im
# training_labels = mask

# Build an array of labels for training the segmentation.
# Here we use rectangles but visualization libraries such as plotly
# (and napari?) can be used to draw a mask on the image.
# training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
# training_labels[:130] = 1
# training_labels[:170, :400] = 1
# training_labels[600:900, 200:650] = 2
# training_labels[330:430, 210:320] = 3
# training_labels[260:340, 60:170] = 4
# training_labels[150:200, 720:860] = 4
#
# sigma_min = 1
# sigma_max = 16
# features_func = partial(feature.multiscale_basic_features,
#                         intensity=True, edges=False, texture=True,
#                         sigma_min=sigma_min, sigma_max=sigma_max)
# features = features_func(img)
# clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
#                              max_depth=10, max_samples=0.05)
# clf = future.fit_segmenter(training_labels, features, clf)
# result = future.predict_segmenter(features, clf)
#
# fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 4))
# ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
# ax[0].contour(training_labels)
# ax[0].set_title('Image, mask and segmentation boundaries')
# ax[1].imshow(result)
# ax[1].set_title('Segmentation')
# ax[2].imshow(img)
# fig.tight_layout()
#
#
# fig, ax = plt.subplots(1, 2, figsize=(9, 4))
# l = len(clf.feature_importances_)
# feature_importance = (
#         clf.feature_importances_[:l//3],
#         clf.feature_importances_[l//10:2*l//3],
#         clf.feature_importances_[2*l//3:])
# sigmas = np.logspace(
#         np.log2(sigma_min), np.log2(sigma_max),
#         num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
#         base=2, endpoint=True)
# for ch, color in zip(range(3), ['r', 'g', 'b']):
#     ax[0].plot(sigmas, feature_importance[ch][::3], 'o', color=color)
#     ax[0].set_title("Intensity features")
#     ax[0].set_xlabel("$\\sigma$")
# for ch, color in zip(range(3), ['r', 'g', 'b']):
#     ax[1].plot(sigmas, feature_importance[ch][1::3], 'o', color=color)
#     ax[1].plot(sigmas, feature_importance[ch][2::3], 's', color=color)
#     ax[1].set_title("Texture features")
#     ax[1].set_xlabel("$\\sigma$")
#
# fig.tight_layout()



# 3D


def segment_rows(ortho_path, annotated_raster_path, save_path=None, show=False):
    """
    Train an RF classification model to segment fore- and background pixels from a .tiff orthophoto
    :param ortho_path: Path to (clipped) orthophoto by micasense camera
    :type ortho_path: str
    :param annotated_raster_path: Path to annotated raster path (created by shp_to_mask.py)
    :type annotated_raster_path: str
    :param save_path: Path to save the segmented image, if None, nothing is saved
    :type save_path: str
    :param show: Plots image if True
    :type bool
    :return: segmented mask (0 = background, 1= foreground, > 1 means more than two classes were annotated)
    :rtype: np.Array
    Example:
    img = segment_rows(r"E:\Doctoraat\Drone\2021\Ruffus\14062021\20210614_Ruffus_MS_NaDir_Clipped.tif",
                       r"E:\Doctoraat\Drone\2021\Ruffus\14062021\ruffus_annotation.tiff",
                       save_path=None)
    returns a binary image
    """
    imo = rasterio.open(ortho_path)
    r, g, b = [imo.read(k) for k in (6, 4,2 )]
    imo.close()
    msk = Image.open(annotated_raster_path)
    mask = np.array(msk)
    msk.close()
    training_labels = mask
    img = np.dstack([r, g, b])
    img = (img / 30000 * 255).astype(np.uint8) # MUST BE UINT8

    sigma_min = 1
    sigma_max = 14
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            channel_axis=-1)
    features = features_func(img)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                 max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels.astype(np.uint8), features, clf)
    result = future.predict_segmenter(features, clf)

    if show:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
        ax[0].imshow(segmentation.mark_boundaries(img, result, mode='thick'))
        ax[0].contour(training_labels)
        ax[0].set_title('Image, mask and segmentation boundaries')
        ax[1].imshow(result)
        ax[1].set_title('Segmentation')
        fig.tight_layout()

    if save_path is not None:
        kwargs = imo.meta.copy()
        kwargs.update({'count': 1})
        with rasterio.open('G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_image_segmentation/segs.tif' , 'w', **kwargs) as dst:
            dst.write(np.array([result]))
    else:
        return result
