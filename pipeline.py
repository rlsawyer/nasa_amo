# Requires (Anaconda) Python 2.7.12 for Data Augmentation "multiprocessing" package

import glob
import os
import shlex
import subprocess
import time
from os import getcwd, mkdir, listdir
from os.path import join, isdir, expanduser
from shutil import copyfile, rmtree

import cv2
import imageio
import numpy as np
import skvideo.io
import xlrd
from PIL import Image
from skimage import morphology, exposure, measure, filters, feature


class InformationRetreival(object):
    def __init__(self):

        self.home = os.path.join(expanduser('~'), 'Documents/Develop')
        self.prototype = os.path.join(self.home, 'prototype')

    @staticmethod
    def extract_frames(sheet, video_dir, output_frames):

        num_neg, num_pos = 0, 0

        for i in range(413, 846):  # range(413, 846)
            # Value of ith row and 2nd column
            fname = "IMG" + str(int(sheet.cell(i, 1).value)) + ".mp4"
            diag = str(sheet.cell(i, 7).value)
            if len(diag) > 0 and (diag != 'INDET'):
                if diag == 'NEG':
                    num_neg += 1
                    flag_neg, flag_pos = 1, 0
                else:
                    num_pos += 1
                    flag_pos, flag_neg = 1, 0

                outpath = os.path.join(output_frames, fname)
                fname = "IMG" + str(int(sheet.cell(i, 1).value)) + ".mp4"
                fpath = os.path.join(video_dir, fname)

                videodata = skvideo.io.vread(fpath)
                tot_frame, height, width, num_chan = videodata.shape

                save_neg, save_pos = 1, 1
                count, count_neg, count_pos = 1, 1, 1

                # iter_neg = (tot_frame / 17)	# 17 x negative -- Floor division by default
                # iter_pos = (tot_frame / 10)	# 10 x positive -- Floor division by default
                vidcap = skvideo.io.vreader(fpath)
                for frame in vidcap:
                    if flag_neg and (count_neg < 18):
                        parts = outpath.split(".")
                        name = parts[0] + "_" + str(count) + ".jpg"
                        # print("Saving negative frame " + name + ", save_neg = " + str(save_neg))
                        cv2.imwrite(name, frame)  # Save frame as JPEG file
                        save_neg += 1
                        count_neg += 1
                    if count_neg == 18:
                        count_neg, save_neg = 1, 1
                        break

                    if flag_pos and (count_pos < 11):
                        parts = outpath.split(".")
                        name = parts[0] + "_" + str(count) + ".jpg"
                        # print("Saving positive frame " + name + ", save_pos = " + str(save_pos))
                        cv2.imwrite(name, frame)  # Save frame as JPEG file
                        save_pos += 1
                        count_pos += 1
                    if count_pos == 11:
                        count_pos, save_pos = 1, 1
                        break

                    if (cv2.waitKey(10)) == 27:  # Exit if Escape is hit
                        break
                    count += 1

                    # print(fname + " has diagnosis " + diag)     # Start of B-MODE IDS

        print("num_neg = " + str(num_neg))
        print("num_pos = " + str(num_pos))

    @staticmethod
    def get_sheet(ref):

        workbook = xlrd.open_workbook(ref)
        sheet = workbook.sheet_by_name('Sheet1')

        return sheet

    @staticmethod
    def print_stats(sheet, type):
        count, count_neg, count_pos = 0, 0, 0

        if type == 'bmode':
            for i in range(414, 846):  # b-mode images range(414, 846)
                # Value of ith row and 2nd column
                diag = str(sheet.cell(i, 7).value)
                if str(sheet.cell(i, 8).value).strip() != '0':
                    if diag == 'NEG':
                        count_neg += 1
                    elif diag == 'POS':
                        count_pos += 1
                    else:
                        pass
                    count += 1

            print('Total number of b-mode videos = ' + str(count))
            print('Number of negative pathology videos = ' + str(count_neg))
            print('Number of positive pathology videos = ' + str(count_pos))
        else:
            for i in range(2, 413):  # m-mode images range(1, 414)
                # Value of ith row and 2nd column
                diag = str(sheet.cell(i, 7).value)
                if str(sheet.cell(i, 8).value).strip() != '0':
                    if diag == 'NEG':
                        count_neg += 1
                    elif diag == 'POS':
                        count_pos += 1
                    else:
                        pass
                    count += 1

            print('Total number of m-mode images = ', count)
            print('Number of negative pathology images = ', count_neg)
            print('Number of positive pathology images = ', count_pos)

    def get_directories(self, type):

        if type == 'bmode':
            source_dir = os.path.join(self.prototype, 'bmode_frames')
            test_oneg = os.path.join(self.prototype, 'bmode_retreival/test/negative')
            test_opos = os.path.join(self.prototype, 'bmode_retreival/test/positive')
            train_oneg = os.path.join(self.prototype, 'bmode_retreival/train/negative')
            train_opos = os.path.join(self.prototype, 'bmode_retreival/train/positive')
        else:
            source_dir = os.path.join(self.prototype, 'mmode_images')
            test_oneg = os.path.join(self.prototype, 'mmode_retreival/test/negative')
            test_opos = os.path.join(self.prototype, 'mmode_retreival/test/positive')
            train_oneg = os.path.join(self.prototype, 'mmode_retreival/train/negative')
            train_opos = os.path.join(self.prototype, 'mmode_retreival/train/positive')

        # Recursively create directories if they don't exist
        if not os.path.isdir(test_oneg):
            os.makedirs(test_oneg)
        if not os.path.isdir(test_opos):
            os.makedirs(test_opos)
        if not os.path.isdir(train_oneg):
            os.makedirs(train_oneg)
        if not os.path.isdir(train_opos):
            os.makedirs(train_opos)

        return source_dir, test_oneg, test_opos, train_oneg, train_opos

    def partition_and_store(self, sheet, type):

        count = 0
        test_cneg, test_cpos, train_cneg, train_cpos = 0, 0, 0, 0

        # 2/12/2018 Information Retrieval Statistics (B-Mode)
        # 404 (148(-) + 256(+)) videos -> 2517(-) + 2561(+) = 5078 images
        # test:  404 x 20% vids =  81 =  29(-) +  52(+) =  493(29x17)(-) +  520(52x10)(+) = 1013 ~ 5078 x 20% images = 1015.6
        # train: 404 x 80% vids = 323 = 120(-) + 203(+) = 2040(120x17)(-) + 2030(203x10)(+) = 4070 ~ 5078 x 80% images = 4062.4
        # total:                = 404 vids

        # 1/13/2018 Information Retrieval Statistics (M-Mode)
        # 209 images
        # test:  21(-) +  20(+) =  41 = 209 x 20%
        # train: 52(-) + 116(+) = 168 = 209 x 80%
        # total:                = 209

        # 3/10/2018 Information Retrieval Statistics (M-Mode)
        # 209 bmp + x 195 jpg = 404 images
        # test:  41(-) +  40(+) =  81 = 404 x 20%
        # train: 90(-) + 233(+) = 323 = 404 x 80%
        # total:                = 404

        source_dir, test_oneg, test_opos, train_oneg, train_opos = self.get_directories(type)

        if type == 'bmode':
            start, end = 414, 846
            TEST_NEG, TEST_POS, TRAIN_NEG, TRAIN_POS = 29, 52, 120, 203
        else:
            start, end = 2, 413
            TEST_NEG, TEST_POS, TRAIN_NEG, TRAIN_POS = 41, 40, 90, 233

        for i in range(start, end):  # m-mode images range(1, 414) / b-mode images range(413, 846)
            # Value of ith row and 2nd column
            if type == 'bmode':
                fname = "IMG" + str(int(sheet.cell(i, 1).value)) + "_*" + ".jpg"
            else:
                fname = "IMG" + str(int(sheet.cell(i, 1).value)) + ".*"

            fpath = os.path.join(source_dir, fname)
            diag = str(sheet.cell(i, 7).value)
            if str(sheet.cell(i, 8).value).strip() != '0':
                if diag == 'NEG' and test_cneg < TEST_NEG:
                    for file in glob.glob(fpath):
                        head, tail = os.path.split(file)
                        outpath = os.path.join(test_oneg, tail)
                        copyfile(file, outpath)
                        test_cneg += 1
                elif diag == 'POS' and test_cpos < TEST_POS:
                    for file in glob.glob(fpath):
                        head, tail = os.path.split(file)
                        outpath = os.path.join(test_opos, tail)
                        copyfile(file, outpath)
                        test_cpos += 1
                else:
                    pass

                if diag == 'NEG' and train_cneg < TRAIN_NEG:
                    for file in glob.glob(fpath):
                        head, tail = os.path.split(file)
                        outpath = os.path.join(train_oneg, tail)
                        copyfile(file, outpath)
                        train_cneg += 1
                elif diag == 'POS' and train_cpos < TRAIN_POS:
                    for file in glob.glob(fpath):
                        head, tail = os.path.split(file)
                        outpath = os.path.join(train_opos, tail)
                        copyfile(file, outpath)
                        train_cpos += 1
                else:
                    pass

                count += 1

        print('test_cneg = ', test_cneg, ', train_cneg = ', train_cneg)
        print('test_cpos = ', test_cpos, ', train_cpos = ', train_cpos)

        print('count = ' + str(count))
        test_count = int(round(count * 0.2))  # Floor division by default
        print('test set = ' + str(test_count) + ' images')  # Floor division by default
        # pos_count = test_count / 2  # Floor division by default
        # print('test set positive = ' + str(pos_count) + ' images')  # Floor division by default
        # print('test set negative = ' + str(test_count - pos_count) + ' images')  # Floor division by default

        train_count = int(round(count * 0.8))  # Floor division by default
        print('train set = ' + str(train_count) + ' images')  # Floor division by default
        # pos_count = train_count / 2  # Floor division by default
        # print('train set positive = ' + str(pos_count) + ' images')  # Floor division by default
        # print('train set negative = ' + str(train_count - pos_count) + ' images')  # Floor division by default


class Preprocess:
    def __init__(self):

        self.PathHome = getcwd()
        self.Mode = 'mmode'
        self.SavePath = join(self.PathHome, 'processed')
        self.GrayImage = np.zeros((1, 1))
        self.CleanedImage = np.zeros((1, 1))
        self.FileExt = '.bmp'

    def set_path_home(self, path):
        self.PathHome = path
        self.SavePath = join(self.PathHome, 'processed')

    def set_mode(self, mode):
        self.Mode = mode

    def set_save_path(self, savename):
        self.SavePath = savename

    def remove_background(self):

        img = self.GrayImage
        img -= np.min(img)
        img /= np.max(img)

        # Use histogram of histogram equalized image to find threshold
        equ = exposure.equalize_hist(img, 256)

        hist, bins = exposure.histogram(equ, 256)
        h_thresh_idx = np.argmax(hist)
        if self.FileExt == '.jpg':
            h_thresh_idx += 4
        else:
            h_thresh_idx += 2

        thresh = bins[h_thresh_idx]
        bw = equ > thresh
        fctr = 0.85
        d0_thr = img.shape[0] * fctr
        d1_thr = img.shape[1] * fctr

        # If an object is too large, increase the threshold to avoid filtering nothing out.
        thr_bad = True
        while thr_bad:
            thresh = bins[h_thresh_idx]
            bw = equ > thresh
            lb_img = measure.label(bw, neighbors=4)
            prps = measure.regionprops(lb_img)
            thr_ok = True
            for p in prps:
                cds = np.transpose(p.coords)
                if (np.max(cds[0]) - np.min(cds[0])) > d0_thr or (np.max(cds[1]) - np.min(cds[1])) > d1_thr:
                    thr_ok = False
                    h_thresh_idx += 1
                    break
            if thr_ok:
                thr_bad = False

        # Label the distinct objects in the thresholded image
        lb_img = measure.label(bw, neighbors=4)

        # Find region properties to determine the size of each object
        prps = measure.regionprops(lb_img)

        # Create a binary image indicating which objects should remain in the image based on size
        keep_cds = np.zeros([img.shape[0], img.shape[1]])

        # Filter out objects that are small.
        thresh2 = img.shape[0] * img.shape[1] / 20
        for p in prps:
            if p.area > thresh2:
                cds = np.transpose(p.coords)
                keep_cds[cds[0], cds[1]] = 1

        # Use the binary image to create the new cleaned image
        bw = morphology.binary_opening(keep_cds, morphology.square(5))
        if self.FileExt == '.jpg':
            bw = morphology.binary_closing(bw, morphology.square(4))
        else:
            bw = morphology.binary_closing(bw, morphology.square(7))

        bw = morphology.remove_small_holes(bw, 500)
        lb_img = measure.label(bw, neighbors=4)
        prps = measure.regionprops(lb_img)
        m_prp = []
        m_val = 0
        for p in prps:
            if p.area > m_val:
                m_prp = p
                m_val = p.area

        cds = np.transpose(m_prp.coords)
        min_r = np.min(cds[0])
        idx = np.where(bw > 0)
        max_r = np.max(idx[0])
        min_c = np.min(cds[1])
        max_c = np.max(cds[1])

        bw[min_r:max_r, min_c:max_c] = 1

        idxs = np.where(bw > 0)

        new_img = np.zeros([img.shape[0], img.shape[1]])
        new_img[idxs[0], idxs[1]] = img[idxs[0], idxs[1]]

        self.CleanedImage = new_img

    def get_vert_lines(self, sz_filt):

        img = self.CleanedImage

        img -= np.min(img)
        img /= np.max(img)

        # Smooth image
        if sz_filt > 0:
            img = filters.median(img, morphology.disk(sz_filt))

        # Use vertical Sobel filter (since we know the lines are always the same orientation)
        sb = filters.sobel_v(img)

        # Normalize data between 0 and 1
        sb -= np.min(sb)
        sb /= np.max(sb)

        # Find Yen threshold
        thr = filters.threshold_yen(sb)

        thr_img = sb < thr
        if np.sum(thr_img) > np.sum(~thr_img):
            thr_img = ~thr_img

        thr_img = morphology.binary_dilation(thr_img, morphology.square(4))
        thr_img = morphology.binary_erosion(thr_img, morphology.square(4))

        return thr_img

    def detect_lines(self):

        img = self.CleanedImage
        img -= np.min(img)
        img /= np.max(img)

        sz_filt = 0

        # Get vertical lines using sobel_v filter.
        thr_img = self.get_vert_lines(sz_filt)

        # If there's a huge number of lines found, use a larger filter to smooth the image and find lines again.
        n_col_lines = np.sum(np.sum(thr_img, axis=0) > 0)
        sz_filt = 1
        while n_col_lines > img.shape[1] / 2:
            thr_img = self.get_vert_lines(sz_filt)
            sz_filt += 2
            n_col_lines = np.sum(np.sum(thr_img, axis=0) > 0)

        return thr_img

    def crop_image(self):

        # Crop image to indices of nonzero values and the width of the largest region.
        img = self.CleanedImage
        img -= np.min(img)
        img /= np.max(img)
        idx = np.where(img > 0)

        mask = img > 0

        if self.Mode == 'mmode':
            pix_per_col = np.sum(mask, axis=0)

            n_pix_thr = np.floor(mask.shape[1] / 6)

            ln_cols = np.where(pix_per_col < n_pix_thr)
            ln_cols = ln_cols[0]

            for l in ln_cols:
                mask[:, l] = 0

        lb_img = measure.label(mask, neighbors=4)
        prps = measure.regionprops(lb_img)

        mx = 0
        mx_idx = 0
        for i in range(len(prps)):
            p = prps[i]
            if p.area > mx:
                mx = p.area
                mx_idx = i

        cds = np.transpose(prps[mx_idx].coords)

        min_c = np.min(cds[1])
        max_c = np.max(cds[1])
        min_r = np.min(idx[0])
        max_r = np.max(idx[0])

        crpd_img = img[min_r:max_r - 1, min_c:max_c - 1]

        return crpd_img

    def blend_lines_into_image(self, thr_img):

        img = self.CleanedImage

        # Normalize image
        base_img = img - np.min(img)
        base_img /= np.max(base_img)

        # This is to avoid having many small line segments. If there are enough line pixels present in a column,
        # the line is extended between the min and max rows, so we just have to deal with one line per column.
        pix_per_col = np.sum(thr_img, axis=0)

        n_pix_thr = np.floor(thr_img.shape[0] / 15)

        ln_cols = np.where(pix_per_col > n_pix_thr)
        ln_cols = ln_cols[0]

        for l in ln_cols:
            idx = np.where(thr_img[:, l] > 0)
            idx = idx[0]
            mx = np.max(idx)
            mn = np.min(idx)
            thr_img[mn:mx, l] = 1

        thr_img = thr_img.astype('int')

        lb_img = measure.label(thr_img, neighbors=4)
        prps = measure.regionprops(lb_img)

        # Threshold used to get rid of extraneous spots/short lines
        thresh = img.shape[0] * img.shape[1] / 5000
        for p in prps:
            cds = np.transpose(p.coords)
            # If any of the object is outside the designated region, remove it from the labeled image.
            if p.area < thresh:
                lb_img[cds[0], cds[1]] = 0

        # Blend lines
        final_img = np.copy(base_img)

        # For each line, fill with neighboring pixel values and filter to obtain final line pixel values.
        prps = measure.regionprops(lb_img)
        for p in prps:
            obj_img = np.zeros([lb_img.shape[0], lb_img.shape[1]])
            cds = np.transpose(p.coords)
            obj_img[cds[0], cds[1]] = 1

            bw = obj_img > 0

            bw_idx = np.where(bw)
            min_r = np.min(bw_idx[0])
            max_r = np.min([np.max(bw_idx[0] + 2), bw.shape[0]])

            bins = np.unique(bw_idx[1])
            hist, b = np.histogram(bw_idx[1], len(bins))

            col_idx = bins[hist > (max_r - min_r)/2]

            if len(col_idx) == 0:
                continue

            min_c = np.min(col_idx) - 3
            max_c = np.max(col_idx) + 2

            init_img = np.copy(base_img)

            # Fill in line with values from neighboring pixels.
            tmp_bfr = 0

            st_idx = np.max([min_c - tmp_bfr, 0])
            fn_idx = np.min([max_c + tmp_bfr, init_img.shape[1]])

            col_thr = 0.4

            rt_idx = 0
            for i in range(st_idx - tmp_bfr, st_idx + int(np.ceil((fn_idx - st_idx) / 2)+1)):
                ns_i = (np.random.randn(max_r - min_r) - 0.1) / 100
                ns_fn_i = (np.random.randn(max_r - min_r) - 0.1) / 100
                inp_i = init_img[min_r:max_r, np.max([i - 1, 1])]
                inp_fn_i = init_img[min_r:max_r, np.min([fn_idx - rt_idx + 1, init_img.shape[1] - 1])]
                tmp_i = inp_i - np.min(inp_i)
                tmp_fn_i = inp_fn_i - np.min(inp_fn_i)
                if np.sum(tmp_i > 0) < (col_thr * len(tmp_i)):
                    inp_i = inp_fn_i
                if np.sum(tmp_fn_i > 0) < (col_thr * len(tmp_fn_i)):
                    inp_fn_i = inp_i

                init_img[min_r:max_r, i] = inp_i + ns_i
                init_img[min_r:max_r, fn_idx - rt_idx] = inp_fn_i + ns_fn_i
                rt_idx += 1

            init_img[init_img < 0] = 0
            init_img[init_img > 1] = 1

            # Smooth image to get line values
            bfr = 0
            tmp_img = filters.gaussian(init_img, sigma=0.5)

            min_c = np.max([min_c - bfr, 0])
            max_c = np.min([max_c + bfr, init_img.shape[1]])
            min_r = np.max([min_r, 0])
            max_r = np.min([max_r + 1, init_img.shape[0]])

            final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img))
            base_img = (base_img - np.min(base_img)) / (np.max(base_img) - np.min(base_img))
            for i in range(min_c, max_c + 1):
                final_img[min_r:max_r, i] = tmp_img[min_r:max_r, i]
                base_img[min_r:max_r, i] = tmp_img[min_r:max_r, i]

        self.CleanedImage = final_img

        # Crop image.
        crpd_final_img = self.crop_image()

        return crpd_final_img

    def remove_text(self):

        cleaned_img = self.CleanedImage
        nb_pixels = 50

        def find_text(im, nb_pix):
            lb_im = measure.label(im, neighbors=8)
            ps = measure.regionprops(lb_im)

            rg_area = 6
            n_img = np.zeros((im.shape[0], im.shape[1]))
            for p in ps:
                cds = np.transpose(p.coords)
                if p.area > rg_area and (np.max(cds[0]) < nb_pix or np.min(cds[0]) > im.shape[0] - nb_pix) and (
                                np.max(cds[1]) < nb_pix or np.min(cds[1]) > im.shape[1] - nb_pix):
                    n_img[cds[0], cds[1]] = 1

            b_im = morphology.binary_dilation(n_img, morphology.square(15))
            return b_im

        def use_sbv(cleaned_im):
            sbv = filters.sobel_v(cleaned_im)

            th = filters.threshold_otsu(sbv)
            im = sbv > th

            if np.sum(im) > np.sum(~im):
                im = sbv < th
            im = morphology.binary_dilation(im, morphology.square(1))

            return im

        if self.FileExt == '.jpg':
            img = use_sbv(cleaned_img)
        else:
            img = feature.peak_local_max(cleaned_img, indices=False)
            if len(np.unique(img)) == 1:
                print('not blobs')
                img = use_sbv(cleaned_img)

            if np.sum(img) > np.sum(~img):
                img = ~img

            img = morphology.binary_dilation(img, morphology.square(2))

        b_img = find_text(img, nb_pixels)

        cleaned_img2 = np.copy(cleaned_img)

        lb_img = measure.label(b_img)
        prps = measure.regionprops(lb_img)

        thr = np.round(nb_pixels * 0.75)
        for p in prps:
            idx = np.transpose(p.coords)
            if len(idx[0]) > 0:
                if img.shape[1] - np.max(idx[1]) < thr:
                    max_c = img.shape[1]
                else:
                    max_c = np.max(idx[1])
                if np.min(idx[1]) < thr:
                    min_c = 1
                else:
                    min_c = np.min(idx[1])

                if img.shape[0] - np.max(idx[0]) < thr:
                    max_r = img.shape[0]
                else:
                    max_r = np.max(idx[0])
                if np.min(idx[0]) < thr:
                    min_r = 1
                else:
                    min_r = np.min(idx[0])

                width = max_c - min_c
                if np.min(idx[1]) > width:
                    cleaned_img2[min_r:max_r, min_c:max_c] = cleaned_img2[min_r:max_r, min_c - width:min_c]
                elif np.max(idx[1]) < (img.shape[1] - width):
                    cleaned_img2[min_r:max_r, min_c:max_c] = cleaned_img2[min_r:max_r, max_c:max_c + width]

        self.CleanedImage = cleaned_img2

    def clean_images(self, home_dir='', save_dir=''):

        if home_dir != '':
            self.set_path_home(home_dir)
        if save_dir != '':
            self.set_save_path(save_dir)

        st_time = time.clock()

        # Run through thr image folders and clean images.
        n_imgs = 10000
        img_idx = 0

        save_dir = self.SavePath

        if self.Mode == 'mmode':
            img_dir = join(self.PathHome,'mmode_images')
        else:
            img_dir = join(self.PathHome,'bmode_images')

        if not isdir(save_dir):
            mkdir(save_dir)
        else:
            prev_files = listdir(save_dir)
            for f in prev_files:
                if isdir(f):
                    rmtree(join(save_dir, f))
                else:
                    os.remove(join(save_dir, f))


        list_imgs = listdir(img_dir)
        for name_img in list_imgs:
            if img_idx < n_imgs:
                if '.bmp' not in name_img and '.jpg' not in name_img and '.png' not in name_img:
                    print(name_img)
                    continue

                path_img = join(img_dir, name_img)
                c_img = Image.open(path_img)
                pil_imgray = c_img.convert('LA')
                c_img = np.asarray(c_img)
                img = np.array(list(pil_imgray.getdata(band=0)), float)
                img.shape = (pil_imgray.size[1], pil_imgray.size[0])

                self.FileExt = name_img[len(name_img) - 4:len(name_img)]
                self.GrayImage = img

                self.remove_background()
                thr_img = self.detect_lines()
                cleaned_img = self.blend_lines_into_image(thr_img)
                cleaned_img -= np.min(cleaned_img)
                cleaned_img /= np.max(cleaned_img)

                crpd_final_img = join(save_dir, name_img)
                self.CleanedImage = cleaned_img
                self.remove_text()

                imageio.imwrite(crpd_final_img[0:len(crpd_final_img) - 4] + '.png', self.CleanedImage)
                img_idx += 1

        end_time = time.clock()

        print('Pre-processing time:', end_time - st_time)


class DataAugmentation(object):
    def __init__(self):

        self.home = expanduser('~')
        self.prototype = os.path.join(self.home, 'Documents/Develop/prototype')
        self.fname = os.path.join(self.home, 'PycharmProjects/DataAugmentation/data_augment.py')

    @staticmethod
    def clahe(source, dest):

        for subdir, dirs, files in os.walk(source):
            for file in files:
                # -----Reading the image-----------------------------------------------------
                fname = os.path.join(subdir, file)
                img = cv2.imread(fname, 0)

                # -----Applying CLAHE to L-channel-------------------------------------------
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(img)

                # -----Converting image from LAB Color model to RGB model--------------------
                file = 'clahe_' + file
                outdir = dest

                output_fname = os.path.join(outdir, file)

                cv2.imwrite(output_fname, cl)
        return 0

    def train_val_split(self, type, percent):

        if type == 'bmode':
            train_dest_neg = os.path.join(self.prototype, 'model/bmode/train/negative')
            train_dest_pos = os.path.join(self.prototype, 'model/bmode/train/positive')
            val_dest_neg = os.path.join(self.prototype, 'model/bmode/val/negative')
            val_dest_pos = os.path.join(self.prototype, 'model/bmode/val/positive')
            source_neg = os.path.join(self.prototype, 'bmode_retreival/train/negative')
            source_pos = os.path.join(self.prototype, 'bmode_retreival/train/positive')
        else:
            train_dest_neg = os.path.join(self.prototype, 'model/mmode/train/negative')
            train_dest_pos = os.path.join(self.prototype, 'model/mmode/train/positive')
            val_dest_neg = os.path.join(self.prototype, 'model/mmode/val/negative')
            val_dest_pos = os.path.join(self.prototype, 'model/mmode/val/positive')
            source_neg = os.path.join(self.prototype, 'mmode_retreival/train/negative')
            source_pos = os.path.join(self.prototype, 'mmode_retreival/train/positive')

        # Recursively create destination directories if they don't exist
        if not os.path.isdir(train_dest_neg):
            os.makedirs(train_dest_neg)
        if not os.path.isdir(train_dest_pos):
            os.makedirs(train_dest_pos)
        if not os.path.isdir(val_dest_neg):
            os.makedirs(val_dest_neg)
        if not os.path.isdir(val_dest_pos):
            os.makedirs(val_dest_pos)

        tcount = 0
        # Negative Cases
        if type == 'mmode':
            sfiles = glob.glob(os.path.join(source_neg, '*.*'))
        else:
            sfiles = glob.glob(os.path.join(source_neg, '*.mp4'))

        tsplit = round(percent * len(sfiles))
        # print('tsplit negative = ', tsplit, '; length of sfiles = ', len(sfiles))
        for fn in sfiles:
            fn = os.path.basename(fn)
            sfile = os.path.join(source_neg, fn)
            if tcount < tsplit:
                dfile = os.path.join(train_dest_neg, fn)
                # print('train_dest_neg = ', train_dest_neg, '; fn = ', fn)
                # print('sfile = ', sfile, '; dfile = ', dfile)
                copyfile(sfile, dfile)
                tcount += 1
                # print('tcount = ', tcount, '; tsplit = ', tsplit)
            else:
                dfile = os.path.join(val_dest_neg, fn)
                copyfile(sfile, dfile)
        # print 'train negative number = ', tcount

        tcount = 0
        # Positive Cases
        if type == 'mmode':
            sfiles = glob.glob(os.path.join(source_pos, '*.*'))
        else:
            sfiles = glob.glob(os.path.join(source_pos, '*.mp4'))

        tsplit = round(percent * len(sfiles))
        # print('tsplit positive = ', tsplit, '; length of sfiles = ', len(sfiles))
        for fn in sfiles:
            fn = os.path.basename(fn)
            sfile = os.path.join(source_pos, fn)
            if tcount < tsplit:
                dfile = os.path.join(train_dest_pos, fn)
                copyfile(sfile, dfile)
                tcount += 1
            else:
                dfile = os.path.join(val_dest_pos, fn)
                copyfile(sfile, dfile)
                # print 'train positive number = ', tcount

    def put_directories(self, type):

        if type == 'bmode':
            clahe_neg = os.path.join(self.prototype, 'bmode_retreival/clahe_negative')
            clahe_pos = os.path.join(self.prototype, 'bmode_retreival/clahe_positive')
            train_neg = os.path.join(self.prototype, 'bmode_retreival/train/negative')
            train_pos = os.path.join(self.prototype, 'bmode_retreival/train/positive')
        else:
            clahe_neg = os.path.join(self.prototype, 'mmode_retreival/clahe_negative')
            clahe_pos = os.path.join(self.prototype, 'mmode_retreival/clahe_positive')
            train_neg = os.path.join(self.prototype, 'mmode_retreival/train/negative')
            train_pos = os.path.join(self.prototype, 'mmode_retreival/train/positive')

        # Recursively create directories if they don't exist
        if not os.path.isdir(clahe_neg):
            os.makedirs(clahe_neg)
        if not os.path.isdir(clahe_pos):
            os.makedirs(clahe_pos)
        if not os.path.isdir(train_neg):
            os.makedirs(train_neg)
        if not os.path.isdir(train_pos):
            os.makedirs(train_pos)

        return clahe_neg, clahe_pos, train_neg, train_pos

    def affine_transform_bmode(self, type):
        # 2/12/2018 Information Retrieval Statistics (B-Mode)
        # 404 (148(-) + 256(+)) videos -> 2517(-) + 2561(+) = 5078 images
        # test:  404 x 20% vids =  81 =  29(-) +  52(+) =  493(29x17)(-) +  520(52x10)(+) = 1013 ~ 5078 x 20% images = 1015.6
        # train: 404 x 80% vids = 323 = 120(-) + 203(+) = 2040(120x17)(-) + 2030(203x10)(+) = 4070 ~ 5078 x 80% images = 4062.4
        # total:                = 404 vids

        # Clahe plus data augmentation
        # original + clahe + kxnumber
        # 2040(-) + 2040(-) + 3x2040(-) = 10200(-) images
        # 2030(+) + 2030(+) + 3x2030(+) = 10150(+) images

        clahe_neg, clahe_pos, train_neg, train_pos = self.put_directories(type)

        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_neg + ' '
        cmd1 = prefix + 'fliph,flipv'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'

        cmd_list = [cmd1, cmd2, cmd3]

        for i in range(1, 4):
            print('i = ', i, ' cmd = ', cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        # self.clahe(train_pos, clahe_pos)
        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_pos + ' '

        cmd1 = prefix + 'noise_0.01,blur_0.50'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'

        cmd_list = [cmd1, cmd2, cmd3]

        for i in range(1, 4):
            print('i = ' + str(i) + ' cmd = ' + cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        # Copy clahe images to train_neg directory
        source, dpath = clahe_neg, train_neg
        for subdir, dirs, files in os.walk(source):
            for file in files:
                # -----Reading the image-----------------------------------------------------
                sfile = os.path.join(subdir, file)
                dfile = os.path.join(dpath, file)
                copyfile(sfile, dfile)

                # Remove clahe_negative & clahe_positive

    def affine_transform_mmode(self, type):
        # 1/13/2018 Information Retrieval Statistics (M-Mode)
        # 209 images
        # test:  21(-) +  20(+) =  41 = 209 x 20%
        # train: 52(-) + 116(+) = 168 = 209 x 80%
        # total:                = 209

        # Clahe plus data augmentation
        # original + clahe + kxnumber
        #  52(-) +  52(-) + 16x52(-) = 936(-) images
        # 116(+) + 116(+) + 6x116(+) = 928(+) images


        # 3/10/2018 Information Retrieval Statistics (M-Mode)
        # 209 bmp + x 195 jpg = 404 images
        # test:  41(-) +  40(+) =  81 = 404 x 20%
        # train: 90(-) + 233(+) = 323 = 404 x 80%
        # total:                = 404

        # Clahe plus data augmentation
        # original + clahe + kxnumber
        #  90(-) +  90(-) + 11x90(-) = 1170(-) images
        # 233(+) + 233(+) + 3x233(+) = 1165(+) images

        clahe_neg, clahe_pos, train_neg, train_pos = self.put_directories(type)

        if not os.path.isdir(clahe_neg):
            os.makedirs(clahe_neg)
        if not os.path.isdir(clahe_pos):
            os.makedirs(clahe_pos)
        if not os.path.isdir(train_neg):
            os.makedirs(train_neg)
        if not os.path.isdir(train_pos):
            os.makedirs(train_pos)

        self.clahe(train_neg, clahe_neg)

        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_neg + ' '
        cmd1 = prefix + 'fliph'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'
        cmd4 = prefix + 'rot_-45,blur_1.0'
        cmd5 = prefix + 'noise_0.03,flipv'
        cmd6 = prefix + 'fliph,trans_-10_0'
        cmd7 = prefix + 'flipv'
        cmd8 = prefix + 'rot_45'
        cmd9 = prefix + 'blur_0.5'
        cmd10 = prefix + 'blur_1.5,trans_20_10'
        cmd11 = prefix + 'noise_0.04,trans_-10_0'
        '''
        cmd12 = prefix + 'rot_90,blur_0.75'
        cmd13 = prefix + 'noise_0.01,rot_-90'
        cmd14 = prefix + 'noise_0.01,blur_0.50'
        cmd15 = prefix + 'flipv,trans_20_10'
        cmd16 = prefix + 'fliph,flipv'
        '''

        cmd_list = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8,
                    cmd9, cmd10, cmd11]

        for i in range(1, 12):
            print('i = ', i, ' cmd = ', cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        self.clahe(train_pos, clahe_pos)
        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_pos + ' '

        cmd1 = prefix + 'noise_0.01,blur_0.50'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'
        '''
        cmd4 = prefix + 'rot_-45,blur_1.0'
        cmd5 = prefix + 'noise_0.03,flipv'
        cmd6 = prefix + 'fliph,trans_-10_0'
        '''

        cmd_list = [cmd1, cmd2, cmd3]

        for i in range(1, 4):
            print('i = ' + str(i) + ' cmd = ' + cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        # Copy clahe images to train_neg directory
        source, dpath = clahe_neg, train_neg
        for subdir, dirs, files in os.walk(source):
            for file in files:
                # -----Reading the image-----------------------------------------------------
                sfile = os.path.join(subdir, file)
                dfile = os.path.join(dpath, file)
                copyfile(sfile, dfile)

        # Copy clahe images to train_pos directory
        source, dpath = clahe_pos, train_pos
        for subdir, dirs, files in os.walk(source):
            for file in files:
                # -----Reading the image-----------------------------------------------------
                sfile = os.path.join(subdir, file)
                dfile = os.path.join(dpath, file)
                copyfile(sfile, dfile)

        # Remove clahe_negative & clahe_positive
        rmtree(clahe_neg)
        rmtree(clahe_pos)


def main():
    dir = '/home/rlee/Documents/pneumothorax/'
    save_path = '/home/rlee/Documents/pneumothorax/processed_bmode'
    process = Preprocess()
    process.set_mode('bmode')
    process.clean_images(dir, save_path)

    '''
    info = InformationRetreival()
    ref = os.path.join(info.home, 'APD_PIG_Master_Data_Sheet.xlsx')
    sheet = info.get_sheet(ref)

    bmode_video = os.path.join(info.prototype, 'bmode_video')
    bmode_frames = os.path.join(info.prototype, 'bmode_frames')

    if not os.path.isdir(bmode_frames):
        os.makedirs(bmode_frames)

    info.extract_frames(sheet, bmode_video, bmode_frames)
    info.print_stats(sheet, 'bmode')
    info.partition_and_store(sheet, 'bmode')

    info.print_stats(sheet, 'mmode')
    info.partition_and_store(sheet, 'mmode')

    process = PreProcess()
    print(process.directory)

    daugment = DataAugmentation()
    daugment.affine_transform_bmode('bmode')
    daugment.affine_transform_mmode('mmode')
    # 80% train / 20% validation
    daugment.train_val_split('mmode', 0.8)
    '''


if __name__ == '__main__':
    main()
