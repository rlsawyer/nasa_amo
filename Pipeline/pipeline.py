# Python 3.5.2 Anaconda

class InformationRetreival(object):
    def __init__(self):
        import os
        from os.path import expanduser

        self.home = expanduser('~')
        self.prototype = os.path.join(self.home, 'Documents/Develop/prototype')

    def extract_frames(self, sheet, video_dir, output_frames):
        import cv2, os, skvideo.io

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

    def get_sheet(self, ref):
        import xlrd

        workbook = xlrd.open_workbook(ref)
        sheet = workbook.sheet_by_name('Sheet1')

        return sheet

    def print_stats(self, sheet):
        count, count_neg, count_pos = 0, 0, 0

        for i in range(413, 846):  # m-mode images range(1, 414) / b-mode images range(413, 846)
            # Value of ith row and 2nd column
            diag = str(sheet.cell(i, 7).value)
            if len(diag) > 0 and (diag != 'INDET'):
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

    def get_directories(self, type):
        import os

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
        from shutil import copyfile
        import os, glob

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

        source_dir, test_oneg, test_opos, train_oneg, train_opos = self.get_directories(type)

        if type == 'bmode':
            start, end = 413, 846
            TEST_NEG, TEST_POS, TRAIN_NEG, TRAIN_POS = 29, 52, 120, 203
        else:
            start, end = 1, 414
            TEST_NEG, TEST_POS, TRAIN_NEG, TRAIN_POS = 21, 20, 52, 116

        for i in range(start, end):  # m-mode images range(1, 414) / b-mode images range(413, 846)
            # Value of ith row and 2nd column
            if type == 'bmode':
                fname = "IMG" + str(int(sheet.cell(i, 1).value)) + "_*" + ".jpg"
            else:
                fname = "IMG" + str(int(sheet.cell(i, 1).value)) + ".bmp"

            fpath = os.path.join(source_dir, fname)
            diag = str(sheet.cell(i, 7).value)
            if len(diag) > 0 and (diag != 'INDET'):
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
        pos_count = test_count / 2  # Floor division by default
        print('test set positive = ' + str(pos_count) + ' images')  # Floor division by default
        print('test set negative = ' + str(test_count - pos_count) + ' images')  # Floor division by default

        train_count = int(round(count * 0.8))  # Floor division by default
        print('train set = ' + str(train_count) + ' images')  # Floor division by default
        pos_count = train_count / 2  # Floor division by default
        print('train set positive = ' + str(pos_count) + ' images')  # Floor division by default
        print('train set negative = ' + str(train_count - pos_count) + ' images')  # Floor division by default


class PreProcess(object):
    def __init__(self):
        self.directory = 'Import Class Directory'


class DataAugmentation(object):
    def __init__(self):
        import os
        from os.path import expanduser

        self.home = expanduser('~')
        self.prototype = os.path.join(self.home, 'Documents/Develop/prototype')
        self.fname = os.path.join(self.home, 'PycharmProjects/DataAugmentation/data_augment.py')

    def clahe(self, source, dest):
        import cv2, os

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
        import glob, os
        from shutil import copyfile

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
        if type == 'mmode':
            sfiles = glob.glob(os.path.join(source_neg, '*.png'))
        else:
            sfiles = glob.glob(os.path.join(source_neg, '*.mp4'))

        tsplit = round(percent * len(sfiles))
        print('tsplit = ', tsplit, '; length of sfiles = ', len(sfiles))
        for fn in sfiles:
            fn = os.path.basename(fn)
            sfile = os.path.join(source_neg, fn)
            if tcount < tsplit:
                dfile = os.path.join(train_dest_neg, fn)
                #print('train_dest_neg = ', train_dest_neg, '; fn = ', fn)
                #print('sfile = ', sfile, '; dfile = ', dfile)
                copyfile(sfile, dfile)
                tcount += 1
                print('tcount = ', tcount, '; tsplit = ', tsplit)
            else:
                dfile = os.path.join(val_dest_neg, fn)
                copyfile(sfile, dfile)

        tcount = 0
        if type == 'mmode':
            sfiles = glob.glob(os.path.join(source_pos, '*.png'))
        else:
            sfiles = glob.glob(os.path.join(source_pos, '*.mp4'))

        tsplit = round(percent * len(sfiles))
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

    def put_directories(self, type):
        import os

        if type == 'bmode':
            clahe_neg = os.path.join(self.prototype, 'bmode_retreival/clahe_negative')
            clahe_pos = os.path.join(self.prototype, 'bmode_retreival/clahe_positive')
            train_neg = os.path.join(self.prototype, 'bmode_retreival/train/negative')
            train_pos = os.path.join(self.prototype, 'bmode_retreival/train/positive')
            self.train_bmode_neg = train_neg
            self.train_bmode_pos = train_pos

        else:
            clahe_neg = os.path.join(self.prototype, 'mmode_retreival/clahe_negative')
            clahe_pos = os.path.join(self.prototype, 'mmode_retreival/clahe_positive')
            train_neg = os.path.join(self.prototype, 'mmode_retreival/train/negative')
            train_pos = os.path.join(self.prototype, 'mmode_retreival/train/positive')
            self.train_mmode_neg = train_neg
            self.train_mmode_pos = train_pos

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

        from shutil import copyfile
        import os, shlex, subprocess

        clahe_neg, clahe_pos, train_neg, train_pos = self.put_directories(type)

        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_neg + ' '
        cmd1 = prefix + 'fliph,flipv'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'

        cmd_list = [cmd1, cmd2, cmd3]

        for i in range(1, 4):
            print('i = ', i, ' cmd = ', cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        #self.clahe(train_pos, clahe_pos)
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
        # test:  21(-) +  20(+) =  41 = 209 x 20%
        # train: 52(-) + 116(+) = 168 = 209 x 80%
        # total:                = 209

        # Clahe plus data augmentation
        # original + clahe + kxnumber
        #  52(-) +  52(-) + 16x52(-) = 936(-) images
        # 116(+) + 116(+) + 6x116(+) = 928(+) images

        from shutil import copyfile
        import os, shlex, subprocess

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

        prefix = '/usr/bin/python3.5 ' + self.fname + ' ' + train_neg + ' '
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
        cmd12 = prefix + 'rot_90,blur_0.75'
        cmd13 = prefix + 'noise_0.01,rot_-90'
        cmd14 = prefix + 'noise_0.01,blur_0.50'
        cmd15 = prefix + 'flipv,trans_20_10'
        cmd16 = prefix + 'fliph,flipv'

        cmd_list = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6, cmd7, cmd8,
                    cmd9, cmd10, cmd11, cmd12, cmd13, cmd14, cmd15, cmd16]

        for i in range(1, 17):
            print('i = ', i, ' cmd = ', cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        self.clahe(train_pos, clahe_pos)
        prefix = '/usr/bin/python2.7 ' + self.fname + ' ' + train_pos + ' '

        cmd1 = prefix + 'noise_0.01,blur_0.50'
        cmd2 = prefix + 'noise_0.01,trans_20_10'
        cmd3 = prefix + 'noise_0.02,fliph'
        cmd4 = prefix + 'rot_-45,blur_1.0'
        cmd5 = prefix + 'noise_0.03,flipv'
        cmd6 = prefix + 'fliph,trans_-10_0'

        cmd_list = [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6]

        for i in range(1, 7):
            print('i = ' + str(i) + ' cmd = ' + cmd_list[i - 1])
            subprocess.call(shlex.split(cmd_list[i - 1]))

        # Copy clahe images to train_pos directory
        source, dpath = clahe_pos, train_pos
        for subdir, dirs, files in os.walk(source):
            for file in files:
                # -----Reading the image-----------------------------------------------------
                sfile = os.path.join(subdir, file)
                dfile = os.path.join(dpath, file)
                copyfile(sfile, dfile)

        # Remove clahe_negative & clahe_positive
                

def main():
    import os
    '''
    info = InformationRetreival()
    ref = os.path.join(info.prototype, 'APD_PIG_Master_Data_Sheet.xlsx')
    sheet = info.get_sheet(ref)

    bmode_video = os.path.join(info.prototype, 'bmode_video')
    bmode_frames = os.path.join(info.prototype, 'bmode_frames')

    if not os.path.isdir(bmode_frames):
        os.makedirs(bmode_frames)

    info.extract_frames(sheet, bmode_video, bmode_frames)

    info.print_stats(sheet)
    info.partition_and_store(sheet, 'bmode')
    info.partition_and_store(sheet, 'mmode')

    process = PreProcess()
    print(process.directory)
    '''
    daugment = DataAugmentation()
    #daugment.affine_transform_bmode('bmode')
    #daugment.affine_transform_mmode('mmode')
    # 80% train / 20% validation
    daugment.train_val_split('mmode', 0.8)

if __name__ == '__main__':
    main()
