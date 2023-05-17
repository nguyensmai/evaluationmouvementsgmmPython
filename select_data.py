import os
import shutil
import sys


# split Keraal dataset based on annotations from two annotators
# for each annotator: no error - score 0, small error - score 0.25, large error - score 0.5
# final score ranging from 0 (both annotators said it's correct) to 1 (both annotators said it's large error)
# each exercise perfomance split into corresponding folder by score (0, 0.25, 0.5, 0,75, 1) and by correct or incorrect (score <= 0.25 or >= 0.5)
def select_data_from_two_annotators(srcFolder, dstFolder, dataTypes, splitByExerciseType):
    annotations1Src = os.path.join(srcFolder, 'annotator1')
    annotations2Src = os.path.join(srcFolder, 'annotator6')
    if not os.path.exists(annotations1Src):
        return
    filesAnnotations = os.listdir(annotations1Src)
    exerciseTypes = ["CTK", "ELK", "RTK"]
    for annotationFilename in filesAnnotations:
        if not annotationFilename.endswith(".anvil") or len(annotationFilename) < 25:
            continue
        # print(annotationFilename)
        score = 0
        annotationFile1 = open(os.path.join(annotations1Src, annotationFilename), 'r', encoding='utf-8')
        text = annotationFile1.readlines()
        if True in ["SmallError" in line for line in text]:
            score = score + 0.25
        if True in ["LargeError" in line for line in text]:
            score = score + 0.5
        annotationFile1.close()
        if annotationFilename == 'G1A-RTK-R5-Roscoff-041.anvil':
            annotationFile2 = open(os.path.join(annotations2Src, annotationFilename), 'r', encoding='utf-8')
        else:
            annotationFile2 = open(os.path.join(annotations2Src, annotationFilename), 'r', encoding='utf-16')
        text = annotationFile2.readlines()
        if True in ["SmallError" in line for line in text]:
            score = score + 0.25
        if True in ["LargeError" in line for line in text]:
            score = score + 0.5
        annotationFile2.close()

        exerciseType = "None"
        if splitByExerciseType:
            for eType in exerciseTypes:
                if eType in annotationFilename:
                    exerciseType = eType
                    break

        for dataType in dataTypes:
            dataSrc = os.path.join(srcFolder, dataType)
            dataDst = os.path.join(dstFolder, dataType, exerciseType)
            if not os.path.exists(dataDst):
                os.makedirs(os.path.join(dataDst, '0'))
                os.makedirs(os.path.join(dataDst, '0.25'))
                os.makedirs(os.path.join(dataDst, '0.5'))
                os.makedirs(os.path.join(dataDst, '0.75'))
                os.makedirs(os.path.join(dataDst, '1.0'))
                os.makedirs(os.path.join(dataDst, 'correct'))
                os.makedirs(os.path.join(dataDst, 'incorrect'))

            dataTypeNote = 'Kinect' if dataType == 'kinect' else 'Anon' if dataType == 'openpose' else 'BP'
            filename = annotationFilename[:4] + dataTypeNote + '-' + annotationFilename[4:-5] \
                       + ('txt' if dataType == 'kinect' else 'json')

            shutil.copyfile(os.path.join(dataSrc, filename), os.path.join(dataDst, str(score), filename))
            shutil.copyfile(os.path.join(dataSrc, filename), os.path.join(dataDst, 'correct' if score < 0.5 else 'incorrect', filename))


def copyDataset(srcFolder, dstFolder, splitByExerciseType = True):
    for name in os.listdir(srcFolder):
        if name.startswith('group'):
            select_data_from_two_annotators(os.path.join(srcFolder, name), os.path.join(dstFolder, name),
                                            ['kinect', 'openpose', 'blazepose'], splitByExerciseType)


if __name__ == '__main__':
    copyDataset(sys.argv[1], sys.argv[2], sys.argv[3] == 'True' if len(sys.argv) > 3 else True)
