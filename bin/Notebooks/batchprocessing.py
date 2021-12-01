#!/usr/bin/env python

from __future__ import print_function

import collections
import csv
import logging
import os

import SimpleITK as sitk

import radiomics
from radiomics import featureextractor


def main():
    outPath = os.path.join(os.getcwd(), "..", "..", "r")
    dataDir = os.path.join(os.getcwd(), "..", "..", "data")
    input_csv = "py_rad_NAV.csv"
    inputCSV = os.path.join(outPath, input_csv)
    outputFilepath = os.path.join(outPath, 'NAV_radiomics.csv')
    progress_filename = os.path.join(outPath, 'pyrad_log.txt')
    #params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')
    params = os.path.join(os.getcwd(), "..", "Params.yaml")
    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    flists = []
    try:
        with open(inputCSV, 'r') as inFile:
            cr = csv.DictReader(inFile, lineterminator='\n')
            flists = [row for row in cr]
    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists))

    if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None

    for idx, entry in enumerate(flists, start=1):

        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['IMAGE'], entry['MASK'])

        imageFilepath = entry['IMAGE']
        maskFilepath = entry['MASK']
        label = entry.get('Label', None)
        #cl = entry['LABEL']

        if str(label).isdigit():
            print("The label is a digit")
            label = int(label)
        else:
            print("The label is not a digit")
            label = None

        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = collections.OrderedDict(entry)
            featureVector['IMAGE'] = os.path.basename(imageFilepath)
            featureVector['MASK'] = os.path.basename(maskFilepath)

            try:
                featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

                with open(outputFilepath, 'a') as outputFile:
                    writer = csv.writer(outputFile, lineterminator='\n')
                    if headers is None:
                        headers = list(featureVector.keys())
                        writer.writerow(headers)

                    row = []
                    for h in headers:
                        row.append(featureVector.get(h, "N/A"))
                    #print(row)
                    writer.writerow(row)

            except Exception:
                logger.error('FEATURE EXTRACTION FAILED', exc_info=True)


if __name__ == '__main__':
    main()
