import os
import glob
import sys
from os.path import isfile
from zipfile import ZipFile

from utils import get_instance_files


def main(argv):
    sid_dir_location = argv[0]
    sid = os.path.basename(sid_dir_location)
    files = glob.glob(sid_dir_location + '/**/*', recursive=True)
    instance_files = [file for file in files if isfile(file)]
    ct_files = get_instance_files(instance_files)
    if len(ct_files) == 0:
        print('error: No valid CT instances found !!')
        exit(-1)
    zip_obj = ZipFile(sid + '.zip', 'w')
    for ct_file in ct_files:
        file_name_only = os.path.basename(ct_file)
        zip_obj.write(ct_file, file_name_only)
    zip_obj.close()


if __name__ == "__main__":
    main(sys.argv[1:])
