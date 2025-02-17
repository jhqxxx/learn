'''
Author: jhq
Date: 2025-02-14 23:06:58
LastEditTime: 2025-02-15 00:46:06
Description: 
'''
from imagededup.methods import PHash
from multiprocessing import freeze_support
import os

def main():
    path = r"D:\data\detection\fire\all_image\train"
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=path)
    duplicates = phasher.find_duplicates(encoding_map=encodings)    
    # sorted_duplicates = sorted(duplicates.keys(), key=lambda k: int(os.path.splitext(k)[0]))
    for duplicate_key in duplicates.keys():
        file_key = os.path.join(path, duplicate_key)
        if os.path.isfile(file_key):
            for f_dup in duplicates[duplicate_key]:
                file_dup = os.path.join(path, f_dup)
                if os.path.isfile(file_dup):
                    print(f'删除重复图片: {file_dup}')
                    os.remove(file_dup)

if __name__ == '__main__':  
    freeze_support()  
    main()