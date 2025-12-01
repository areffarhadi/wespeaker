# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import pickle
import os

import lmdb
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('in_scp_file', help='input scp file')
    parser.add_argument('out_lmdb', help='output lmdb')
    args = parser.parse_args()
    return args


def get_total_wav_size(scp_file):
    total_size = 0
    with open(scp_file, 'r', encoding='utf8') as fin:
        for line in fin:
            wav_path = line.strip().split()[1]
            if os.path.exists(wav_path):
                total_size += os.path.getsize(wav_path)
    return total_size


def main():
    args = get_args()
    
    # Calculate total size of wav files
    total_wav_size = get_total_wav_size(args.in_scp_file)
    print(f"Total size of wav files: {total_wav_size / (1024*1024*1024):.2f} GB")
    
    # Try with 1GB first
    map_size = int(math.pow(1024, 3))  # 1GB in bytes
    
    while True:
        try:
            print(f"Trying with map_size: {map_size / (1024*1024*1024):.2f} GB")
            db = lmdb.open(args.out_lmdb, map_size=map_size)
            txn = db.begin(write=True)
            keys = []
            
            with open(args.in_scp_file, 'r', encoding='utf8') as fin:
                lines = fin.readlines()
                for i, line in enumerate(tqdm(lines)):
                    arr = line.strip().split()
                    assert len(arr) == 2
                    key, wav = arr[0], arr[1]
                    keys.append(key)
                    with open(wav, 'rb') as fin:
                        data = fin.read()
                    try:
                        txn.put(key.encode(), data)
                    except lmdb.MapFullError:
                        # If we hit MapFullError, close current db and retry with double size
                        db.close()
                        map_size *= 2
                        print(f"\nIncreasing map_size to: {map_size / (1024*1024*1024):.2f} GB")
                        raise
                    
                    # Write flush to disk
                    if i % 100 == 0:
                        txn.commit()
                        txn = db.begin(write=True)
            
            # If we get here, we succeeded
            txn.commit()
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', pickle.dumps(keys))
            db.sync()
            db.close()
            print(f"\nSuccessfully created LMDB with {map_size / (1024*1024*1024):.2f} GB")
            break
            
        except lmdb.MapFullError:
            continue
        except lmdb.MemoryError:
            # If we hit MemoryError, try with half the size
            map_size = int(map_size / 2)
            print(f"\nMemoryError: Reducing map_size to: {map_size / (1024*1024*1024):.2f} GB")
            if map_size < 1024*1024:  # If less than 1MB, give up
                print("Error: Cannot allocate even minimal memory!")
                break
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            break


if __name__ == '__main__':
    main()

