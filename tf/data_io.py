import os
import struct
import numpy as np
import biotite.sequence.io.fasta as fasta


def get_sequences(in_filename="sequence_Coloeus_monedula.fasta",
                  out_filename="test_sequence_Coloeus_monedula.txt"):
    fasta_file = fasta.FastaFile.read(in_filename)
    items = fasta_file.items()
    
    with open(out_filename, "w") as f:
        for _, seq in items.__iter__():
            f.write(seq+"\n")
    
    
def get_mnist(path="/media/valdecar/storage3/data/mnist",
              filename="t10k-images.idx3-ubyte"):

    '''convert mnist (original) binary to arrays
    REF: http://monkeythinkmonkeycode.com/mnist_decoding/
    '''

    with open(os.path.join(path, filename), 'rb') as f:
        mw_32bit = f.read(4)
        n_numbers_32bit = f.read(4)  # number of images
        n_rows_32bit = f.read(4)  # number of rows of each image
        n_columns_32bit = f.read(4)   # number of columns of each image
        n_numbers = struct.unpack('>i', n_numbers_32bit)[0]
        n_rows = struct.unpack('>i', n_rows_32bit)[0]
        n_columns = struct.unpack('>i', n_columns_32bit)[0]
        
        ds_images = []
        for i in range(n_numbers):
            img = []
            for r in range(n_rows):
                row = []
                for l in range(n_columns):
                    byte = f.read(1)
                    pixel = struct.unpack('>B', byte)[0]
                    row.append(pixel)
                img.append(row)
            ds_images.append(np.array(img))
    return(ds_images)


if __name__ == "__main__":
    print("get_sequences():")
    get_sequences()
    print("\ndone")
