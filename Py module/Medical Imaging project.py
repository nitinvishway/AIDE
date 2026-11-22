import pydicom
from pydicom.data import get_testdata_file

path = get_testdata_file("JPEG2000_UNC.dcm")  # uncompressed version, easiest to work with
ds = pydicom.dcmread(path)

print(ds)  # full header dump first — see everything