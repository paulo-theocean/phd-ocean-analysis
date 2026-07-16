from pathlib import Path

import xarray as xr

DATA_FILE = Path("/home/paulo/Documents/data/atmospheric/JRA3Q/q2_JRA3Q_y1958.nc")

def main():
    ds = xr.open_dataset(DATA_FILE)

    print("\nDATASET")
    print(ds)
    print("\nDATA VARIABLES")
    for name in ds.data_vars:
        print(f"- {name}: dims={ds[name].dims}, shape={ds[name].shape}")

    print("\nCOORDINATES")
    for name in ds.coords:
        print(f"- {name}: dims={ds[name].dims}, shape={ds[name].shape}")

    print("\nDIMENSIONS")
    print(ds.dims)


if __name__ == "__main__":
    main()