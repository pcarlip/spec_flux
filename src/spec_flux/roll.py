import xarray as xr


def roll_da(ds: xr.DataArray, args: dict) -> xr.DataArray:
    return ds.roll(args)


def shift_da(ds: xr.DataArray, args: dict) -> xr.DataArray:
    return ds.shift(args)
