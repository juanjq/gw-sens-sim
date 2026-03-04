import numpy as np
import healpy as hp
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord


def IndexToDeclRa(index, nside):
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)

def DeclRaToIndex(decl, ra, nside):
    return hp.pixelfunc.ang2pix(
        nside, np.radians(-decl + 90.),
        np.radians(360. - ra)
    )
    
def healpix2map(healpix_data, ra_bins, dec_bins):
    
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)

    # Convert the latitude and longitude to theta and phi
    theta, phi = np.radians(90 - dec_grid), np.radians(ra_grid)
    
    nside = hp.npix2nside(len(healpix_data)) # nside of the grid

    # Convert theta, phi to HEALPix indices and create a 2D map using the HEALPix data
    hp_indices = hp.ang2pix(nside, theta, phi)

    return (healpix_data[hp_indices])

def get_hp_map_thresholds(healpix_data, threshold_percent=[0.9, 0.68]):
    
    # We sort the tresholds itself in descending order
    threshold_percent = np.sort(threshold_percent)[::-1]
    
    # Sort in descending order and normalize
    sorted_data = np.sort(healpix_data)[::-1] / np.sum(healpix_data)
    cumulative_sum = np.cumsum(sorted_data)

    # Find the values corresponding to the thresholds
    indexes_map = [np.searchsorted(cumulative_sum, t) for t in threshold_percent]
    # Then we find the thresholds
    threshold_maps = [sorted_data[min(index, len(sorted_data) - 1)] for index in indexes_map]
    
    return threshold_maps

def get_2d_map_hotspot(map_data_2d, ra_bins, dec_bins):
    
    # Computing coordinate of maximum probability
    max_prob_index = np.unravel_index(np.argmax(map_data_2d), map_data_2d.shape)
    
    max_prob_ra, max_prob_dec = ra_bins[max_prob_index[1]], dec_bins[max_prob_index[0]]
    max_prob_coords = SkyCoord(ra=max_prob_ra, dec=max_prob_dec, unit=u.deg, frame="icrs")
    return max_prob_coords

def fix_gadf_header_lst(hdr, obs_id):
    """Safely injects timestamps and forces types only if keywords exist."""
    
    # Force numerical types safely for the most important float elements
    for k in ["RA_PNT", "DEC_PNT", "ALT_PNT", "AZ_PNT", "TSTART", "TSTOP"]:
        if k in hdr:
            try:
                hdr[k] = float(hdr[k])
            except (ValueError, TypeError):
                pass

    # Calculate timestamps ONLY if TSTART exists in THIS header
    tstart_val = hdr.get("TSTART")
    if tstart_val is not None:
        mjd_ref = hdr.get("MJDREFI", 51544) + hdr.get("MJDREFF", 0.0)
        # Convert seconds-of-day to iso strings
        t_start_obj = Time(mjd_ref + (tstart_val / (3600 * 24)), format="mjd", scale="utc")
        hdr.set("TIME-OBS", t_start_obj.iso.split(" ")[1])
        hdr.set("DATE-OBS", t_start_obj.iso.split(" ")[0])
        
        tstop_val = hdr.get("TSTOP")
        if tstop_val is not None:
            t_stop_obj = Time(mjd_ref + (tstop_val / (3600 * 24)), format="mjd", scale="utc")
            hdr.set("TIME-END", t_stop_obj.iso.split(" ")[1])
            hdr.set("DATE-END", t_stop_obj.iso.split(" ")[0])

    # Setting the proper identity keywords
    hdr.update({
        "TELESCOP": "CTA-N", "INSTRUME": "LST-1", "OBS_ID": int(obs_id), "HDUCLASS": "GADF"
    })

def create_dl3_file_lst(pointing_files, output_path, obs_id):
    """Merges segments, adds POINTING HDU, and applies all fixes."""
    # Stack Tables
    merged_ev = vstack([Table.read(f, hdu="EVENTS") for f in pointing_files])
    merged_gti = vstack([Table.read(f, hdu="GTI") for f in pointing_files])
    
    with fits.open(pointing_files[0]) as h_tmp:
        # Build HDU List
        new_hdul = fits.HDUList([
            fits.PrimaryHDU(header=h_tmp["EVENTS"].header.copy()),
            fits.BinTableHDU(data=merged_ev, header=h_tmp["EVENTS"].header.copy(), name="EVENTS"),
            fits.BinTableHDU(data=merged_gti, header=h_tmp["GTI"].header.copy(), name="GTI")
        ])
        # Add IRFs
        for hdu_name in ["EFFECTIVE AREA", "ENERGY DISPERSION", "PSF", "BKG"]:
            if hdu_name in h_tmp: new_hdul.append(h_tmp[hdu_name].copy())

    # Create POINTING HDU
    ev_hdr = new_hdul["EVENTS"].header
    cols = fits.ColDefs([
        fits.Column(name="TIME", format="D", unit="s", array=np.array([ev_hdr["TSTART"]])),
        fits.Column(name="RA_PNT", format="D", unit="deg", array=np.array([float(ev_hdr["RA_PNT"])])),
        fits.Column(name="DEC_PNT", format="D", unit="deg", array=np.array([float(ev_hdr["DEC_PNT"])])),
        fits.Column(name="ALT_PNT", format="D", unit="deg", array=np.array([float(ev_hdr["ALT_PNT"])])),
        fits.Column(name="AZ_PNT", format="D", unit="deg", array=np.array([float(ev_hdr["AZ_PNT"])])),
    ])
    pnt_hdu = fits.BinTableHDU.from_columns(cols, name="POINTING")
    pnt_hdu.header.update({"HDUCLASS": "GADF", "HDUCLAS1": "POINTING", "OBS_ID": obs_id})
    new_hdul.append(pnt_hdu)

    # Final Header Fixes
    for hdu in new_hdul:
        if hdu.name in ["EVENTS", "GTI", "EFFECTIVE AREA", "ENERGY DISPERSION", "PRIMARY"]:
            fix_gadf_header_lst(hdu.header, obs_id)
            
    new_hdul.writeto(output_path, overwrite=True)
