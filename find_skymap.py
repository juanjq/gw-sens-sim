import os, json
import numpy as np
from minio import Minio
from astropy.io import fits
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from ligo.gracedb.rest import GraceDb

def search_gwtc(event_name, data_folder):
    """Search the ODA Hub GWTC Catalog."""
    credentials_env = os.environ.get("S3_CREDENTIALS")
    if credentials_env:
        credentials = json.loads(credentials_env)
    else:
        credentials = {"endpoint": "minio-dev.odahub.fr", "secure": True}

    try:
        client = Minio(
            endpoint=credentials["endpoint"],
            secure=credentials.get("secure", True),
            access_key=credentials.get("access_key"),
            secret_key=credentials.get("secret_key"),
        )
        
        # Clean the name to match filenames by date
        search_term = event_name.replace("GW", "").replace("S", "").replace("G", "")
        objects = client.list_objects(bucket_name="gwtc", recursive=True)
        
        matches = []
        for obj in objects:
            if search_term in obj.object_name and ".fits" in obj.object_name:
                matches.append(obj)
        
        if matches:
            # Sort to find the most detailed filename (last update)
            matches.sort(key=lambda x: len(x.object_name), reverse=True)
            best_match = matches[0]
            local_path = os.path.join(data_folder, best_match.object_name.split("/")[-1])
            
            client.fget_object("gwtc", best_match.object_name, local_path)
            return local_path
        else:
            print(f" - No file found in GWTC for {event_name}...\n")
            
    except Exception as e:
        print(f" - GWTC search failed: {e}")
    return None



def search_gracedb(event_name, data_folder):
    """Search in GraceDB API."""
    try:
        client = GraceDb("https://gracedb.ligo.org/api/")
        files_dict = client.files(event_name).json()
        
        # Search priority
        targets = [
            "bayestar.multiorder.fits", 
            "lalinference.multiorder.fits", "bayestar.fits.gz", "skymap.fits.gz"
        ]
        found_file = next((t for t in targets if t in files_dict), None)

        if not found_file:
            # Fallback: take the alphabetically last .fits file
            fits_files = [f for f in files_dict.keys() if ".fits" in f]
            if fits_files:
                found_file = sorted(fits_files)[-1]

        if found_file:
            file_url = files_dict[found_file]
            local_path = os.path.join(data_folder, f"{event_name}_{found_file}")
            
            response = client.get_file(file_url)
            with open(local_path, "wb") as f:
                f.write(response.read())
            return local_path
        else:
            print(f" - No file found in GraceDB for {event_name}...\n")
            
    except Exception as e:
        print(f" - GraceDB search failed: {e}")
    return None

def get_skymap(event_name, data_folder="data"):
    """
    Tries GWTC first, then GraceDB.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Try GWTC
    print(f"Searching GWTC Catalog for {event_name}")
    path = search_gwtc(event_name, data_folder)
    if path:
        print(f" - Found in GWTC: {path}\n")
        return path

    # Try GraceDB
    print(f"Searching GraceDB for {event_name} ---")
    path = search_gracedb(event_name, data_folder)
    if path:
        print(f" - Found in GraceDB: {path}\n")
        return path

    print(f"Event '{event_name}' not found in any database.")
    return None

def plot_gw(ra_grid, dec_grid, data_ligo_2d, logscale=False):
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111, projection="mollweide")
    
    cmesh = ax.pcolormesh(
        ra_grid, dec_grid, np.flip(data_ligo_2d, axis=1), cmap="cylon", 
        norm=LogNorm() if logscale else None
    )
    
    plt.grid(color="lightgray")
    cbar = plt.colorbar(
        cmesh, ax=ax, orientation="horizontal", 
        pad=0.1, fraction=0.05, aspect=20, label="Probability"
    )
    ax.spines["geo"].set_linewidth(1.2); cbar.outline.set_linewidth(1.2)
    ax.set_xticklabels([])
    plt.show()

def get_event_time(file_path):
    """Extracts the GPS time (T0) from the FITS header."""
    with fits.open(file_path) as hdul:
        header = hdul[1].header  # Usually the data is in the first extension
        
        # Common keywords for the event time
        gps_time = header.get('OBJECT_GPS',  # Specific trigger time
                   header.get('GPSCREAT',    # Often used as a fallback
                   header.get('DATE-OBS')))  # Sometimes stored as UTC string
        
        # If it's a MOC file, check the 'DIST' or 'PROB' extension headers
        instruments = header.get('INSTRUME', 'Unknown')
        
        return datetime.fromisoformat(gps_time), instruments