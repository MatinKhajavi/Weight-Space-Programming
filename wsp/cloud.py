"""Modal on the profile named in .env, so nothing runs on whichever Modal profile happens to be active."""
import os
import sys

from wsp.runner import read_dotenv


def load_modal():
    """Import modal on the profile named by MODAL_PROFILE in .env. Modal reads MODAL_PROFILE when it is first imported,
    so an earlier import with another profile stops the run."""
    profile = read_dotenv().get("MODAL_PROFILE")
    if not profile:
        sys.exit("No MODAL_PROFILE in .env; refusing to run on whatever Modal profile is active.")
    if "modal" in sys.modules and os.environ.get("MODAL_PROFILE") != profile:
        sys.exit(f"modal was imported before MODAL_PROFILE was set to {profile!r}; refusing to run.")
    os.environ["MODAL_PROFILE"] = profile
    import modal
    return modal
