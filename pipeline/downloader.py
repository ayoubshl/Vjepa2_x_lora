"""
Downloads and extracts EK-100 RGB frames for one participant.

Wraps the epic_downloader.py script. Downloads .tar files
one at a time, extracts each immediately, deletes the .tar
to minimize disk usage.
"""

import os
import sys
import tarfile
import shutil
import threading


class FrameDownloader:
    """
    Downloads RGB frames for a single participant using
    the EpicDownloader class, then extracts .tar files.

    Args:
        downloader_dir:  path to download/ folder containing
                         epic_downloader.py and data/ folder
        frames_dir:      path to frames/ folder where extracted
                         JPEGs will live
        output_base:     temp download location for .tar files
    """

    def __init__(self, downloader_dir, frames_dir):
        self.downloader_dir = downloader_dir
        self.frames_dir = frames_dir
        self.output_base = os.path.join(downloader_dir, 'temp')

        # Add downloader dir to path so we can import it
        if downloader_dir not in sys.path:
            sys.path.insert(0, downloader_dir)

    def download_participant(self, participant_id):
        """
        Downloads and extracts all RGB frames for one participant.
        Blocking call — returns when fully done.

        Args:
            participant_id: e.g. 'P01'

        Returns:
            True if successful, False if failed
        """
        print(f"\n{'='*50}")
        print(f"Downloading frames for {participant_id}...")
        print(f"{'='*50}")

        try:
            # Import the downloader
            from epic_downloader import EpicDownloader

            downloader = EpicDownloader(
                base_output=self.output_base,
                splits_path_epic_55=os.path.join(
                    self.downloader_dir, 'data', 'epic_55_splits.csv'),
                splits_path_epic_100=os.path.join(
                    self.downloader_dir, 'data', 'epic_100_splits.csv'),
                md5_path=os.path.join(
                    self.downloader_dir, 'data', 'md5.csv'),
                errata_path=os.path.join(
                    self.downloader_dir, 'data', 'errata.csv'),
            )

            # Download RGB frames only
            downloader.download(
                what=('rgb_frames',),
                participants=[participant_id],
                splits='all',
                challenges='all',
            )

            # Extract all .tar files for this participant
            self._extract_and_cleanup(participant_id)

            print(f"{participant_id} download + extraction complete")
            return True

        except Exception as e:
            print(f"ERROR downloading {participant_id}: {e}")
            return False

    def _extract_and_cleanup(self, participant_id):
        """
        Finds all .tar files downloaded for a participant,
        extracts them to frames_dir, deletes the .tar immediately.
        """
        # The downloader saves to: output_base/EPIC-KITCHENS/PXX/rgb_frames/
        download_path = os.path.join(
            self.output_base, 'EPIC-KITCHENS',
            participant_id, 'rgb_frames'
        )

        if not os.path.exists(download_path):
            print(f"Warning: no download folder found at {download_path}")
            return

        tar_files = sorted([
            f for f in os.listdir(download_path)
            if f.endswith('.tar')
        ])

        print(f"Extracting {len(tar_files)} tar files for {participant_id}...")

        participant_frames = os.path.join(self.frames_dir, participant_id)
        os.makedirs(participant_frames, exist_ok=True)

        for tar_name in tar_files:
            tar_path = os.path.join(download_path, tar_name)

            try:
                # Extract
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=participant_frames)

                # Delete tar immediately
                os.remove(tar_path)
                video_id = tar_name.replace('.tar', '')
                print(f"  extracted + deleted: {tar_name}")

            except Exception as e:
                print(f"  ERROR extracting {tar_name}: {e}")

        # Clean up the temp download folder
        epic_kitchens_dir = os.path.join(self.output_base, 'EPIC-KITCHENS')
        if os.path.exists(epic_kitchens_dir):
            shutil.rmtree(epic_kitchens_dir)

    def download_in_background(self, participant_id):
        """
        Starts download in a background thread.
        Returns the thread object — call thread.join() to wait.

        Args:
            participant_id: e.g. 'P01'

        Returns:
            threading.Thread (already started)
        """
        thread = threading.Thread(
            target=self.download_participant,
            args=(participant_id,),
            daemon=True
        )
        thread.start()
        print(f"Background download started: {participant_id}")
        return thread

    def delete_participant(self, participant_id):
        """
        Deletes all extracted frames for a participant.

        Args:
            participant_id: e.g. 'P01'
        """
        participant_dir = os.path.join(self.frames_dir, participant_id)

        if os.path.exists(participant_dir):
            shutil.rmtree(participant_dir)
            print(f"Deleted frames: {participant_id}")
        else:
            print(f"Nothing to delete: {participant_id}")

    def participant_ready(self, participant_id):
        """
        Checks if a participant's frames are on disk.

        Args:
            participant_id: e.g. 'P01'

        Returns:
            True if the participant folder exists and has content
        """
        participant_dir = os.path.join(self.frames_dir, participant_id)

        if not os.path.exists(participant_dir):
            return False

        # Check it has at least one video subfolder
        contents = os.listdir(participant_dir)
        return len(contents) > 0