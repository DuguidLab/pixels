pixels
======


Data Organisation
-----------------

New raw data should be moved into your ``raw`` folder, inside a folder whose
name starts with the format "YYMMDD_mouseID" for that recording session. These
files are automatically compressed for storage so data processing copies them
to the ``interim`` folder if they are not already present there, uncompressing
if required. The data is processed from ``interim`` and the results saved in
``processed``, where they can be accessed for analyses. Note that everything in
the ``interim`` folder can be regenerated, so the entire interim folder can be
deleted without losing anything important.


::

   data
   ├── interim
   │   └── YYMMDD_mouseID_extrainfo
   │       ├── YYMMDD_mouseID_gN_t0.imec0.ap.bin
   │       ├── YYMMDD_mouseID_gN_t0.imec0.ap.meta
   │       ├── YYMMDD_mouseID_gN_t0.imec0.lf.bin
   │       ├── YYMMDD_mouseID_gN_t0.imec0.lf.meta
   │       ├── USB_Camera.tdms
   │       ├── USB_Camerameta.tdms
   │       ├── NeuropixelBehaviour(0).tdms
   │       └── cache
   │           └── * see below
   ├── processed
   │   └── YYMMDD_mouseID_extrainfo
   │       ├── YYMMDD_mouseID_gN_t0.imec0.ap_processed.h5
   │       ├── YYMMDD_mouseID_gN_t0.imec0.lf_processed.h5
   │       ├── NeuropixelBehaviour(0)_processed.h5
   │       ├── action_labels_0.npy
   │       ├── sync_0.png
   │       └── lag.json
   └── raw
       └── YYMMDD_mouseID_extrainfo
           ├── YYMMDD_mouseID_gN_t0.imec0.ap.bin.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.ap.meta.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.lf.bin.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.lf.meta.tar.gz
           ├── USB_Camera.tdms.tar.gz
           ├── USB_Camerameta.tdms.tar.gz
           ├── NeuropixelBehaviour(0).tdms.tar.gz
           └── extra
               └── ** see below

\* Some basic analyses will save the result of their calculations into this
cache folder.

** Any files collected on the recording day that should be ignored by the
pipeline should be put inside a subfolder(s) within the session's folder. The
name of the folder(s) are not important.


Pipeline
--------

::

   [raw] Compressed raw data
   
   [interim]    raw spike data             behavioural tdms          LFP data          camera
                      ┃                           ┃                     ┃                 ┃
                      v                           ┃                     ┃                 ┃
                 spike sorting                    v                     v                 v
              downsample to 1kHz      create action labels 1kHz      resample            DLC +
              phy manual curation                 ┃                     ┃              resample
                      ┃                           ┃                     ┃                 ┃
                      v                           v                     v                 v
   [processed]   spike data                 action labels           1kHz LFP      1kHz DLC coordinates


The first processing step run will align the probe recordings and the
behavioural data and save the ``lag`` - that is, the number of points of
overhang at the start and end of the behavioural data - into ``lag.json`` in
the processed folder. The sync step will also save a figure to ``sync_0.png``,
which should be checked to visually confirm that the syncing went well.


Conda
-----

These commands can be used to create a conda environment with all libraries
used by the pipeline:

.. code-block:: python

    conda create -n pixels numpy pandas nptdms scipy matplotlib opencv -c conda-forge
    conda activate pixels
    pip install ffmpeg-python spikeinterface probeinterface

This does not include deeplabcut and it's dependencies - see the deeplabcut
docs for how to install.


Resources
---------

* Cortex Lab neuropixels wiki: https://github.com/cortex-lab/neuropixels/wiki
* Cortex Lab phy: https://github.com/cortex-lab/phy
* SpikeInterface: https://github.com/SpikeInterface/spiketoolkit
* Kilosort2: https://github.com/MouseLand/Kilosort2
* Allen Institute pixels spike sorting: https://github.com/alleninstitute/ecephys_spike_sorting
* DeepInsight: https://github.com/CYHSM/DeepInsight
