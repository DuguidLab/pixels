pixels
======


Data Organisation
-----------------

New raw data should be moved into your ``raw`` folder, inside a folder whose
name starts with the format "YYMMDD_mouseID" for that recording session. These
files are automatically compressed for storage so data processing copies them
to the ``interim`` folder if they are not already present there, uncompressing
if required. The data is processed from ``interim`` and the results saved in
``processed``, where they can be accessed for analyses.


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
   │       └── YYMMDD_mouseID.tdms
   ├── processed
   │   └── YYMMDD_mouseID_extrainfo
   │       ├── TBD
   │       └── TBD
   └── raw
       └── YYMMDD_mouseID_extrainfo
           ├── YYMMDD_mouseID_gN_t0.imec0.ap.bin.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.ap.meta.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.lf.bin.tar.gz
           ├── YYMMDD_mouseID_gN_t0.imec0.lf.meta.tar.gz
           ├── USB_Camera.tdms.tar.gz
           ├── USB_Camerameta.tdms.tar.gz
           └── YYMMDD_mouseID.tdms.tar.gz


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


Resources
---------

* Cortex Lab neuropixels wiki: https://github.com/cortex-lab/neuropixels/wiki
* Cortex Lab phy: https://github.com/cortex-lab/phy
* SpikeInterface: https://github.com/SpikeInterface/spiketoolkit
* Kilosort2: https://github.com/MouseLand/Kilosort2
* Allen Institute pixels spike sorting: https://github.com/alleninstitute/ecephys_spike_sorting
* DeepInsight: https://github.com/CYHSM/DeepInsight
