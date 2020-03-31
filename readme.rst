pixels
======


Data Organisation
-----------------

 * Collected data:
    * SpikeGLX data (.ap.bin)
    * SpikeGLX metadata (.ap.meta)
    * local field potential data (.lf.bin)
    * local field potential metadata (.lf.meta)
    * behaviour channels (.tdms)
    * camera data (.tdms)
    * camera metadata (meta.tdms)
 * Organised by session:
    * folder name should start with: YYMMDD_mouseID


Data Processing
---------------

https://github.com/cortex-lab/neuropixels/wiki/Recommended_preprocessing


Resources
---------

 * Neuropixels github wiki: https://github.com/cortex-lab/neuropixels/wiki
 * SpikeInterface: https://github.com/SpikeInterface/spiketoolkit
 * Kilosort2: https://github.com/MouseLand/Kilosort2
 * Allen Institute pixels spike sorting: https://github.com/alleninstitute/ecephys_spike_sorting


Pipeline
--------

[raw] Raw data for compression.

[interim]    raw spike data                   behavioural tdms            LFP data    camera
                   ┃                                 ┃                       ┃           ┃
                   v                                 ┃                       ┃           ┃
              spike sorting                          v                       v           v
           downsample to 1kHz            create action labels 1kHz        resample      DLC +
           phy manual curation                       ┃                       ┃        resample
                   ┃                                 ┃                       ┃           ┃
                   v                                 v                       v           v
[processed]   spike data                       action labels             1kHz LFP   1kHz DLC coordinates
