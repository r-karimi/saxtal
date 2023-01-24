# saxtal

NOTE: In order to install the saxtal env on another computer, run:

conda env create -f environment.yml


TODO list:

0. Implement rule to only mask 2x1 or 1x2 clusters of pixels or larger
	- Investigate why some hotpixels aren't being filtered

1. Test applying masking to individual frames from patch average, then going to per-particle alignment

2. Restructure program to take command line arguments

3. Implement pyFFTW as a flag, test overall script execution time

Minor:
- Write helper function to plot indices of detected spots
