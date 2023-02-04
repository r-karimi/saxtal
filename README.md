# saxtal

NOTE: In order to install the saxtal env on another computer, run:

conda env create -f environment.yml


TODO list:

3. Process PBS-apo data
4. Compare motion estimates

Minor:
- Implement pyFFTW use as a flag, test overall script execution time
- Eliminate micrograph_filename_out parameter in mask_movies by replacing call to mask_image() with code inside mask_image()
