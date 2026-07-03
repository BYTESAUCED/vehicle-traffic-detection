# Prompts Used (Codex)

This file records the plain English prompts used to guide code changes.


## Prompt 1: Change background and draw graphs

Change the background of the Streamlit app to white and use a light colored
theme. Below the per image results, draw two bar graphs. The first bar graph
shows the density distribution, that is how many images fall into each label of
low, medium, high, and unclear. The second bar graph shows the total number of
detected vehicles per class across all the images in the current batch.


## Prompt 2: Format code

Go through the pyguide file and format the code so it follows that style guide.
Fix the comments and docstrings to match the guide. Only refactor these things
and nothing else: the Streamlit caching and the fragment usage. Do not change
any other logic or behavior.



## Prompt 3: Create the dataset download structure

Edit the download script so it automatically creates the train, valid, and test
folder structure that the model expects. Create the test split as a copy of the
validation split.



## Prompt 4: Auto save the CSV and set default confidence

The saved CSV should reflect the current detection confidence level. Save it
automatically under a folder and show a notification that it was saved. Set the
default detection confidence to 0.25.


## Prompt 5: Adaptive per image confidence

Add adaptive per image confidence filtering. Run inference once at a low base
threshold of 0.10 to catch all candidate boxes. For each image find the highest
confidence score, then keep only boxes whose confidence is at least the larger
of 0.10 and the max confidence minus 0.15. This scales the acceptance bar down
for images where the best detection is naturally low and up for cleaner images.


## Prompt 6: Otsu confidence clustering and app improvements

Add a better confidence method based on Otsu clustering. For each image, take
the detection confidences from the low base threshold pass, build a histogram,
and use Otsu's method to find the threshold that maximizes the between class
variance between the noise cluster and the signal cluster. Apply a safety floor
of 0.10 so the threshold never goes below the base level.
Also keep the fixed slider confidence and the adaptive method as separate
options that save to separate files. When a row in the summary table is clicked,
open that image in the viewer. Allow renaming the saved CSV file in a small text
box.


## Prompt 7: Documentation

Update the README basic text. Cover the dataset subset used,
how frames were selected, the model setup and training, how to run inference,
how to start the Streamlit app, how the density label is computed, and known
limitations also Docstring that might be useful in the readme file. Also write a model training document that describes the process and
leaves placeholders for GitHub to link images of results and annotations.


## Prompt 8: Confidence score clustering pipeline (Otsu)

Implement a better per-image confidence pipeline. Run the detection model once
per image at confidence 0.05, which is the only forward pass; every other step
is post-processing on those outputs.

For each image, take the confidence scores of the detections and run Otsu's
method on them. Otsu finds the natural gap between the signal cluster (real
vehicles) and the noise cluster (false positives) by maximizing the between
class variance of the confidence distribution. This adapts to each image's own
distribution rather than using a global margin.

If the image has fewer than 5 detections, Otsu cannot find a meaningful split,
so fall back to keeping the top 35 percent strongest detections by confidence
instead. Either way, enforce a hard floor of 0.10 so the filter never accepts a
box the model is very uncertain about.

Use density cutoffs derived from the actual histogram rather than generic
values. The low to medium boundary is 5. The medium to high boundary is the
natural trough at 11, which is better than Q3 = 14 because 14 inflates the
medium class. So 1 to 5 is low, 6 to 11 is medium, and 12 or more is high, with
0 detections being unclear.

Cap the displayed count at roughly the 99th percentile so a few very high
outlier images do not distort the UI or aggregate stats. The label is still
computed from the true count; only the displayed value is capped.

Correction: do not filter to a car class only. Count all vehicle classes when
computing the vehicle count and the density label.
