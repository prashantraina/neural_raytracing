# Scripts

These are various scripts used for training our models. All scripts are intended to be run as
```sh
$python3 <script>
```
In order to configure the script for different models, there are variables defined at the top of
the script. Each one will automatically use a single GPU if it is available, otherwise they will
default to CPU but they were only tested on GPU.

In order to configure the scripts, you need to specify a path to where the dataset is. For
NeRF/NeRV, this is the folder which contains the camera positions. For DTU, it is the scan
folder.

## Data

The data for NeRF can be found
[here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), and the data
we use for DTU can be found from
[IDR](https://github.com/lioryariv/idr/blob/main/data/download_data.sh). In order to run on
their data, you'll have to manually specify the `DIR` variable in each script.
For [`colocate.py`](colocate.py) and [`nerfle.py`](nerfle.py), you will need to recreate the
dataset using Mitsuba, which can be done from the `mitsuba_scenes` directory in the root
directory.
I'm not entirely sure where NeRV's public dataset is, as at the time of release we got it by
emailing the author.

## Output & Results

Models should be output into the `models/` folder, and training images will be put into the
`outputs/` folder.

After training, most scripts will automatically run the test set. Some, specifically NeRV, have
a separate test set. In addition, visualization and editing must be done after training, and
there are additional scripts for playing around with those.

- Any script that ends with `vis` is used for visualization. These output some set of views if
  an integrator is provided, as well as BSDF maps and BSDF visualizations.
- Scripts for testing are prefixed with `test`. These will render test-set views of a trained
  model.
