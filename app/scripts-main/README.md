# msc-thesis-scripts
Python scripts to reproduce results from my MSc thesis.

## STA/LTA trigger
An ensemble STA/LTA trigger is implemented in `stalta_trigger.py`. All
parameters can be set in `stalta_params.py`. This results in a "raw"
catalog that needs to be further processed. This can be achieved with the
script `merge_events.py` which takes a raw catalog as input and merges
all entries that belong to the same event, i.e. cleaning the raw catalog from
duplicates. All events in a catalog can be plotted by passing the cleaned
catalog to the script `plot_events_from_catalogue.py`. All catalogs are saved
in the `output/` folder.

## Extracting features for clustering
For my MSc thesis I extracted two different types of features related to
velocities and signal coherence. There are two corresponding scripts:
`extract_features_velo.py` and `extract_features_covmat.py`. The extracted
features are stored in the `output/` folder.

## Information on accessing AWS data
### Mount EBS
Following are the steps I used to mount an EBS volume. Before that, the volume
needs to be created and attached to the EC2 instance. This can be done via the
AWS web interface. Subsequently, it should show up when running `lsblk`.
Usually there is not yet a file system on the new EBS volume. This can be
checked by running `file -s /dev/NAME`, where NAME is the name of the EBS
volume as shown when running `lsblk`. I usually use the ext4 file system,
e.g. `mkfs -t ext4 /dev/NAME`. The volume can then be mounted to the
filesystem. I usually run:
```
mkdir /dataStorage
mount /dev/xvdf /dataStorage
df -hT
```
The last command checks if the EBS volume was successfully mounted.

### Download data from S3 to EBS
For this, the awscli must be installed on the instance.
```
aws s3 cp bucket_name/ target_folder/ --options
```
For example:
* bucket_name = s3://rhoneglacier-eth/new/20200707/
* options: --recursive; --exclude “*”; --include "idas2_UTC_20200707_074*"

Of course you can use other options like the boto3 you used in your python scripts.

### NOTES:
* You need root access to read and write to the EBS volume
* You need the AWS key to access S3
* For the start: It would probably be really useful to have a bash script that:
  * Creates EBS volume (Let's say G2 with 500 GB)
  * Attaches EBS volume to instance
  * Starts the instance
  * Makes filesystem on EBS volume and mount
  * Downloads one day of data to the EBS volume
