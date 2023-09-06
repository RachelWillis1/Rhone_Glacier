#!/bin/sh
startall=$(date +%s)

echo ----
echo Preparing input/output folders
# prepare folders:
mkdir -p input input/DAS_data output output/event_pngs output/extr_features
echo - done

echo ----
echo Preparing input data filelist
# check folders and files in S3:
aws s3 ls --summarize --human-readable s3://rhoneglacier-eth/new/ > ./output/folderlist.csv
aws s3 ls --summarize --human-readable --recursive s3://rhoneglacier-eth/new/ > ./output/filelist.csv

aws s3 cp s3://csm-luna/RG-output/codes/S3foldefileliststxt.py ./output/
python3 ./output/S3foldefileliststxt.py
rm -I output/S3foldefileliststxt.py
echo - done

echo ----
echo Selecting input data filelist
# USER: time - from to (from mintime to maxtime)
echo Define time range for input data:
echo minimum time: 20200706_071820.361
echo maximum time: 20200804_001520.542
read -p 'Input your start time: ' starttime
read -p 'Input your end time: ' endtime
echo Your selected time range is: $starttime - $endtime

# automatic py script creating filelist in the spec interval
export starttime
export endtime
aws s3 cp s3://csm-luna/RG-output/codes/S3filelist_selectiontxt.py ./output/
python3 ./output/S3filelist_selectiontxt.py
rm -I output/S3filelist_selectiontxt.py
echo - done

# timestamp for output
# $starttime-$endtime

# load data from S3 to EBS:
# see selected files from filelist to be loaded

echo -----
echo START Processing of selected files
# 5120MB = 5GB # USER-specified or automatically check EBS
MINAVAIL="5000"

LEN_FILELIST=$(cat ./output/S3selectedfilelist.txt |wc -l)
I_FILE="0"

for path in $(cat ./output/S3selectedfilelist.txt);
do
        #cp data
        aws s3 cp s3://rhoneglacier-eth/$path ./input/DAS_data/
        I_FILE=$((I_FILE +1))

        # check the available space:
        AVAIL=$(df -m --output=avail / | grep -v Avail)

     if (( $(echo "$AVAIL < $MINAVAIL" |bc) )); then
            echo "/input/DAS_data is full and ready for processing"
            # run docker image, create container and process data:
            # docker run --rm --platform linux/amd64 --mount src="$(pwd)",target=/app/sharedfolder,type=bind 974367744326.dkr.ecr.us-east-2.amazonaws.com/rgtestxall:10_cpu_count
            docker run --rm --platform linux/amd64 --mount src="$(pwd)",target=/app/sharedfolder,type=bind 974367744326.dkr.ecr.us-east-2.amazonaws.com/rgtestxall:10_cpu_count_noplot
            # temp_time_stamp for outputfiles - rename outputfiles
            mv output/catalogue_raw_temp_processed.csv output/"catalogue_FS_raw_processed-$starttime-$endtime-ifile$I_FILE.csv"
            rm output/catalogue_raw_temp.csv
            mv output/selectedfilelist.txt output/"selectedfilelist-$starttime-$endtime-ifile$I_FILE.txt"
            mv output/extr_features/velo_temp.npy output/extr_features/"velo_temp-$starttime-$endtime-ifile$I_FILE.npy"
            mv output/extr_features/covmat_temp.npy output/extr_features/"covmat_temp-$starttime-$endtime-ifile$I_FILE.npy"

            # save results to S3:
            aws s3 sync ./output/ s3://csm-luna/RG-output/

            # delete data from /input/DAS_data:
            rm -f input/DAS_data/idas*

        elif (( $(echo "$I_FILE == $LEN_FILELIST" |bc) )); then
            echo "last file in the list loaded!"
            # run docker image, create container and process data:
            #docker run --rm --platform linux/amd64 --mount src="$(pwd)",target=/app/sharedfolder,type=bind 974367744326.dkr.ecr.us-east-2.amazonaws.com/rgtestxall:10_cpu_count
            docker run --rm --platform linux/amd64 --mount src="$(pwd)",target=/app/sharedfolder,type=bind 974367744326.dkr.ecr.us-east-2.amazonaws.com/rgtestxall:10_cpu_count_noplot
            # temp_time_stamp for outputfiles - rename outputfiles
            mv output/catalogue_raw_temp_processed.csv output/"catalogue_STALTA_raw_temp_processed-$starttime-$endtime-ifile$I_FILE.csv"
            rm output/catalogue_raw_temp.csv
            mv output/selectedfilelist.txt output/"selectedfilelist-$starttime-$endtime-ifile$I_FILE.txt"
            mv output/extr_features/velo_temp.npy output/extr_features/"velo_temp-$starttime-$endtime-ifile$I_FILE.npy"
            mv output/extr_features/covmat_temp.npy output/extr_features/"covmat_temp-$starttime-$endtime-ifile$I_FILE.npy"

            # save results to S3:
            aws s3 sync ./output/ s3://csm-luna/RG-output/

            # delete data from /input/DAS_data:
            rm -f input/DAS_data/idas*
      fi
done
echo - DONE - All files processed

# kmeans


endall=$(date +%s)
