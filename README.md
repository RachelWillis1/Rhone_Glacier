# Example of DAS data processing on AWS
- as a testing dataset, we use RhoneGlacier DAS dataset
- python based STA/LTA detection from continous data, extracting features (velocity, coherency), kmeans clustering and creating catalogue.

### Instructions:
(I == AWS UI; T == your terminal; W == web browser)

#### 0. Open a Terminal
The commands in the following sections will need to be run from your computer's terminal program <br>
To open the terminal (T) on your **local machine**: 
- Mac: Type 'terminal' into Spotlight
- Windows: Type 'powershell' in the system search bar
- Linux: Type 'terminal' in the system search bar

## Create docker image for this example:
- It is expected that you have Docker installed on your machine
- Download all the folders/files from this_github_repo/Docker/ to one of your empty workfolder and keep there structure:
   - app/
      - scripts-main/
      - pip_list_install.txt
      - sharedfolder/ # Note that you have to create this empty folder before you build the image. This folder will be a shared folder between container and host
   - Dockerfile

- T: yourworkfolder: docker build --platform linux/amd64 -t yourtag .
- T: yourworkfolder: docker images <br>
=> Tadaaaaa – now should see your yourtag image. In this case image with linux, python, python libraries/packages and folder with all the codes. 

## Run containerized example Devito code on AWS:
#### 00. Open Mines AWS
To be able to follow further instructions, you need to have access to Mines/Luna AWS and access to AWS services (EC2,ECR,S3) in us-east-2 region
- login to Mines AWS
- open management console (I)
- I: stay in us-east-2 region
- T: make sure that you have AWS CLI installed: T: aws --version
- I: get aws authentication - see "Command line or programmatic access" and copy "Set AWS environment variables"
- T: paste here = AWS environment variables are set

#### 1. Start EC2 and check if you have docker installed <br>
- I: Create/start EC2
   - recommended to create high vCPU instance 
   - Alternatively use prepared Frantisek-oneEC2-RG-c6a.32xL25 (with CSM-Luna_keypair.pem saved in S3 bucket: s3://csm-luna/CSM-Luna_keypair.pem)
- T: ssh to the EC2
   - see connect instructions for your runnning EC2 instance
   - example: ssh -i "CSM-Luna_keypair.pem" eec2-user@ec2-18-220-173-119.us-east-2.compute.amazonaws.com
- T: sudo su
- T: yum install docker –y
   - do only if needed to install docker. Check if docker is installed on EC2 with "T: docker --version"
- T: service docker start
- T: docker images
   - if you don't see devito image(s), pull devito image from AWS ECR

#### 2. Pull docker images from ECR (if needed):
- I: see ECR images
- I: see "view push commands" instructions for the selected ECR image
   - T: Retrieve an authentication token and authenticate your Docker client to your registry. Use the AWS CLI (in T).
   - T: docker pull rgtestxall:10_cpu_count (or rgtestxall:10_cpu_count_noplot)
- T: docker images
   - now you should see the image

#### 3. Create EBS volume and attach it to EC2
- I: Create EBS when starting new EC2 or create new EBS volume (I:EC2/Volumes) and attach it to existing EC2
   - create EBS (in the same zone as EC2!)
   - alternatively you can use prepared Frantisek-EBS4TBmaxIOPS-oneEC2
- T: lsblk
- T: sudo file -s /dev/name
   - system type
- T: sudo lsblk –f
   - all devs info
- T: sudo mkfs -t ext4 /dev/name 
   - create file system, alternatively mkfs -t xfs /dev/name
- T: sudo mkdir /yourdir
   - creates a mount point directory for the volume
- T: sudo mount /dev/name /yourdir 	
   - mounts the volume at the directory
- T: df –hT		
   - check
- T:df -hT /dev/name
   - check memory usage of xvdf
- T: aws ec2 delete-volume --volume-id vol-???
   - deletes volume ???

#### 4. Run runx.sh:
- the script contains all the commands
- I: get aws authentication - see "Command line or programmatic access" and copy "Set AWS environment variables"
- T: paste here = AWS environment variables are set
- T: from ec2-user to yourdir: cd ../../yourdir/
- T: yourdir: sudo su
- copy folder thisrepo/codes to s3://csm-luna/RG-output/
- get runx.sh file: T: yourdir: aws s3 cp s3://csm-luna/RG-output/codes/runx.sh ./
- T: yourdir: ./runx.sh # Note that USER has to define the time range for the processed data (e.g., 20200707_074000.000 - 20200707_075000.000) <br>
=> Tadaaaaa – now you process the data automatically and the results are saved to s3://csm-luna/RG-output/

## To change settings of processing got to files:
- yourdir/runx.sh # for example change of prepared images (with or without plotting STA/LTA detections) 
- Docker/scripts-main/job.sh # changes in a list of processing steps to be run in the container
- Docker/scripts-main/settings.py # changes of STA/LTA detection parameters, number of used CPUs
