## 0. Open a Terminal
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
