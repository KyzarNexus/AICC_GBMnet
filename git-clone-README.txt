# To clone this repo, please modify the following line to run on your local machine when cd'd into your repo location of choice. 

$ cd [repo parent directory, the child directory will be called 'GBMnet' and contains the files]
$ git clone [USER]@esplhpccompbio-lv01.csmc.edu:/common/compbiomed-aicampus/team1/biomed_imaging/Code/git-repo/GBMnet.git

# If you add a new file to the repo, add it to git by using the following commands:
$ git add [filename]

# Below is an example of committing changes to your local branch and pushing them to the main remote branch. This should work for both working remotely or in the hpc. 

$ cd GBMnet
$ vim git-clone-README

$ git commit -am 'Fix for README file'
$ git push origin main

# To download and merge changes on the main branch to your local branch, run the following line:
git pull origin main