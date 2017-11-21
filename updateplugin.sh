# script doesn't work as expected
# check other methods later: https://stackoverflow.com/questions/23937436/add-subdirectory-of-remote-repo-with-git-subtree


git remote add -f -t master --no-tags deepdrive-plugin git@github.com:crizCraig/deepdrive-plugin.git
git rm -rf Plugins/DeepDrivePlugin
rm -rf Plugins/DeepDrivePlugin
# git pull deepdrive-plugin master
git read-tree --prefix=Plugins/ -u deepdrive-plugin/master:Plugins/
