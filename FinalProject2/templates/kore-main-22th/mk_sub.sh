#!/usr/bin/bash
#

if [ "$1" != "" ]
then
  echo "working o submission $1 ..."
  rm -rf __pycache__ */__pycache__ */*/__pycache__
  tar -cf - *.py tools config | gzip -c >/tmp/$1.tar.gz
  mkdir ../$1
  gzip -dc /tmp/$1.tar.gz | ( cd ../$1; tar -xvf - )
  cp mk_sub.sh ../$1/.
fi
